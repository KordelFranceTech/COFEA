"""
Classes to create and edit the Python representation of the field.
Contains 'field', 'GridCell', and 'DataPoint' classes.
'DataPoint' has two child classes: Yield and Protein Points.
"""
import re

import pyproj
from shapely.geometry import Polygon, shape, Point
import numpy as np
import pysal as ps
from shapely.ops import cascaded_union, transform
from shapely import wkb
try:
    import ogr
except:
    from osgeo import ogr
import os, math, pandas, time, random, fiona, shortuuid, ast
from copy import deepcopy

from ..fileIO import WKTFiles, ShapeFiles
from ..util import *


class Field:
    """
    @param: field_dict = {
        fld_shp_file = '', 
        yld_file = '', pro_file = '', 
        grid_file = '', applied_file = '', 
        buffer = 0, strip_bool = False, 
        bin_strategy = 'distr', 
        yld_bins = 1, pro_bins = 1, 
        cell_width = 50, cell_height = 100,
        nitrogen_values = [],
        run_ga = False
    """

    def __init__(self, field_dict = None):
        # initialize values
        self.id = 0
        self.conversion_measure = 6076.1154855643 * 60
        self.field_name = ''

        if field_dict:
            self.field_shape_file = field_dict["fld_shp_file"]
            self.yld_file = field_dict["yld_file"]
            self.pro_file = field_dict["pro_file"]
            self.grid_file = field_dict["grid_file"]
            self.as_applied_file = field_dict["applied_file"]
            self.buffer_ = -(field_dict["buffer"]/ self.conversion_measure)
            self.strip_trial = field_dict["strip_bool"]
            self.binning_strategy = field_dict["bin_strategy"]

            self.num_yield_bins = field_dict["yld_bins"]
            self.num_pro_bins = field_dict["pro_bins"]
            self.cell_width = field_dict["cell_width"]/ self.conversion_measure
            self.cell_height = field_dict["cell_height"]/ self.conversion_measure
            self.nitrogen_list = ast.literal_eval(field_dict["nitrogen_values"])
            self.run_ga = field_dict["run_ga"]
            self.base_rate = 120
        
        else:
            self.field_shape_file = None
            self.yld_file = None
            self.pro_file = None
            self.grid_file = None
            self.as_applied_file = None
            self.buffer_ = -(120/self.conversion_measure)
            self.strip_trial = False
            self.binning_strategy = 'distr'

            self.num_yield_bins = 3
            self.num_pro_bins = 1
            self.cell_width = 120/self.conversion_measure
            self.cell_height = 650/self.conversion_measure
            self.nitrogen_list = [40, 60, 80, 100, 120, 150]
            self.base_rate = 120

        self.total_ylpro_bins = self.num_pro_bins * self.num_yield_bins

        self.cell_list = []
        self.protein_bounds = []
        self.yield_bounds = []

        self.field_shape = None
        self.field_shape_buffered = None

        self.yield_points = None
        self.protein_points = None

        self.ylpro_string_matrix = []
        self.ylpro_dict = None
        self.n_dict = {st:idx for idx, st in enumerate(self.nitrogen_list)}
        self.expected_nitrogen_strat, self.max_strat, self.min_strat = 0, 0, 0
        self.max_fertilizer_rate = 0
        self.max_jumps = 0
        self.fixed_costs = 1000  # fixed costs in dollars for field/farm
        self.latlong_crs = 'epsg:4326'
        self.field_crs = ''
        self.aa_crs = ''

    def print_field_info(self):
        print("buffer: ", self.buffer_, "\ncell height: ", self.cell_height, "\ncell width: ", self.cell_width, "\n nitrogen values: ", self.nitrogen_list)
        print("files: \n", self.grid_file, self.field_shape_file)

    def generate_random_id(self):
        self.id = shortuuid.uuid()

    def create_field(self):
        self.generate_random_id()
        self.create_field_shape()

        # read in yield and protein points
        if self.yld_file is not None:
            self.yield_points = Yield(self.yld_file)
        if self.pro_file is not None:
            self.protein_points = Protein(self.pro_file)

        # Set full grid creation in motion. Includes assigning yield and protein to cells.
        if self.grid_file:
            self.cell_list = self.create_grid_from_file()

        else:
            self.cell_list = self.create_grid_for_field()

        print('number of cells: ', len(self.cell_list))
        if self.strip_trial:
            self.cell_list = self.create_strip_trial()

        if self.as_applied_file:
            self.order_cells()

        self.define_binning_bounds()
        self.set_cell_bins()
        self.cell_list = self.assign_nitrogen_distribution()
        self.ylpro_string_matrix = self.create_ylpro_string_matrix()
        self.ylpro_dict = {st: idx for idx, st in enumerate(self.ylpro_string_matrix)}
        self.expected_nitrogen_strat, self.max_strat, self.min_strat = self.calc_expected_bin_strat()
        self.max_fertilizer_rate = max(self.nitrogen_list) * len(self.cell_list)

    def create_field_shape(self):
        # from shapely import speedups
        # speedups.disable()
        # takes all the chunks of a field and then composes them to make a giant
        # polygon out of the field boundaries
        shapefile = fiona.open(self.field_shape_file)
        self.field_crs = shapefile.crs['init']
        polygons = [shape(feature['geometry']) for feature in shapefile]
        self.field_shape = cascaded_union(polygons)
        if self.field_crs.lower() != self.latlong_crs:
            # print('transforming fiel shape: ', self.field_crs, self.latlong_crs)
            project = pyproj.Transformer.from_crs(pyproj.CRS(self.field_crs), pyproj.CRS(self.latlong_crs),
                                                  always_xy=True).transform
            self.field_shape = transform(project, self.field_shape)
        self.field_shape_buffered = self.field_shape.buffer(self.buffer_, cap_style=3)


    def create_grid_for_field(self):
        """
        Returns list of GridCell objects that cover the entire field and fall inside the field boundary.

        FOLLOWING CODE PARTIALLY THANKS TO:
        https://github.com/mlaloux/My-Python-GIS_StackExchange-answers/blob/master/Generate%20grid%20programmatically%20using%20QGIS%20from%20Python.md
        """
        # initialize values
        grid_cell_list = []
        xmin, ymin, xmax, ymax = self.field_shape.bounds
        id = 0

        # Set grid cell dimensions and define grid boundaries
        rows = (ymax - ymin) / self.cell_height
        cols = (xmax - xmin) / self.cell_width
        ring_xleft_origin = xmin
        ring_xright_origin = xmin + self.cell_width
        ring_ytop_origin = ymax
        ring_ybottom_origin = ymax - self.cell_height

        # iterate through number of columns and rows that need to be created based on grid and field dimensions
        for i in np.arange(cols):
            ring_ytop = ring_ytop_origin
            ring_ybottom = ring_ybottom_origin
            for j in np.arange(rows):
                new_grid_cell = GridCell([ring_xleft_origin, ring_ybottom, ring_xright_origin, ring_ytop])
                # Basic grid creation, checks if cell that was created falls into field bounds.
                # Adjusts the current pointer coordinates
                if self.field_shape_buffered.contains(new_grid_cell.small_bounds):
                    new_grid_cell.original_index = id
                    new_grid_cell.sorted_index = id
                    new_grid_cell.set_inner_datapoints(self.yield_points, self.protein_points)
                    grid_cell_list.append(new_grid_cell)
                    id += 1
                ring_ytop = ring_ytop - self.cell_height
                ring_ybottom = ring_ybottom - self.cell_height

            ring_xleft_origin = ring_xleft_origin + self.cell_width
            ring_xright_origin = ring_xright_origin + self.cell_width
            id += 1
        return grid_cell_list

    def create_grid_from_file(self):
        grid_cell_list = []
        wkt = WKTFiles(self.grid_file)
        poly_cells = wkt.grid_read()
        for i, poly in enumerate(poly_cells):
            xmin, ymin, xmax, ymax = poly.bounds
            cell = GridCell([xmin, ymin, xmax, ymax])
            cell.set_inner_datapoints(self.yield_points, self.protein_points)
            cell.sorted_index = i
            cell.original_index = i
            grid_cell_list.append(cell)
        return grid_cell_list

    def create_strip_trial(self):
        """
        Create strip trial from grid cells
        """
        # initialize values
        grid_cell_list = []
        full_strip_set = []
        x_coord = 0
        cell_list = [x for x in self.cell_list]
        for i, cell in enumerate(cell_list):
            if x_coord == 0:
                if self.field_shape.contains(cell.small_bounds):
                    x_coord = cell.bottomleft_x
                    grid_cell_list.append(cell)
            elif i == len(cell_list)-1:
                if self.field_shape.contains(cell.small_bounds):
                    grid_cell_list.append(cell)
                if len(grid_cell_list) != 0:
                    full_strip_set.append(grid_cell_list)
            else:
                # Check if x-coordinate matches for the consecutive cells
                if self.field_shape.contains(cell.small_bounds) \
                        and x_coord == cell.bottomleft_x:
                    # Append cell to current strip if it does
                    grid_cell_list.append(cell)
                elif not self.field_shape.contains(
                        cell.small_bounds) and x_coord == cell.bottomleft_x:
                    if len(grid_cell_list) != 0:
                        full_strip_set.append(grid_cell_list)
                    grid_cell_list = []
                elif x_coord != cell.bottomleft_x:
                    # first_cell_of_strip = True
                    if len(grid_cell_list) != 0:
                        full_strip_set.append(grid_cell_list)
                    grid_cell_list = []
                    x_coord = cell.bottomleft_x
                    if self.field_shape.contains(cell.small_bounds):
                        grid_cell_list.append(cell)

        final_cells = []
        for i, c in enumerate(full_strip_set):
            cell_indeces = [x.original_index for x in c]
            s = [s.true_bounds for s in c]
            strip = cascaded_union(s)
            cell_strip = GridCell(strip.bounds)
            cell_strip.sorted_index = i
            cell_strip.original_index = cell_indeces
            final_cells.append(cell_strip)

        return final_cells

    def order_cells(self, progressbar=None):
        """
        Looks at the as applied map and orders cells accordingly.
        """
        start_time = time.time()
        id_val = 0
        as_applied_points = []
        aa_points = fiona.open(self.as_applied_file)
        self.aa_crs = aa_points.crs['init']
        project = pyproj.Transformer.from_crs(pyproj.CRS(self.aa_crs), pyproj.CRS(self.latlong_crs),
                                              always_xy=True).transform
        if self.aa_crs.lower() != self.latlong_crs:
            latlong_convert = True
        else:
            latlong_convert = False
        filepath, file_extension = os.path.splitext(self.as_applied_file)
        if file_extension == '.shp':
            for i,point in enumerate(list(aa_points)):
                if point['geometry']['type'] == "Point":
                    if i==0:
                        print(".......................................................\nAs Applied type is Point")
                    pt = Point(point['geometry']['coordinates'])
                    if latlong_convert:
                        pt = transform(project, pt)
                    as_applied_points.append(pt)
                elif point['geometry']['type'] == "MultiPoint":
                    if i==0:
                        print(":::::::::::::::::::::::::::::::::::\nAs Applied type is MultiPoint")
                    pt = Point(point['geometry']['coordinates'][0])
                    if latlong_convert:
                        pt = transform(project, pt)
                    as_applied_points.append(pt)
                elif i == 0 and point['geometry']['type'] == "Polygon" and len(point['geometry']['coordinates'][0]) > 1:
                    print('field data line')
                elif i >= 0 and point['geometry']['type'] == "Polygon" and point['properties'].contains('LONGITUDE'):
                    as_applied_points.append(Point(float(point['properties']['LONGITUDE']), float(point['properties']['LATITUDE'])))
                    if i == 0:
                        print("---------------------------------\nAs Applied type is Polygon")
                else:
                    print('OTHER CASES')
                    print(point['geometry']['type'])
                    print(point['geometry']['coordinates'])
                    print(point['properties'])
            print('length of points set ',len(as_applied_points))

        elif file_extension == '.csv':
            data = pandas.read_csv(self.as_applied_file)
            data.columns = data.columns.str.lower()
            data.columns = data.columns.str.strip()
            for i, row in data.iterrows():
                if is_hex(row['geometry']):
                    if i == 0:
                        print("******************************\nAs Applied type is hexagon geometry")
                    wkt_point = wkb.loads(row['geometry'], hex=True)  # ogr.CreateGeometryFromWkb(bts)
                else:
                    if i == 0:
                        print("=======================\nAs Applied type is regular geometry")
                    wkt_point = row['geometry']
                point = ogr.CreateGeometryFromWkt(str(wkt_point))
                as_applied_points.append(Point(point.GetX(), point.GetY()))
            """
            if 'LONGITUDE' in data.columns:
                latlong = True
                lon = data['LONGITUDE'].tolist()
                lat = data['LATITUDE'].tolist()
                for i, l in enumerate(lon):
                    points.append([Point(float(l), float(lat[i])),i])
            else:
                x = data['X'].tolist()
                y = data['Y'].tolist()
                for i, l in enumerate(x):
                    points.append([Point(float(l), float(y[i])), i])
            """
        else:
            print("As applied file is not a csv or shp file")

        shape_ = self.field_shape_buffered
        good_as_applied_points = [point for point in as_applied_points if shape_.contains(point)]
        i = 0

        ordered_cells = [x for x in self.cell_list]
        counter = 0
        for p, point in enumerate(good_as_applied_points):
            while i < len(ordered_cells):
                cell = next(x for x in self.cell_list if x.original_index == ordered_cells[i].original_index)
                bounds = cell.true_bounds
                if bounds.contains(point) and cell.is_sorted:
                    og_index = cell.original_index
                    if progressbar is not None:
                        progressbar.update_progress_bar("Ordering cell... " + str(og_index), (counter / len(self.cell_list)) * 100)
                    cell.sorted_index = id_val
                    cell.is_sorted = True
                    id_val = id_val + 10
                    del ordered_cells[i]
                    counter += 1
                    break
                i = i + 1
            i = 0
        unordered = [[i, cell] for i, cell in enumerate(self.cell_list) if not cell.is_sorted]

        for t in unordered:
            i = t[0]
            for j, next_cell in enumerate(self.cell_list[i + 1:]):
                if next_cell.is_sorted:
                    self.cell_list[i].sorted_index = next_cell.sorted_index - 1 - j
                    self.cell_list[i].is_sorted = True

        elapsed_time = time.time() - start_time
        print('\ntime to order cells: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    def define_binning_bounds(self, data_to_bin_on='cells'):
        """
        @param: binning_strategy: binning using the number of values ('distr') or the values themselves ('yld_pro')
        @param: data_to_bin_on: binning on 'datapoints' or on 'cells'
        """

        if data_to_bin_on == 'cells':
            # get cell values for yield and protein and sort them to identify min and max values.
            yield_values = [cell.yield_ for cell in self.cell_list]
            yield_values.sort()
            if self.cell_list[0].pro_ != 0:
                pro_values = [cell.pro_ for cell in self.cell_list]
                pro_values.sort()

            # This looks at the yield and protein values per cell to determine bin bounds.
            # E.g.: There are 9 cells with ordered yields [3,4,4,9,9,9,12,14,15]
            if self.binning_strategy == 'yld_pro':
                # Take the min = 3, max = 16, number of yield bins = 3
                # step_size = (15-3) / 3 = 4
                step_size = (yield_values[-1] - yield_values[0]) / self.num_yield_bins
                # Upper bounds are calculated based on yield step size
                # 1st bound: 3 + 4*1, 2nd bound: 3 + 4*2, 3rd bound: 3 + 4*3
                # --> 7, 11, 15
                self.yield_bounds = [yield_values[0] + (step_size * (i + 1)) for i in range(
                    self.num_yield_bins)]
                self.yield_bounds[-1] = self.yield_bounds[-1] + 1  # Because 15 needs to be included, final bound + 1
                # Repeat for protein
                if self.cell_list[0].pro_ != 0:
                    pro_steps = (pro_values[-1] - pro_values[0]) / self.num_pro_bins
                    self.protein_bounds = [pro_values[0] + (pro_steps * (i + 1)) for i in range(
                        self.num_pro_bins)]
                    self.protein_bounds[-1] = self.protein_bounds[-1] + 1

            # This looks at the number of cells to determine bin bounds.
            # E.g.: There are 9 cells with ordered yields [3,4,4,9,9,9,12,14,15]
            elif self.binning_strategy == 'distr':
                cell_max = len(self.cell_list)
                # step size = 9 / 3 = 3
                step_size = cell_max / self.num_yield_bins
                # Upper bounds are calculated by dividing yields on their cell index
                # 1st index: (3 * (0+1)) - 1 = 2, 2nd index: (3 * (1+1)) - 1 = 5, 3rd index: (3 * (2+1)) - 1 = 8
                # --> 4, 9, 15 are the corresponding yield value bounds
                self.yield_bounds = [yield_values[round((step_size * (i + 1))) - 1] for i in range(
                    self.num_yield_bins)]
                self.yield_bounds[-1] = self.yield_bounds[-1] + 1
                # Repeat for protein
                if self.cell_list[0].pro_ != 0:
                    pro_steps = cell_max / self.num_pro_bins
                    self.protein_bounds = [pro_values[round((pro_steps * (i + 1))) - 1] for i in range(
                        self.num_pro_bins)]
                    self.protein_bounds[-1] = self.protein_bounds[-1] + 1
        elif data_to_bin_on == 'datapoints':
            if self.binning_strategy == 'yld_pro':
                self.yield_bounds = ps.esda.mapclassify.Equal_Interval(np.asarray(self.yield_points.datapoints),
                                                                       k=self.num_yield_bins).bins.tolist()
                self.yield_bounds[-1] = self.yield_bounds[-1] + .1
                if isinstance(self.protein_points.datapoints, pandas.DataFrame):
                    self.protein_bounds = ps.esda.mapclassify.Equal_Interval(np.asarray(self.protein_points.datapoints),
                                                                             k=self.num_pro_bins).bins.tolist()
                    self.protein_bounds[-1] = self.protein_bounds[-1] + .1
            elif self.binning_strategy == 'distr':
                self.yield_bounds = ps.esda.mapclassify.Quantiles(np.asarray(self.yield_points.datapoints),
                                                                  k=self.num_yield_bins).bins.tolist()
                self.yield_bounds[-1] = self.yield_bounds[-1] + 1
                if isinstance(self.protein_points.datapoints, pandas.DataFrame):
                    self.protein_bounds = ps.esda.mapclassify.Quantiles(np.asarray(self.protein_points.datapoints),
                                                                        k=self.num_pro_bins).bins.tolist()
                    self.protein_bounds[-1] = self.protein_bounds[-1] + .1

    def set_cell_bins(self):
        for cell in self.cell_list:
            if cell.yield_ != 0:
                for i in range(len(self.yield_bounds)):
                    if cell.yield_ <= self.yield_bounds[i]:
                        cell.yield_bin = i + 1
                        break
            if cell.pro_ != 0:
                for i in range(len(self.protein_bounds)):
                    if cell.pro_ <= self.protein_bounds[i]:
                        cell.pro_bin = i + 1
                        break

            if cell.yield_bin == -1:
                if cell.yield_ != 0:
                    bin_list = [x for x in range(1, self.num_yield_bins + 1)]
                    cell.yield_bin = random.choice(bin_list)
                else:
                    cell.yield_bin = 1
            if cell.pro_bin == -1:
                if cell.pro_ != 0:
                    bin_list = [x for x in range(1, self.num_pro_bins + 1)]
                    cell.pro_bin = random.choice(bin_list)
                else:
                    cell.pro_bin = 1

    def assign_random_nitrogen_binned(self, cells_to_assign):
        # takes a set of cells and assigns equal-ish distribution of nitrogen
        i = 0
        random.shuffle(self.nitrogen_list)
        for cell in cells_to_assign:
            cell.nitrogen = self.nitrogen_list[i % len(self.nitrogen_list)] #random.randint(0, len(self.nitrogen_list)-1)
            i = i + 1

    def assign_nitrogen_distribution(self):
        cell_list = [c for c in self.cell_list]
        if self.num_yield_bins == 1 or self.yield_points is None:
            random.shuffle(self.nitrogen_list)
            for cell in cell_list:
                cell.nitrogen = self.nitrogen_list[random.randint(0, len(self.nitrogen_list)-1)]

        else:
            # stratifies combinations of different bins and then assigns nitrogen rates
            for i in range(1, self.num_yield_bins + 1):
                for j in range(1, self.num_pro_bins + 1):
                    cells_in_curr_ylpro_combo = []
                    for cell in cell_list:
                        if cell.yield_bin == i and cell.pro_bin == j:
                            cells_in_curr_ylpro_combo.append(cell)
                    self.assign_random_nitrogen_binned(cells_in_curr_ylpro_combo)
        return cell_list

    def create_ylpro_string_matrix(self):
        i = 1
        j = 0
        index = 0
        ylpro_string_matrix = np.zeros(self.total_ylpro_bins, dtype=int)
        while i <= self.num_yield_bins:
            while j < self.num_pro_bins:
                ylpro_string_matrix[index] = str(i) + str(j)
                index = index + 1
                j = j + 1
            i = i + 1
            j = 0
        return ylpro_string_matrix.tolist()

    def calc_expected_bin_strat(self):
        """
        Calculates expected number of cells for each nitrogen bin.
        Returns expected stratification, maximum and minimum potential stratification counts
        """
        num_nitrogen = len(self.nitrogen_list)
        num_cells = len(self.cell_list)
        ideal_nitrogen_cells = num_cells / num_nitrogen

        cell_strat = int(ideal_nitrogen_cells / self.total_ylpro_bins)
        min_strat = 0
        expected_bin_strat = []
        for i in range(1, self.num_yield_bins + 1):
            for j in range(1, self.num_pro_bins + 1):
                cells_in_bin = sum(cell.yield_bin == i and cell.pro_bin == j for cell in self.cell_list)
                cell_strat_min = cells_in_bin % num_nitrogen
                spread_of_min_strat_per_bin = cell_strat_min / num_nitrogen
                strat = cells_in_bin / num_nitrogen
                expected_bin_strat.append([strat, spread_of_min_strat_per_bin])
                min_strat += cell_strat_min
        max_strat = 2 * cell_strat * self.total_ylpro_bins * (num_nitrogen - 1)

        return expected_bin_strat, max_strat, min_strat

    def create_strip_groups(self, overlap=False, overlap_ratio=0.1):
        cell_indeces = self.create_strip_trial()
        factors = []
        single_cells = []
        sum = 0
        for j, strip in enumerate(cell_indeces):
            if len(strip.original_index) == 1:
                single_cells.append(sum)
            else:
                factors.append([i + sum for i, og in enumerate(strip.original_index)])
            sum = sum + len(strip.original_index)
        if single_cells:
            factors.append(single_cells)
        if overlap:
            new_factors = []
            nr_of_cells = [int(np.ceil(len(f) * overlap_ratio)) for f in factors]
            for i, f in enumerate(factors):
                if i != len(factors) -1:
                    nf = []
                    for j in range(1, nr_of_cells[i]+1):
                        nf.append(f[-j])
                    for j in range(nr_of_cells[i+1]):
                        nf.append(f[j])
                    new_factors.append(nf)
            factors.extend(new_factors)
        print(len(factors))
        return factors


class GridCell:
    def __init__(self, coordinates, nitrogen=-1):
        self.is_sorted = False
        self.original_index = -1
        self.sorted_index = -1
        self.conversion_measure = 6076.1154855643 * 60

        # coordinates and polygon definitions
        self.bottomleft_x = coordinates[0]
        self.bottomleft_y = coordinates[1]
        self.upperright_x = coordinates[2]
        self.upperright_y = coordinates[3]
        self.coordinates = coordinates
        self.true_bounds = Polygon(
            [(coordinates[0], coordinates[1]), (coordinates[2], coordinates[1]), (coordinates[2], coordinates[3]),
             (coordinates[0], coordinates[3])])
        self.small_bounds = Polygon(
            [(coordinates[0] + 1.5 / 364567.2, coordinates[1] + 1.5 / 364567.2),
             (coordinates[2] - 1.5 / 364567.2, coordinates[1] + 1.5 / 364567.2),
             (coordinates[2] - 1.5 / 364567.2, coordinates[3] - 1.5 / 364567.2),
             (coordinates[0] + 1.5 / 364567.2, coordinates[3] - 1.5 / 364567.2)])
        self.gridcell_size = self.true_bounds.area*self.conversion_measure*self.conversion_measure
        self.folium_bounds = [[coordinates[1], coordinates[0]], [coordinates[3], coordinates[2]]]

        # values within cell
        self.yield_points = []
        self.protein_points = []
        self.yield_ = -1
        self.pro_ = -1

        # values to be assigned
        self.yield_bin = -1
        self.pro_bin = -1
        self.nitrogen = nitrogen

    def to_dict(self):
        return {
            'protein': self.pro_,
            'yield': self.yield_,
            'nitrogen': self.nitrogen,
            'index': self.sorted_index
        }

    def set_inner_datapoints(self, yld=None, pro=None):
        # determine yield and protein points that lie within cell
        if yld is not None:
            self.yield_points = yld.set_datapoints(self)
            if len(self.yield_points) !=0:
                self.set_avg_yield()
            else:
                self.yield_ = 0
        if pro is not None:
            self.protein_points = pro.set_datapoints(self)
            if len(self.protein_points) != 0:
                self.set_avg_protein()
            else:
                self.pro_ = 0

    def set_avg_yield(self):
        self.yield_ = np.mean(self.yield_points[:, 2])

    def set_avg_protein(self):
        self.pro_ = np.mean(self.protein_points[:, 2])


class DataPoint:
    def __init__(self, filename):
        self.filename = filename
        self.datapoints = []
        self.id = -1
        self.gridcell = -1

    def create_dataframe(self, datatype=''):
        """
        Create a dataframe containing yield or protein values and their coordinates in the form:
        [ [X1, Y1, idx, value1], [X2, Y2, idx, value2], ..., [Xn, Yn, idx, valuen] ]
        for all n datapoints in the file with numeric values.
        """

        filepath, file_extension = os.path.splitext(str(self.filename))
        datapoints = []
        if 'shp' in file_extension:
            datapoints = ShapeFiles(self.filename).read_shape_file(datatype)
        else:
            if isinstance(self.filename, pandas.DataFrame):
                file = self.filename
            elif 'csv' in file_extension:
                file = pandas.read_csv(self.filename)
            for i, row in file.iterrows():
                datapoint = []
                if is_hex(row['geometry']):
                    wkt_point = wkb.loads(row['geometry'], hex=True)  # ogr.CreateGeometryFromWkb(bts)
                elif "," in row['geometry']:
                    point = ''.join(re.findall("[0-9.,-]", row['geometry']))
                    wkt_point = "POINT("+re.sub(",", " ", point)+")"
                elif "point" in row['geometry'].lower:
                    wkt_point = row['geometry']
                elif row['lon']:
                    wkt_point = "POINT(" + str(row['lon']) + " " + str(row['lat']) + ")"
                point = ogr.CreateGeometryFromWkt(str(wkt_point))
                datapoint.append(point.GetX())
                datapoint.append(point.GetY())
                datapoint.append(i)

                if row[datatype] != 'NONE' and not math.isnan(row[datatype]):
                    datapoint.append(row[datatype])
                    datapoints.append(datapoint)
        return np.array(datapoints)

    def set_datapoints(self, gridcell):
        points_in_cell = self.datapoints[(self.datapoints[:, 0] >= gridcell.bottomleft_x) &
                                         (self.datapoints[:, 0] <= gridcell.upperright_x) &
                                         (self.datapoints[:, 1] <= gridcell.upperright_y) &
                                         (self.datapoints[:, 1] >= gridcell.bottomleft_y)]
        # for pt in points_in_cell:
        #     pt.gridcell = gridcell.original_index
        return points_in_cell
        # if not math.isnan(yieldInSquare[:, 2].mean()) and len(yieldInSquare) != 0:
        #     gridcell.yield_ = yieldInSquare[:, 2].mean()
        # else:
        #     gridcell.yield_ = 0


class Yield(DataPoint):
    def __init__(self, filename):
        super().__init__(filename)
        self.datapoints = self.create_dataframe('yld')


class Protein(DataPoint):
    def __init__(self, filename):
        super().__init__(filename)
        self.datapoints = self.create_dataframe('pro')