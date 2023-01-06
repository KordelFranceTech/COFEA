"""
Classes to create and edit the Python representation of the field.
Contains 'field', 'GridCell', and 'DataPoint' classes.
'DataPoint' has two child classes: Yield and Protein Points.

A coordinate system in U.S. survey feet may be converted to a coordinate system in meters by scaling the system to a scale factor of 0.304800609601. 
An exact conversion can be accomplished by multiplying U.S. survey feet by the fraction 1200/3937.

A coordinate system in meters may be converted to a coordinate system in U.S. survey feet by scaling the system to a scale factor of 3.28083333333. 
An exact conversion can be accomplished by multiplying meters by the fraction 3937/1200.
"""
from lib2to3.pytree import convert
from platform import win32_edition
import re
from select import select
from tkinter import Grid

from shapely.geometry import Polygon, shape, Point, LineString, MultiPolygon, MultiLineString
from shapely.ops import split
import numpy as np
import pysal as ps
from shapely.ops import cascaded_union, transform, unary_union
from shapely import wkb
from shapely.affinity import rotate
from copy import deepcopy
import os, math, pandas, time, ogr, random, fiona, shortuuid, itertools, haversine, pyproj

from utilities.fileIO import WKTFiles, ShapeFiles
from utilities.util import *


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
        fertilizer_values = [],
        run_ga = False
    """

    def __init__(self, id=0, field_dict=None):
        # initialize values
        self.id = id
        self.conversion_measure = 6076.1154855643*60  # convert feet to Nautical degrees by dividing by this number
        self.field_name = ''

        if field_dict:
            """
            Data files
            """
            self.field_shape_file = field_dict["fld_shp_file"] # boundary file to get field outline
            self.yld_file = field_dict["yld_file"] # file containing yield data points, used to discretize into bins for stratification of nitrogen rates
            self.pro_file = field_dict["pro_file"] # file containing protein data points of wheat, also used for stratifacation purposes
            self.ab_line_file = field_dict["abline"] # ab-line file, line connecting starting point A of the farming machine to the end of the first "column" at point B.
            # determines orientation of the field based on this ab-line, i.e., north to south or east to west, or at an angle.
            self.grid_file = field_dict["grid_file"] # If there is an existing grid, read in the existing file
            self.as_applied_file = field_dict["applied_file"]  # Actual application of fertilizer/seeding rate
            
            """
            Prescription info
            """
            self.buffer_ = -float(field_dict["buffer"]) # How many feet from the boundary should the prescription grid start
            self.strip_trial = field_dict["strip_bool"] # ability to set up a strip trial instead of a grid trial.

            """
            Stratification info
            previous year yield for each gridcell -> stratify nitrogen rates across yield bins
            6 gridcells -> 60-80-90-80-40-100 --> order yield values -> 40-60-80-80-90-100
            Binning strategy determines where to split yield values to creates yield bins.
            For example, 3 yield bins: low, medium, high yield bins. 
            low yield bin: 40-60, medium: 80-80, high: 90-100
            """
            self.binning_strategy = field_dict["bin_strategy"]
            self.num_yield_bins = field_dict["yld_bins"]
            self.num_pro_bins = field_dict["pro_bins"]

            """
            To better fill out the field, give min-max length options to look between
            """
            self.cell_length_min = float(field_dict["cell_length_min"])
            self.cell_length_max = float(field_dict["cell_length_max"])
            self.cell_width = float(field_dict["cell_width"])

            """
            If there is no AB line, the angle can be specified manually in degrees
            """
            self.degrees = field_dict["degrees"]

            """
            Whatever is being applied to the field for the experimental trial.
            For example: Nitrogen, seeding rate (this is not fertilizer, but it works too), any other fertilizer
            """
            if field_dict["fertilizer_name_1"] != '':
                self.fertilizer_name_1 = field_dict["fertilizer_name_1"]
            else:
                self.fertilizer_name_1 = 'value1'
            print('fertilizer values: ', str(field_dict["fertilizer_values_1"]))
            self.fertilizer_list_1 = [float(re.sub('[^\d\.]', '', x)) for x in str(field_dict["fertilizer_values_1"]).strip('[]').replace(' ', '').split(',')]
            self.base_rate_1 = float(field_dict["base_rate_1"])
            if field_dict["fertilizer_name_2"] != '':
                self.fertilizer_name_2 = field_dict["fertilizer_name_2"]
            else:
                self.fertilizer_name_2 = 'value2'
            if field_dict["fertilizer_values_2"] != '[]':
                self.fertilizer_list_2 = [float(re.sub('[^\d\.]', '', x)) for x in str(field_dict["fertilizer_values_2"]).strip('[]').replace(' ', '').split(',')]
            else:
                self.fertilizer_list_2 = []
            self.base_rate_2 = float(field_dict["base_rate_2"])

            self.run_ga = field_dict["run_ga"]  # not really used any more with the latin square trials taking over
            self.latin_square = field_dict["latin_square"]  # default, set to True
        
        else:
            """
            Initialization of empty values if a dictionary was not sent through to initialize the values
            """
            self.field_shape_file = None
            self.yld_file = None
            self.pro_file = None
            self.ab_line_file = None
            self.grid_file = None
            self.as_applied_file = None
            self.buffer_ = -45
            self.strip_trial = False

            self.binning_strategy = 'distr'
            self.num_yield_bins = 1
            self.num_pro_bins = 1

            self.cell_width = 150
            self.cell_length_min = 300
            self.cell_length_max = 350
            self.degrees = 0

            self.fertilizer_name_1 = ""
            self.fertilizer_list_1 = [20,40,60,80, 100]
            self.base_rate_1 = 50

            self.fertilizer_name_2 = ""
            self.fertilizer_list_2 = []
            self.base_rate_2 = 0

            self.run_ga = False
            self.latin_square = False

        """
        Variables filled as the prescription is created
        """
        self.columns = []
        self.cell_list = [] # list of GridCell objects
        self.cell_polys = [] # list of shapely polygon objects representing the gridcells
        self.protein_bounds = []
        self.yield_bounds = []

        self.field_shape = None # field shape polygon read in from boundary (bbox) file
        self.field_shape_buffered = None # polygon with buffer zone removed
        self.angle = 0 
        self.ab_line = None
        self.rotation_center = None # rotation center for correct grid rotation
        self.smallest_cell_area = self.cell_width*self.cell_length_min

        self.yield_points = None
        self.protein_points = None

        self.latlong_crs = 'epsg:4326'
        self.us_feet_crs = 'epsg:2263'
        self.latitude = 0
        self.field_crs = ''
        self.aa_crs = ''
        self.second_item_in_list_flag = ''
        self.fixed_costs = 1000 

    def print_field_info(self):
        print("buffer: ", self.buffer_, "\ncell height: ", self.cell_length_min, "\ncell width: ", self.cell_width, "\n applicator values: ", self.fertilizer_list_1)
        print("files: \n", self.grid_file, self.field_shape_file)

    def generate_random_id(self):
        self.id = shortuuid.uuid()
    
    def create_field(self):
        self.generate_random_id()

        # Set degree of rotation
        if self.ab_line_file is not None:
            self.angle = self.calculate_rotation_angle()
        elif self.degrees != 0:
            self.angle = float(self.degrees)

        # Read in field shape file and create buffered field shape
        self.create_field_shape()

        # read in yield and protein points and save them as Yield and Protein instances of the DataPoint class
        if self.yld_file is not None:
            self.yield_points = Yield(self.yld_file)
        if self.pro_file is not None:
            self.protein_points = Protein(self.pro_file)

        # Set full grid creation in motion. Includes assigning yield and protein to cells.
        if self.grid_file:
            self.cell_list = self.create_grid_from_file()
        else:
            if self.cell_length_min == self.cell_length_max:
                self.cell_list = self.create_basic_grid_for_field()
            elif self.strip_trial:
                self.cell_length_min = 100
                self.create_strip_trial()
            else:
                # Create columns
                if self.angle != 0:
                    rotated_field = rotate(self.field_shape_buffered, -self.angle, origin=self.rotation_center)
                    xmin, ymin, xmax, ymax = rotated_field.bounds
                    self.columns = self.recursive_split_poly(rotated_field, ymin, ymax)
                else:
                    xmin, ymin, xmax, ymax = self.field_shape_buffered.bounds
                    self.columns = self.recursive_split_poly(self.field_shape_buffered, ymin, ymax)
                self.create_variable_length_grid()

        if self.latin_square:
            self.create_non_diagonal_latin_square()
        else:
            self.assign_applicator_distribution()

    def calculate_rotation_angle(self):
        '''
        Calculate by how much the field needs to be rotated based on AB line input.
        '''
        shapefile = fiona.open(self.ab_line_file)
        abline_crs = shapefile.crs #['init']
        try:
            abline_crs = abline_crs['init']
        except KeyError:
            abline_crs = "epsg:4326"
            print("No crs value found")
        ab_line = [shape(feature['geometry']) for feature in shapefile][0]
        # if ab_line.geom_type != "LineString":
        #     raise ABLineFileError("Angle could not be calculated from the provided AB line.")
        if abline_crs.lower() != self.latlong_crs:
            project = pyproj.Transformer.from_crs(pyproj.CRS(abline_crs), pyproj.CRS(self.latlong_crs),
                                                  always_xy=True).transform
            ab_line = transform(project, ab_line)
        pt1 = ab_line.coords[0]
        pt2 = ab_line.coords[1]
        x_diff = pt2[0] - pt1[0]
        y_diff = pt2[1] - pt1[1]
        angle_in_radians = math.atan2(y_diff, x_diff)
        angle_in_degrees = math.degrees(angle_in_radians)
        return angle_in_degrees

    def create_field_shape(self):
        '''
        Read in the boundary shape file.
        create regular field Polygon and buffered field Polygon.
        Set rotation origin to center of buffered polygon for consistent rotation
        '''
        from shapely import speedups
        speedups.disable()
        # takes all the chunks of a field and then composes them to make a giant
        # polygon out of the field boundaries
        shapefile = fiona.open(self.field_shape_file)
        #[print(feature['geometry']) for feature in shapefile]
        field_crs = shapefile.crs #['init']
        try:
            self.field_crs = field_crs['init']
        except KeyError:
            self.field_crs = "epsg:4326"
            print("No crs value found")
        polygons = [shape(feature['geometry']) for feature in shapefile]
        # if polygons[0].geom_type != "Polygon" and polygons[0].geom_type != "MultiPolygon":
        #     raise BoundaryFileError("Could not read field shape from the provided files.")
        field_shape = unary_union(polygons)
        self.latitude = field_shape.centroid.coords[0][1]
        self.field_shape = field_shape.buffer(0)
        if self.field_crs.lower() != self.latlong_crs:
            print('transforming field shape: ', self.field_crs, self.latlong_crs)
            project = pyproj.Transformer.from_crs(pyproj.CRS(self.field_crs), pyproj.CRS(self.latlong_crs),
                                                  always_xy=True).transform
            self.field_shape = transform(project, self.field_shape)
        self.field_shape_buffered = self.field_shape.buffer(self.buffer_/self.conversion_measure, cap_style=3)
        if self.angle != 0:
            self.rotation_center = self.field_shape_buffered.centroid
    
    def select_plot_length(self, col, min, max):
        '''
        Recursively find best fitting plot length in provided value range.
        Recursion is divide and conquer based to speed up the process.
        Based on covered area of the plots.
        '''
        middle = min+round(abs(max-min)/2)
        #print("minmaxmid", min, max, middle)
        if min != middle:
            area_covered_1, min1, cells_1 = self.select_plot_length(col, min, middle)
            area_covered_2, min2, cells_2 = self.select_plot_length(col, middle, max)
            #print("competing: ", area_covered_1, area_covered_2)
            if area_covered_1 > area_covered_2:
                winner = min1
                cells = cells_1
                area_covered = area_covered_1
            else:
                winner = min2
                cells = cells_2
                area_covered = area_covered_2
            #print("won: ",area_covered)
            return area_covered, winner, cells
        else:
            if self.angle != 0:
                field_shape = rotate(self.field_shape_buffered, -self.angle, origin=self.rotation_center)
            else:
                field_shape = self.field_shape_buffered
            converted_min = min/self.conversion_measure
            number_of_cells = np.floor(col.shorter_length/ converted_min) + 2 # (min/364567.2)
            xmin, ymin, xmax, ymax = col.polygon.bounds
            ymin_cell = ymin
            cells = []
            for i in np.arange(number_of_cells):
                ymax_cell = ymin_cell + converted_min
                poly = Polygon([(xmax, ymin_cell), (xmin, ymin_cell), (xmin, ymax_cell), (xmax,ymax_cell)]) # 5
                #small_poly = Polygon([(xmax- five_feet, ymin_cell+(five_feet)), (xmin+(five_feet), ymin_cell+(five_feet)), (xmin+(5), ymin_cell-(five_feet)), (xmax-(five_feet),ymin_cell-(five_feet))])
                if field_shape.contains(poly):
                    cells.append(poly)
                ymin_cell = ymax_cell #+  converted_min #/364567.2
            return MultiPolygon(cells).area*self.conversion_measure , min, cells
    
    def recursive_split_poly(self, poly_to_split, ymin_fld, ymax_fld, init=True):
        '''
        Recursively split the field into strips.
        '''
        xmin, ymin, xmax, ymax = poly_to_split.bounds
        if init:
            xmin = xmin + (10/self.conversion_measure) 
            xmax = xmax - (10/self.conversion_measure) 
        total_width = Point(xmin, ymin).distance(Point(xmax, ymin))
        column_pt = xmin + self.cell_width/self.conversion_measure
        if np.floor(total_width*self.conversion_measure) <= (self.cell_width): 
            #print(poly_to_split.area*self.conversion_measure*self.conversion_measure , self.smallest_cell_area)
            if poly_to_split.area*self.conversion_measure*self.conversion_measure < self.smallest_cell_area:
                return []
            else:
                print(poly_to_split)
                return [Column(poly_to_split, 1, xmin=xmin, xmax=column_pt, field=self.field_shape)]
        else:
            new_column = []
            line = LineString([Point(column_pt, ymin_fld), Point(column_pt, ymax_fld)])
            if poly_to_split.geom_type != 'GeometryCollection': 
                split_poly = list(split(poly_to_split, line))
                if split_poly:                
                    for c in split_poly:
                        new_column.extend(self.recursive_split_poly(c, ymin_fld, ymax_fld, False))
            else:
                for c in poly_to_split:
                    if c:
                        new_column.extend(self.recursive_split_poly(c, ymin_fld, ymax_fld, False))
            return new_column
            
    def create_variable_length_grid(self):
        '''
        Based on final columns and best area coverage.
        Convert cells to GridCell objects.
        '''
        total_cells = []
        for i, col in enumerate(self.columns):
            area, cell_length, cells = self.select_plot_length(col, self.cell_length_min-1, self.cell_length_max+1)
            total_cells.extend(cells)
        self.cell_polys = MultiPolygon(total_cells)
        for cell in total_cells:    
            new_cell = GridCell(cell.bounds)
            new_cell.set_polygon_bounds(cell)
            self.cell_list.append(new_cell)
        if self.angle != 0:
            self.cell_list = self.rotate_grid_cells(self.cell_polys , self.cell_list)
            

    def create_basic_grid_for_field(self):
        """
        Classic Grid creation, no variable length plots.
        Returns list of GridCell objects that cover the entire field and fall inside the field boundary.

        FOLLOWING CODE PARTIALLY THANKS TO:
        https://github.com/mlaloux/My-Python-GIS_StackExchange-answers/blob/master/Generate%20grid%20programmatically%20using%20QGIS%20from%20Python.md
        """

        # initialize values
        grid_cell_list = []
        if self.angle != 0:
            xmin, ymin, xmax, ymax = rotate(self.field_shape, self.angle, origin=self.rotation_center).bounds
            rotated = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)])
            xmin, ymin, xmax, ymax = self.field_shape.bounds
            field = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)])
            xmin, ymin, xmax, ymax = unary_union(MultiPolygon([rotated, field])).bounds
            print(xmin, ymin, xmax, ymax)
        else:
            xmin, ymin, xmax, ymax = self.field_shape.bounds
        id = 0
        cell_length = self.cell_length_min/self.conversion_measure #/364567.2
        cell_width = self.cell_width/self.conversion_measure
        # Set grid cell dimensions and define grid boundaries
        if self.angle == 0:
            rows = np.ceil((ymax - ymin) / cell_length)
            cols = np.ceil((xmax - xmin) / cell_width)
        else:
            rows = np.ceil((ymax - ymin) / cell_length)*1.5
            cols = np.ceil((xmax - xmin) / cell_width)*1.5
        ring_xleft_origin = xmin
        ring_xright_origin = xmin + cell_width
        ring_ytop_origin = ymax
        ring_ybottom_origin = ymax - cell_length

        polygon_list = []
        final_cell_list = []

        ab_line_top_point = None
        ab_line_bottom_point = None
        first_column = True
        first_row = True

        # iterate through number of columns and rows that need to be created based on grid and field dimensions
        for i in np.arange(cols):    
            ring_ytop = ring_ytop_origin
            ring_ybottom = ring_ybottom_origin
            for j in np.arange(rows):
                new_grid_cell = GridCell([ring_xleft_origin, ring_ybottom, ring_xright_origin, ring_ytop])
                polygon_list.append(new_grid_cell.true_bounds)
                if self.angle != 0:
                    grid_cell_list.append(new_grid_cell)
                elif self.angle == 0 and self.field_shape_buffered.contains(new_grid_cell.true_bounds):
                    if first_column and first_row:
                        ab_line_top_point = Point((new_grid_cell.bottomleft_x+new_grid_cell.upperright_x)/2, new_grid_cell.upperright_y)
                        first_row = False
                    new_grid_cell.original_index = id
                    new_grid_cell.sorted_index = id
                    new_grid_cell.set_inner_datapoints(self.yield_points, self.protein_points)
                    new_grid_cell.folium_bounds = [[new_grid_cell.bottomleft_y, new_grid_cell.bottomleft_x], [new_grid_cell.upperright_y, new_grid_cell.upperright_x]]
                    if j-1 == rows or not self.field_shape_buffered.contains(Polygon( 
                        [(ring_xleft_origin, (ring_ybottom - (self.cell_length_min)) ),
                        (ring_xright_origin, (ring_ybottom - (self.cell_length_min))),
                        (ring_xright_origin , (ring_ytop - (self.cell_length_min)) ),
                        (ring_xleft_origin , (ring_ytop - (self.cell_length_min)))])):
                        new_grid_cell.is_last_in_col = True
                        if first_column:
                            ab_line_bottom_point = Point((new_grid_cell.bottomleft_x+new_grid_cell.upperright_x)/2, new_grid_cell.bottomleft_y)
                            first_column = False
                    final_cell_list.append(new_grid_cell)
                    self.cell_polys.append(new_grid_cell.true_bounds)
                    id += 1
                ring_ytop = ring_ytop - cell_length
                ring_ybottom = ring_ybottom - cell_length          
            ring_xleft_origin = ring_xleft_origin + cell_width
            ring_xright_origin = ring_xright_origin + cell_width
            id += 1
        id = 0
        if self.angle == 0:
            self.ab_line = LineString([ab_line_top_point, ab_line_bottom_point])
        else:
            #final_cell_list = grid_cell_list
            final_cell_list = self.rotate_grid_cells(polygon_list, grid_cell_list, og = True)
        return final_cell_list
    
    def rotate_grid_cells(self, polygon_list, grid_cell_list = [], og = False):
        '''
        Rotate basic grid cells
        '''
        id = 0
        count = 0
        first_row = True
        first_column = True
        final_cell_list = []
        ab_line_top_point = None
        ab_line_bottom_point = None
        multipolygon = MultiPolygon(polygon_list)
        rotated = rotate(multipolygon, self.angle, origin=self.rotation_center)
        self.cell_polys = []
        for rot, cell in zip(rotated, grid_cell_list):
            # Basic grid creation, checks if cell that was created falls into field bounds.
            # Adjusts the current pointer coordinates
            og_bounds = cell.true_bounds
            cell.set_polygon_bounds(rot)
            cell.original_index = id
            cell.sorted_index = id
            cell.set_inner_datapoints(self.yield_points, self.protein_points)
            cell.original_bounds = og_bounds
            if og:
                if self.field_shape_buffered.contains(cell.true_bounds):
                    final_cell_list.append(cell)
                    self.cell_polys.append(rot)
                    if first_row:
                        ab_line_top_point = Point((cell.bottomleft_x+cell.upperright_x)/2, cell.upperright_y)
                        first_row = False
            else:
                if first_row:
                    ab_line_top_point = Point((cell.bottomleft_x+cell.upperright_x)/2, cell.upperright_y)
                    first_row = False
                final_cell_list.append(cell)
                self.cell_polys.append(rot)
            if count != len(grid_cell_list)-1:
                next_cell = grid_cell_list[count+1]
                next_cell.set_polygon_bounds(rotated[count+1])
                if not self.field_shape_buffered.contains(next_cell.true_bounds):
                    cell.is_last_in_col = True
                    if first_column:
                        ab_line_bottom_point = Point((cell.bottomleft_x+cell.upperright_x)/2, cell.bottomleft_y)
                        first_column = False
                elif cell.bottomleft_y != next_cell.upperright_y:
                    cell.is_last_in_col = True
                    if first_column:
                        ab_line_bottom_point = Point((cell.bottomleft_x+cell.upperright_x)/2, cell.bottomleft_y)
                        first_column = False
            else:
                cell.is_last_in_col = True
            id += 1
            count+=1
        print(ab_line_top_point)
        self.ab_line = LineString([ab_line_top_point, ab_line_bottom_point])
        return final_cell_list


    def create_grid_from_file(self):
        '''
        If user already has a grid shape file, read it in and create cell list for these cells
        '''
        grid_cell_list = []
        first_row = True
        first_column = True
        ab_line_top_point = None
        ab_line_bottom_point = None
        filepath, file_extension = os.path.splitext(str(self.grid_file))
        if file_extension == '.csv':
            wkt = WKTFiles(self.grid_file)
            poly_cells = wkt.read_grid_file()
        else:
            shp = ShapeFiles(self.grid_file)
            poly_cells = shp.read_grid_file()
        for i, poly in enumerate(poly_cells):
            xmin, ymin, xmax, ymax = poly.bounds
            cell = GridCell([xmin, ymin, xmax, ymax])
            cell.set_inner_datapoints(self.yield_points, self.protein_points)
            cell.sorted_index = i
            cell.original_index = i
            cell.set_polygon_bounds(poly)
            grid_cell_list.append(cell)
            if first_row:
                ab_line_top_point = Point((cell.bottomleft_x+cell.upperright_x)/2, cell.upperright_y)
                first_row = False
            if first_column:
                next_cell = poly_cells[i+1]
                next_cell_ymin = next_cell.bounds[1]
                if cell.bottomleft_y != next_cell_ymin:
                    cell.is_last_in_col = True
                    ab_line_bottom_point = Point((cell.bottomleft_x+cell.upperright_x)/2, cell.bottomleft_y)
                    first_column = False
        self.ab_line = LineString([ab_line_top_point, ab_line_bottom_point])
        return grid_cell_list

    def create_strip_trial(self):
        """
        Create strip trial from grid cells
        """

        self.cell_list = self.create_basic_grid_for_field()
        # initialize values
        grid_cell_list = []
        full_strip_set = []
        x_coord = 0

        for i, cell in enumerate(self.cell_list):
            if self.angle == 0:
                bottomleft_x = cell.bottomleft_x
                true_bounds = cell.true_bounds
            elif self.angle != 0:
                bottomleft_x = cell.original_bounds.bounds[0]
                true_bounds = cell.original_bounds

            if x_coord == 0:
                if self.field_shape.contains(cell.true_bounds):
                    x_coord = bottomleft_x
                    grid_cell_list.append(true_bounds)
            else:
                if x_coord == bottomleft_x:
                    if i != len(self.cell_list)-1:
                        next_cell = self.cell_list[i+1]
                    if self.field_shape.contains(cell.true_bounds):
                        grid_cell_list.append(true_bounds)
                    if bottomleft_x == next_cell.bottomleft_x and cell.bottomleft_y != next_cell.upperright_y:
                        if len(grid_cell_list) != 0:
                            full_strip_set.append(grid_cell_list)
                        grid_cell_list = []
                elif x_coord != cell.bottomleft_x:
                    if len(grid_cell_list) != 0:
                        full_strip_set.append(grid_cell_list)
                    grid_cell_list = []
                    x_coord = bottomleft_x
                    grid_cell_list.append(true_bounds)

        final_cells = []
        polygon_list = []
        for i, s in enumerate(full_strip_set):
            strip = unary_union(s)
            cell_strip = GridCell(strip.bounds)
            cell_strip.folium_bounds = [[cell_strip.bottomleft_y, cell_strip.bottomleft_x], [cell_strip.upperright_y, cell_strip.upperright_x]]
            cell_strip.sorted_index = i
            final_cells.append(cell_strip)
            polygon_list.append(cell_strip.true_bounds)
        self.cell_polys = polygon_list
        if self.angle != 0:
            temp_cells = []
            rotated = rotate(MultiPolygon(cell.cell_polys), self.angle, origin=self.rotation_center)
            for rot, cell in zip(rotated, final_cells):
                cell.set_polygon_bounds(rot)
                temp_cells.append(cell)
            final_cells = temp_cells

        self.cell_list = final_cells
    
    def calculate_index_order(self, number_of_applications):
        '''
        Calculate order of indeces for latin square design.
        E.g. five different rates, order would be: 1,3,5,4,2 or in Python index form: 0,2,4,3,1
        '''
        idx = 0
        indeces = []
        reverse = False
        if number_of_applications%2 == 0:
            odd_idx = False
        else:
            odd_idx = True
        for x in range(number_of_applications):
            print(x, number_of_applications)
            if idx < number_of_applications and not reverse:
                indeces.append(idx)
                idx+=2
            elif idx >= number_of_applications-1:
                if not odd_idx:
                    idx = number_of_applications-1
                    indeces.append(idx)
                else:
                    idx = number_of_applications-2
                    indeces.append(idx)
                idx -= 2
                reverse = True
            elif idx > 0 and reverse:
                indeces.append(idx)
                idx -= 2
        return indeces
    
    def create_basic_latin_square(self):
        '''
        Basic latin square with same rates on diagonal.
        No randomization of column start rates.
        '''
        to_apply = []
        if self.fertilizer_list_2:
            to_apply = self.combine_fertilizer_lists()
        else:
            to_apply = [[x] for x in self.fertilizer_list_1]

        indeces = self.calculate_index_order(len(to_apply))

        cell_index = 0
        for cell in self.cell_list:
            cell.applicator_value = to_apply[indeces[cell_index]]
            if cell_index == len(indeces)-1:
                cell_index = 0
            else:
                cell_index +=1
            if cell.is_last_in_col:
                cell_index = 0
                indeces.append(indeces.pop(0))
    
    def create_non_diagonal_latin_square(self):
        '''
        Latin square with randomly initalized starting rate sequence for the columns.
        '''
        to_apply = []
        if self.fertilizer_list_2:
            to_apply = self.combine_fertilizer_lists()
        else:
            to_apply = [[x] for x in self.fertilizer_list_1]

        # column indeces with set jumps
        column_wise_indeces = self.calculate_index_order(len(to_apply))
        # randomly ordered row indeces
        row_wise_indeces = [x for x in range(len(to_apply))]
        row_wise_indeces = random.sample(row_wise_indeces, len(row_wise_indeces))

        cell_index = 0
        current_column = 0
        # set correct starting order
        print(column_wise_indeces[0], row_wise_indeces[current_column])
        while column_wise_indeces[0] != row_wise_indeces[current_column]:
            column_wise_indeces.append(column_wise_indeces.pop(0))
        for cell in self.cell_list:
            cell.applicator_value = to_apply[column_wise_indeces[cell_index]]
            if cell_index == len(column_wise_indeces)-1:
                cell_index = 0
            else:
                cell_index +=1
            if cell.is_last_in_col:
                cell_index = 0
                if current_column == len(row_wise_indeces)-1:
                    current_column = 0
                else:
                    current_column += 1
                # rotate until start index is correct
                while column_wise_indeces[0] != row_wise_indeces[current_column]:
                    column_wise_indeces.append(column_wise_indeces.pop(0))

    
    def create_two_tier_latin_square(self):
        '''
        NOT FUNCTIONAL YET (or is it? )
        Create 2-tier latin square with extra applicator rates to minimize jumps more.
        '''
        to_apply = []
        if self.fertilizer_list_2:
            to_apply = self.combine_fertilizer_lists()
        else:
            to_apply = self.fertilizer_list_1
        if len(to_apply)%2 == 0:
            largest_index = len(to_apply)-2
        else:
            largest_index = len(to_apply) - 1
        col_nr = 0
        idx = 0
        for cell in self.cell_list:
            cell.applicator_value = to_apply[idx]
            if idx < largest_index:
                idx += 2
            else:
                if col_nr == 1:
                    idx = 1
                elif col_nr == 0:
                    idx = 0
            if cell.is_last_in_col:
                if col_nr == 1:
                    col_nr = 0
                    idx = 0
                elif col_nr == 0:
                    col_nr = 1
                    idx = 1

    """
    MONTANA FIELDS SPECIFIC METHODS:
    1. Cell ordering for genetic algorithm
    2. All methods used for stratification
    """
    def order_cells(self, progressbar=None):
        """
        Looks at the as applied map and orders cells accordingly.
        Used for jump score minimization when using Genetic Algorithm.
        NOT used when creating latin square designs.
        Check if this is working for shapefiles and csv files with different fields.
        """
        start_time = time.time()
        id_val = 0
        as_applied_points = []
        latlong = False
        filepath, file_extension = os.path.splitext(self.as_applied_file)

        if file_extension == '.shp':
            for i,point in enumerate(list(fiona.open(self.as_applied_file))):
                if point['geometry']['type'] == "Polygon":
                    as_applied_points.append([Point(float(point['properties']['LONGITUDE']), float(point['properties']['LATITUDE'])),i])
                    if i == 0:
                        latlong = True
                elif point['geometry']['type'] == "Point":
                    as_applied_points.append([Point(point['geometry']['coordinates']),i])

        if file_extension == '.csv':
            data = pandas.read_csv(self.as_applied_file)
            data.columns = data.columns.str.lower()
            data.columns = data.columns.str.strip()
            for i, row in data.iterrows():
                if is_hex(row['geometry']):
                    wkt_point = wkb.loads(row['geometry'], hex=True)  # ogr.CreateGeometryFromWkb(bts)
                else:
                    wkt_point = row['geometry']
                point = ogr.CreateGeometryFromWkt(str(wkt_point))
                as_applied_points.append(Point(point.GetX(), point.GetY()))
        else:
            print("As applied file is not a csv")

        if latlong:
            project = pyproj.Transformer.from_crs(pyproj.CRS(self.aa_crs), pyproj.CRS(self.latlong_crs),
                                              always_xy=True).transform
            shape_ = transform(project, self.field_shape_buffered)
        else:
            shape_ = self.field_shape_buffered
        good_as_applied_points = [point for point in as_applied_points if shape_.contains(point)]
        i = 0

        ordered_cells = deepcopy(self.cell_list)
        counter = 0
        for p, point in enumerate(good_as_applied_points):
            while i < len(ordered_cells):
                cell = next(x for x in self.cell_list if x.original_index == ordered_cells[i].original_index)
                if latlong:
                    bounds = transform(project, cell.true_bounds)
                else:
                    bounds = cell.true_bounds
                if bounds.contains(point) and cell.is_sorted:
                    og_index = deepcopy(cell.original_index)
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
            for next_cell in self.cell_list[i + 1:]:
                if next_cell.is_sorted:
                    self.cell_list[i].sorted_index = next_cell.sorted_index - 1
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
        '''
        Set what protein or yield bin each cell belongs to.
        '''
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

    def combine_fertilizer_lists(self):
        '''
        Create list of all unique combinations of the two applicator rates.
        These combinations now form the new rates to be applied across the field.
        '''
        combined = []
        if len(self.fertilizer_list_2) < len(self.fertilizer_list_1):
            permut = itertools.permutations(self.fertilizer_list_1, len(self.fertilizer_list_2))
            list_2 = [x for x in self.fertilizer_list_2]
            self.second_item_in_list_flag = self.fertilizer_name_2
        else:
            permut = itertools.permutations(self.fertilizer_list_2, len(self.fertilizer_list_1))
            list_2 = [x for x in self.fertilizer_list_1]
            self.second_item_in_list_flag = self.fertilizer_name_1
        for comb in permut:
            zipped = zip(comb, list_2)
            combined.extend(list(zipped))
        return list(set(combined))
    
    def assign_random_binned_applicator(self, cells_to_assign, to_apply):
        '''
        Random applicator value assignment
        '''
        # takes a set of cells and assigns equal-ish distribution of whatever substances need to be applied (= the applicator value)
        i = 0
        random.shuffle(to_apply)
        for cell in cells_to_assign:
            cell.applicator_value = to_apply[i % len(to_apply)] #random.randint(0, len(self.fertilizer_list_1)-1)
            i = i + 1

    def assign_applicator_distribution(self):
        '''
        Stratify applicator values across binned cells
        '''
        to_apply = []
        if self.fertilizer_list_2:
            combined = self.combine_fertilizer_lists()
            print(combined)
            to_apply = random.sample(combined, k=len(combined))
        else:
            to_apply = random.sample(self.fertilizer_list_1, k=len(self.fertilizer_list_1))
            to_apply = [ [x] for x in to_apply]

        if self.num_yield_bins == 1 or self.yield_points is None:
            for cell in self.cell_list:
                cell.applicator_value = to_apply[random.randint(0, len(to_apply)-1)]

        else:
            self.define_binning_bounds()
            self.set_cell_bins()
            # stratifies combinations of different bins and then assigns applicator rates
            for i in range(1, self.num_yield_bins + 1):
                for j in range(1, self.num_pro_bins + 1):
                    cells_in_curr_ylpro_combo = []
                    for cell in self.cell_list:
                        if cell.yield_bin == i and cell.pro_bin == j:
                            cells_in_curr_ylpro_combo.append(cell)
                    self.assign_random_binned_applicator(cells_in_curr_ylpro_combo, to_apply)

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

class Column:
    '''
    Column class to store column objects and other relevant features to create grid.
    '''
    def __init__(self, polygon, section_id, xmax, xmin, field) -> None:
        self.polygon = polygon
        self.field = field
        self.xmax = xmax
        self.xmin = xmin
        self.section_id = section_id
        self.uid = shortuuid.uuid()
        self.conversion_measure = 6076.1154855643*60 
        self.shorter_length = 0
        self.ymin_cell_start = 0
        self.adjust_polygon()

    def __eq__(self, __o: object) -> bool:
        if self.uid == __o.uid:
            return True
        else:
            return False
    
    def adjust_polygon(self):
        """
        Adjusts the polygon into a rectangle for weirdly shaped columns
        """
        tens = 10/self.conversion_measure
        coords = np.array(self.polygon.boundary.coords)
        lengths_matrix = []
        for i, point in enumerate(coords):
            if i < len(coords)-1:
                print(i)
                pt1 = Point(point[0], point[1])
                pt2 = Point(coords[i+1][0], coords[i+1][1])
                lengths_matrix.append([pt1.distance(pt2), i, LineString((pt1, pt2))])
        lengths_matrix.sort()
        self.shorter_length = lengths_matrix[-2][0]
        xmin, ymin1, xmax, ymax1 = lengths_matrix[-1][-1].bounds
        xmin, ymin2, xmax, ymax2 = lengths_matrix[-2][-1].bounds
        ymins = sorted([ymin1, ymin2])
        ymaxs = sorted([ymax1, ymax2])
        self.ymin_cell_start = ymins[1]
        new_polygon = Polygon([(self.xmin, ymins[1]+tens), (self.xmax, ymins[1]+tens), (self.xmax, ymaxs[0]-tens), (self.xmin, ymaxs[0]-tens)])
        if self.field.contains(new_polygon):
            self.polygon = new_polygon

class GridCell:
    '''
    Gridcell class to store all information pertaining to a single plot in the trial.
    '''
    def __init__(self, coordinates):
        self.is_sorted = False
        self.original_index = -1
        self.sorted_index = -1
        self.is_last_in_col = False

        # coordinates and polygon definitions
        self.true_bounds = Polygon(
            [(coordinates[0], coordinates[1]), (coordinates[2], coordinates[1]), (coordinates[2], coordinates[3]),
             (coordinates[0], coordinates[3])])
        self.bottomleft_x = coordinates[0]
        self.bottomleft_y = coordinates[1]
        self.upperright_x = coordinates[2]
        self.upperright_y = coordinates[3]
        self.conversion_measure = 6076.1154855643*60 
        five_feet = 5/self.conversion_measure
        self.small_bounds = Polygon( #/364567.2
            [(coordinates[0] + five_feet , coordinates[1] + five_feet ),
             (coordinates[2] - five_feet , coordinates[1] + five_feet ),
             (coordinates[2] - five_feet , coordinates[3] - five_feet ),
             (coordinates[0] + five_feet , coordinates[3] - five_feet )])
        self.folium_bounds = None
        self.original_bounds = None

        # values within cell
        self.yield_points = []
        self.protein_points = []
        self.yield_ = -1
        self.pro_ = -1

        # values to be assigned
        self.yield_bin = -1
        self.pro_bin = -1
        self.applicator_value = -1

    def to_dict(self):
        return {
            'protein': self.pro_,
            'yield': self.yield_,
            'applicator_value': self.applicator_value,
            'index': self.sorted_index
        }

    def set_polygon_bounds(self, bounds):
        coordinates = bounds.exterior.coords
        xmin, ymin, xmax, ymax = bounds.bounds
        self.pt1 = Point(xmin, ymin)
        self.pt2 = Point(xmax, ymax)
        self.bottomleft_x = xmin
        self.bottomleft_y = ymin
        self.upperright_x = xmax
        self.upperright_y = ymax
        self.true_bounds = bounds
        self.folium_bounds = [[coordinates[0][1], coordinates[0][0]], [coordinates[1][1], coordinates[1][0]], [coordinates[2][1], coordinates[2][0]], [coordinates[3][1], coordinates[3][0]]]

    def set_inner_datapoints(self, yld=None, pro=None):
        """
        @param yld: Yield object containing all read in yield datapoints
        @param pro: Protein object
        Determine yield and protein points that lie within cell
        """
        if yld is not None:
            # set_datapoints checks which points lie within the gridcell
            # as I am typing this, I realize this is weird....
            # Shouldnt the set datapoints method be in the gridcell object? Why did I put in the datapoints object...
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
        self.yield_ = np.mean(self.yield_points)

    def set_avg_protein(self):
        self.pro_ = np.mean(self.protein_points)


class DataPoint:
    def __init__(self, filename):
        self.filename = filename
        self.datapoints = []

    def create_dataframe(self, datatype=''):
        """
        Create a dataframe containing yield or protein values and their coordinates in the form:
        [ [X1, Y1, value1], [X2, Y2, value2], ..., [Xn, Yn, valuen] ]
        for all n datapoints in the file with numeric values.
        """

        filepath, file_extension = os.path.splitext(str(self.filename))
        datapoints = []
        if 'csv' in file_extension:
            file = pandas.read_csv(self.filename)
            for i, row in file.iterrows():
                datapoint = []
                try:
                    if is_hex(row['geometry']):
                        wkt_point = wkb.loads(row['geometry'], hex=True)  # ogr.CreateGeometryFromWkb(bts)
                    else:
                        wkt_point = row['geometry']
                except Exception as e:
                    print(str(e))
                    if datatype == 'yld':
                        datatype_string = 'yield '
                    elif datatype =='pro':
                        datatype_string = 'protein '
                    else:
                        datatype_string = ''
                    # raise FileHandleError("Your uploaded " + datatype_string + "csv file does not contain a 'geometry' column. ")
                point = ogr.CreateGeometryFromWkt(str(wkt_point))
                datapoint.append(point.GetX())
                datapoint.append(point.GetY())
                if row[datatype] != 'NONE' and not math.isnan(row[datatype]):
                    datapoint.append(row[datatype])
                    datapoints.append(datapoint)
        elif 'shp' in file_extension:
            datapoints = ShapeFiles(self.filename).read_shape_file(datatype)
        return np.array(datapoints)

    def set_datapoints(self, gridcell):
        points_in_cell = self.datapoints[(self.datapoints[:, 0] >= gridcell.bottomleft_x) &
                                         (self.datapoints[:, 0] <= gridcell.upperright_x) &
                                         (self.datapoints[:, 1] <= gridcell.upperright_y) &
                                         (self.datapoints[:, 1] >= gridcell.bottomleft_y)]
        return points_in_cell[:, 2]

class Yield(DataPoint):
    def __init__(self, filename):
        super().__init__(filename)
        self.datapoints = self.create_dataframe('yld')


class Protein(DataPoint):
    def __init__(self, filename):
        super().__init__(filename)
        self.datapoints = self.create_dataframe('pro')