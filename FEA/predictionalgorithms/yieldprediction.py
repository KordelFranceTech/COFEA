from numpy import var
from sklearn.ensemble import RandomForestRegressor
from pyproj import Transformer
from optimizationproblems.prescription import Prescription
import pandas as pd
import numpy as np
import pickle, random, copy

from utilities.field.field_creation import GridCell


class YieldPredictor:
    """
    Wrapper class to create yield predictions using different TRAINED machine learning models.

    NOTE:
    I definitely did some weird things here that should be cleaned up, specifically related to the datastructure generation.
    I was having a lot of issues with the projections of the data, which all need to be in lat-lon to be processed, 
    but farmers often have their data projected in a specific coordinate system, which creates a lot of issues.
    """
    def __init__(self, field, agg_data_file, trained_model, data_headers, nitrogen_header='n_lbs_ac', prescription=None,
                 cnn_bool=False, weeds_model=None, weeds_headers=None):
        df = pd.read_csv(agg_data_file)
        self.headers = {c: i for i, c in enumerate(df.columns)}
        self.nitrogen_header = nitrogen_header
        self.dps = df
        self.nitrogen_dataframe = pd.DataFrame()
        self.full_df = pd.DataFrame()
        self.data_headers = data_headers
        self.weeds_headers = weeds_headers
        self.adjusted_data_headers = copy.deepcopy(self.data_headers)
        self.adjusted_data_headers.append('cell_index')
        self.field = field
        self.gridcell_size = self.field.cell_list[0].gridcell_size / 43560
        self.model = trained_model
        self.cnn_bool = cnn_bool
        self.weeds_model = weeds_model
        self.variables = []
        self.cell_predictions = []

        if not cnn_bool:
            # creating dataframe to adjust
            if 'cell_index' in self.dps.columns.to_list():
                self.full_df = self.dps
            else:
                self.full_df = create_indexed_dataframe(field=field, headers=self.headers,
                                                    dps=self.dps)
            # self.data_headers.append('cell_index')
            self.nitrogen_dataframe = self.full_df.loc[:, data_headers]
            self.weeds_dataframe = self.full_df.loc[:, weeds_headers]

    def adjust_nitrogen_data(self, prescription, cnn=False):
        """
        Adjust cell Nitrogen rates of dataframe in place for prediction
        """
        if isinstance(prescription, Prescription):
            if isinstance(prescription.variables[0], GridCell):
                self.variables = [x.nitrogen for x in prescription.variables]
            else:
                self.variables = prescription.variables
        else:
            if isinstance(prescription[0], GridCell):
                self.variables = [x.nitrogen for x in prescription]
            else:
                self.variables = prescription
        if not isinstance(self.model, list):
            if not cnn:
                for i, cell_N in enumerate(self.variables):
                    self.full_df.loc[self.full_df['cell_index'] == i].loc[:, self.nitrogen_header] = cell_N
                    self.nitrogen_dataframe = self.full_df.loc[:, self.data_headers]
                    self.weeds_dataframe = self.full_df.loc[:, self.weeds_headers]
            else:
                for i, cell_N in enumerate(self.variables):
                    center_ids = list(self.model.centers[self.model.centers['cell_index'] == i][2])
                    for j in range(len(self.model.patches)):
                        if j in center_ids:
                            nr = self.model.patches[j, 0, 0, :]
                            nr[nr != 0] = cell_N
                            self.model.patches[j, 0, 0, :] = nr
        else:
            pass

    def calculate_yield(self, cnn=False):
        """
        Very convoluted right now because of the columns needed.
        Original data 'self.nitrogen_dataframe' needed for prediction does not use cell index,
        but we need the cell index for each data point to calculate the predicted yield for each cell.

        1. Use highest variability cells to replace with experimental rates, i.e., look at difference in point predictions within a cell.
        2. How many cells should we use for experimental rates?
        """
        if cnn:
            stats_path = '/home/amy/projects/OFPETool-master/static/uploads/Model-Hyper3DNet-sec35middle--Objective-yld/' + self.field.field_name + '_statistics.npy'
            [maxs, mins, maxY, minY] = np.load(stats_path, allow_pickle=True)

            yield_predictions = self.model.model.predictSamples(datasample=self.model.patches, maxs=maxs, mins=mins,
                                                                batch_size=256)

            actual_yield = 0
            for i in range(len(self.field.cell_list)):
                center_ids = self.model.centers.loc[self.model.centers['cell_index'] == i][2]
                cell_pred = np.take(yield_predictions, center_ids, axis=0)
                avg = np.mean(np.array(cell_pred))
                self.cell_predictions.append(avg)
                if not np.isnan(avg):
                    actual_yield += avg * self.gridcell_size
        elif isinstance(self.model, list):
            actual_yield = 0
            for i in range(len(self.field.cell_list)):
                actual_yield += self.model[i][self.variables[i]]

        else:
            adjusted_nitrogen_dataframe = self.full_df.loc[:, self.adjusted_data_headers]
            yield_predictions = self.model.predict(self.nitrogen_dataframe)

            adjusted_nitrogen_dataframe.loc[:, 'predicted'] = yield_predictions
            actual_yield = 0
            for i in range(len(self.field.cell_list)):
                cell_pred = adjusted_nitrogen_dataframe.loc[adjusted_nitrogen_dataframe['cell_index'] == i]
                avg = np.mean(cell_pred['predicted'])
                if not np.isnan(avg):
                    actual_yield += avg * self.gridcell_size
        return actual_yield

    def calculate_weeds(self, cnn=False):
        if not cnn:
            weeds_predictions = self.weeds_model.predict(self.nitrogen_dataframe)
        else:
            weeds_predictions = 0
        return weeds_predictions


def get_points_in_cell(gridcell, dps):
    # Get cell location information
    bl_x, bl_y = gridcell.bottomleft_x, gridcell.bottomleft_y
    ur_x, ur_y = gridcell.upperright_x, gridcell.upperright_y
    # Get all points in the cell
    return dps[(dps[:, 1] >= bl_x) &
                            (dps[:, 1] <= ur_x) &
                            (dps[:, 0] <= ur_y) &
                            (dps[:, 0] >= bl_y)]


def create_indexed_dataframe(dps, field, headers=None, transform_to_latlon=False, transform_from_latlon=False, epsg_string='epsg:32612'):
    """
    Method to assign cell index to each datapoint, includes ability to transform data from and to latlong coordinate system.

    field = Field class object, can be read in from existing field pickle files
    headers = original dataframe headers
    dps = datapoints read in from file
    transform_to_latlon =  points to be read are not in latitude/longitutude, but they need to be because the gridcells are
    transform_from_latlon = output coordinates should not be in latitude/longitutude
    """

    all_points_df = pd.DataFrame()
    project_from_latlong = Transformer.from_crs(field.latlong_crs, epsg_string)
    project_to_latlong = Transformer.from_crs(epsg_string, field.latlong_crs)  # 'epsg:32612')
    if headers:
        x_int = headers['x']
        y_int = headers['y']
    else:
        x_int = 0
        y_int = 1

    if transform_to_latlon:
        xy = np.array(
            [np.array(project_to_latlong.transform(x, y)) for x, y in zip(dps[:, x_int], dps[:, y_int])])
        dps[:, x_int] = xy[:, 0]
        dps[:, y_int] = xy[:, 1]
    dps.loc[:, 'point_index'] = np.arange(0, len(dps))

    np_dps = dps.to_numpy()
    for i, gridcell in enumerate(field.cell_list):
        # Get all points in the cell
        points_in_cell = get_points_in_cell(gridcell, np_dps)
        # Set nitrogen value for points
        if len(points_in_cell) > 0:
            cell_df = pd.DataFrame(points_in_cell)
            #            cell_df.loc[:, n_int] = gridcell.nitrogen
            cell_df.loc[:, 'cell_index'] = i
            all_points_df = pd.concat([cell_df, all_points_df])

    if transform_from_latlon:
        xy = np.array(
            [np.array(project_from_latlong.transform(x, y)) for x, y in
             zip(all_points_df[x_int], all_points_df[y_int])])
        all_points_df.loc[:, x_int] = xy[:, 0]
        all_points_df.loc[:, y_int] = xy[:, 1]
    if headers:
        headers['point_index'] = -1
        headers['cell_index'] = -1
        all_points_df.columns = headers
    all_points_df.sort_values(by=all_points_df.columns[2], inplace=True)
    return all_points_df


if __name__ == '__main__':
    agg_file = "C:/Users/f24n127/Documents/Work/Ag/Data/broyles_sec35mid_2016_yl_aggreg_20181112.csv"
    df = pd.read_csv(agg_file)
    y_labels = df['yl_2016']
    data_to_use = ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi_2012', 'ndvi_2014', 'ndvi_2015', 'yl14_nn_bu_ac',
                   'n15_lbs_ac', 'n14_lbs_ac']
    x_data = df[data_to_use]
    rf = RandomForestRegressor()
    rf.fit(x_data, y_labels)

    field_file = "../utilities/saved_fields/sec35mid.pickle"
    field = pickle.load(open(field_file, 'rb'))
    field.fixed_costs = 1000
    random_global_variables = random.choices([80, 100, 120, 140], k=len(field.cell_list))
    pr = Prescription(variables=random_global_variables, field=field)
    yp = YieldPredictor(prescription=pr, field=field, agg_data_file=agg_file, trained_model=rf,
                        data_headers=data_to_use)
    yp.calculate_yield()
