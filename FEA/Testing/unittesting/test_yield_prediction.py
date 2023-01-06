import unittest
import pandas as pd

from predictionalgorithms.yieldprediction import *
from optimizationproblems.prescription import *
from sklearn.ensemble import RandomForestRegressor

class TestRFYieldPrediction(unittest.TestCase):
    def setUp(self) -> None:
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
        random_global_variables = random.choices([80,100,120,140], k=len(field.cell_list))
        pr = Prescription(variables=random_global_variables, field=field)
        yp = YieldPredictor(prescription=pr, field=field, agg_data_file=agg_file, trained_model=rf, data_headers=data_to_use)

