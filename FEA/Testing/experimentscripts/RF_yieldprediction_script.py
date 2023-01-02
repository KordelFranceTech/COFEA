from audioop import avg
import pandas as pd
import pickle, random, re, os, time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

field = pickle.load(open(path + '/utilities/saved_fields/millview.pickle', 'rb'))
agg_file_20 = "~/Documents/Work/OFPE/Data/millview/mv20.7.csv"
agg_file_21 = "~/Documents/Work/OFPE/Data/millview/mv21.7.csv"


k_fold = KFold(10)

df20 = pd.read_csv(agg_file_20)
df21 = pd.read_csv(agg_file_21)

df = pd.merge(df20, df21, on=('x', 'y', 'lon', 'lat', 'elev', 'slope'), sort=False, suffixes=('_2020', '_2021'))
print(df)

y_label = df['yld_2021'].to_numpy()
yld_data_to_use = ['x', 'y', 'elev', 'slope', 
'prev_yld_2020', 'aa_sr_2020', 'ndvi_py_s_2020', 'ndwi_py_s_2020', 'bm_wd_2020', 'carboncontent10cm_2020', 'phw10cm_2020', 'watercontent10cm_2020', 'sandcontent10cm_2020', 'claycontent10cm_2020', 
'aa_sr_2021', 'ndvi_py_s_2021', 'ndwi_py_s_2021', 'bm_wd_2021', 'carboncontent10cm_2021', 'phw10cm_2021', 'watercontent10cm_2021', 'sandcontent10cm_2021', 'claycontent10cm_2021']
x_data = df[yld_data_to_use].to_numpy()
rf_yld = RandomForestRegressor()

y_weedslabel = df['bm_wd_2021'].to_numpy()
wds_data_to_use = ['x', 'y', 'elev', 'slope', 
'yld_2020', 'prev_yld_2020', 'aa_sr_2020', 'ndvi_py_s_2020', 'ndwi_py_s_2020', 'carboncontent10cm_2020', 'phw10cm_2020', 'watercontent10cm_2020', 'sandcontent10cm_2020', 'claycontent10cm_2020', 
'yld_2021', 'aa_sr_2021', 'ndvi_py_s_2021', 'ndwi_py_s_2021', 'carboncontent10cm_2021', 'phw10cm_2021', 'watercontent10cm_2021', 'sandcontent10cm_2021', 'claycontent10cm_2021']
x_weedsdata = df[wds_data_to_use].to_numpy()
rf_weeds = RandomForestRegressor()
total_weeds = []
total_yield = []

for k, (train, test) in enumerate(k_fold.split(x_data, y_label)):
    print('training')
    rf_yld.fit(x_data[train], y_label[train])
    rf_weeds.fit(x_weedsdata[train], y_weedslabel[train])

    print('scores')
    yield_score = rf_yld.score(x_data[test], y_label[test])
    yield_rmse = mean_squared_error(rf_yld.predict(x_data[test]), y_label[test])
    weeds_score = rf_weeds.score(x_weedsdata[test], y_weedslabel[test])
    weeds_rmse = mean_squared_error(rf_weeds.predict(x_weedsdata[test]), y_weedslabel[test])

    total_weeds.append(weeds_rmse)
    total_yield.append(yield_rmse)

    print(
        "[fold {0}] Yld score: {1:.5f}, yld rmse: {2:.5f}, weeds score: {3:.5f}, weeds rmse: {4:.5f}".format(
            k, yield_score, yield_rmse, weeds_score, weeds_rmse
        )
    )

print('weeds: ', np.mean(total_weeds), '\nyield: ', np.mean(total_yield))
