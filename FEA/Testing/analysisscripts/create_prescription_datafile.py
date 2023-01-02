import csv
from operator import attrgetter
from pyproj import Transformer
from predictionalgorithms.yieldprediction import create_indexed_dataframe
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from optimizationproblems.prescription import Prescription
from utilities.multifilereader import MultiFileReader

try:
    import _pickle as pickle
except:
    import pickle
import re
import pandas as pd
import numpy as np


aggregated_data_files = ["../../../Documents/Work/OFPE/Data/Henrys/wood_henrys_10m_yld_2016-2020_UPDATE.csv",
                         "../../../Documents/Work/OFPE/Data/Sec35West/broyles_sec35west_10m_yld_2016-2020_UPDATE.csv",
                         "../../../Documents/Work/OFPE/Data/Sec35Mid/broyles_sec35mid_10m_yld_2016-2020_UPDATE.csv"]  # "../../../Documents/Work/OFPE/Data/Henrys/wood_henrys_10m_yld_2016-2020_UPDATE.csv"]
field_files = ["../utilities/saved_fields/millview.pickle"]
field_names = ["millview"]

if __name__ == '__main__':
    objectives = ['jumps', 'strat', 'fertilizer_rate']

    for fieldfile, agg_file, name in zip(field_files, aggregated_data_files, field_names):
        mf = MultiFileReader(name)
        experiment_filenames = mf.path_to_files
        field = pickle.load(open(fieldfile, 'rb'))
        print(field.latlong_crs, field.aa_crs, field.field_crs)
        project_to_latlong = Transformer.from_crs('epsg:32612', field.latlong_crs)  # 32629, 32612 # 'epsg:32612'
        df = pd.read_csv(agg_file)
        xy = np.array([np.array(project_to_latlong.transform(x, y)) for x, y in zip(df['x'], df['y'])])
        df['x'] = xy[:, 0]
        df['y'] = xy[:, 1]
        headers = {c: i for i, c in enumerate(df.columns)}
        header_index = {i: c for i, c in enumerate(df.columns)}
        header_index[len(df.columns)] = 'N'
        dps = df.to_numpy()

        nondom_archive = []
        filenames = [x for x in experiment_filenames if name in x.lower()]
        for experiment in filenames:
            feamoo = pickle.load(open(experiment, 'rb'))
            # print(feamoo.iteration_stats[-1])
            nondom_archive.extend(feamoo.nondom_archive)
            field_method_name = re.search(r'.*\/FEAMOO\/(.*)_trial', experiment)

        find_center_obj = []
        nondom_indeces = find_non_dominated(np.array([np.array(x.objective_values) for x in nondom_archive]))
        nondom_archive = [nondom_archive[i] for i in nondom_indeces]
        for obj in objectives:
            filename_to_write = '../../MOO_final_prescriptions/' + name + '_combined_prescription_' + obj + '_objective_runs.csv'
            nondom_archive.sort(key=attrgetter(obj))
            prescription = nondom_archive[0]
            find_center_obj.append(np.array(prescription.objective_values))
            #        print(prescription.objective_values)
            #        print([x.nitrogen for x in prescription.variables])
            all_points_df = create_indexed_dataframe(prescription, field, headers, dps)
            all_points_df.to_csv(filename_to_write)
        find_center_obj = np.array(find_center_obj)
        length = find_center_obj.shape[0]
        sum_x = np.sum(find_center_obj[:, 0])
        sum_y = np.sum(find_center_obj[:, 1])
        sum_z = np.sum(find_center_obj[:, 2])
        point = np.array([sum_x / length, sum_y / length, sum_z / length])
        nondom_objectives = np.array([np.array(sol.objective_values) for sol in nondom_archive])
        dist = np.sum((nondom_objectives - point) ** 2, axis=1)
        idx = np.argmin(dist)
        prescription = nondom_archive[idx]
        filename_to_write = '../../MOO_final_prescriptions/' + name + '_combined_prescription_center_objective_runs.csv'
        all_points_df = create_indexed_dataframe(prescription, field, headers, dps)
        all_points_df.to_csv(filename_to_write)
