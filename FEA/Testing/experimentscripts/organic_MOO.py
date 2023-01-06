"""
variable: Seeding rate
objectives: 
    1. net return/yield
    2. Minimizing weed -> one group of all weeds 
                        -> 60-85 points for 80-230 acre fields
                        -> each point is a quarter meter square
                        -> in each square, volume of the weeds is estimated (m3 per m2)
                        -> strong biomass correlation 
                        -> weeds compete with the crops for resources, so more weeds means more weed-seeds which could 
                        take over a field. Theoretically, planting a lot of crop seeds would reduce the weed influence, 
                        however, this has not necessarily been found to be correct.
        Thistle patch growth -> how does seeding rate influence this growth? High seeding rate seems beneficial for the First year,
        but the second year the thistle comes back stronger. But not enough data yet to include in a model.
    3. Rotation of crops to avoid creating optimal ecosystem for bugs and pests 

Organic vs Conventional: maneure spreading is the most expensive part, and more labor.
conventional farming: pesticide control -> ask bruce and paul about data for this: could influence net return since it costs money
plus how beneficial is it actually?

Hannah's bug research: area vs ecological refuge: based on natural features, if there is low yield in an area, 
should we change this to a biodiversity area, since having such an area positively influences yield right outside of it?
--> how can this be changed into an optimization problem.

!!!! Cell size: min 200m because that is the most accurate size for yield monitoring.
NDWI 
"""

from sklearn.ensemble import RandomForestRegressor
from predictionalgorithms.yieldprediction import YieldPredictor
from optimizationproblems.prescription import Prescription
from FEA.factorarchitecture import FactorArchitecture
from utilities.util import *
from MOO.MOEA import *
from MOO.MOFEA import MOFEA
import pandas as pd
from datetime import timedelta
import pickle, random, re, os, time

# suppressing chain warnings, should fix this
pd.options.mode.chained_assignment = None

fea_runs = 10
ga_runs = [100]
population_sizes = [250]
upper_bound = 225

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

field = pickle.load(open(path + '/utilities/saved_fields/millview.pickle', 'rb'))
agg_file = "~/Documents/Work/OFPE/Data/all_agg_exp_2.csv"
reduced_agg_file = "~/Documents/Work/OFPE/Data/millview/reduced_millview20_21_spatial.csv"

df = pd.read_csv(agg_file)
df20 = df.loc[(df['year']==2020) & (df['field']=='millview')]
df21 = df.loc[(df['year']==2021) & (df['field']=='millview')]

df_20_21 = pd.merge(df20, df21, on=('x', 'y', 'lon', 'lat', 'elev', 'slope'), sort=False, suffixes=('_2020', '_2021'))

y_label = df_20_21['yld_2021'].to_numpy()
yld_data_to_use = ['x', 'y', 'elev', 'slope', 
'prev_yld_2020', 'aa_sr_2020', 'ndvi_py_s_2020', 'ndwi_py_s_2020', 'bm_wd_2020', 'carboncontent10cm_2020', 'phw10cm_2020', 'watercontent10cm_2020', 'sandcontent10cm_2020', 'claycontent10cm_2020', 
'aa_sr_2021', 'ndvi_py_s_2021', 'ndwi_py_s_2021', 'bm_wd_2021', 'carboncontent10cm_2021', 'phw10cm_2021', 'watercontent10cm_2021', 'sandcontent10cm_2021', 'claycontent10cm_2021']
x_data = df_20_21[yld_data_to_use].to_numpy()
rf_yld = RandomForestRegressor()

y_weedslabel = df_20_21['bm_wd_2021'].to_numpy()
wds_data_to_use = ['x', 'y', 'elev', 'slope', 
'yld_2020', 'prev_yld_2020', 'aa_sr_2020', 'ndvi_py_s_2020', 'ndwi_py_s_2020', 'carboncontent10cm_2020', 'phw10cm_2020', 'watercontent10cm_2020', 'sandcontent10cm_2020', 'claycontent10cm_2020', 
'yld_2021', 'aa_sr_2021', 'ndvi_py_s_2021', 'ndwi_py_s_2021', 'carboncontent10cm_2021', 'phw10cm_2021', 'watercontent10cm_2021', 'sandcontent10cm_2021', 'claycontent10cm_2021']
x_weedsdata = df_20_21[wds_data_to_use].to_numpy()
rf_weeds = RandomForestRegressor()

rf_yld.fit(x_data, y_label)
rf_weeds.fit(x_weedsdata, y_weedslabel)

print('Millview -- SPEA')
field.fixed_costs = 1000
#random_global_variables = random.choices([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], k=len(field.cell_list))
random_global_variables = [random.randrange(0, upper_bound) for x in range(len(field.cell_list))]
pr = Prescription(variables=random_global_variables, field=field)
yp = YieldPredictor(prescription=pr, field=field, agg_data_file=reduced_agg_file, trained_model=rf_yld, weeds_model=rf_weeds, data_headers=yld_data_to_use, weeds_headers=wds_data_to_use)
FA = FactorArchitecture(len(field.cell_list))
FA.factors = field.create_strip_groups(overlap=True)
FA.get_factor_topology_elements()
nsga = NSGA2

# ORGANIC WHEAT: seeding rate cost = 12$ / bushel (1 bushel = 60 lbs) BUT the applied is in lbs per acre --> convert this!
# yield price = 20$ / bushel
# HEMP is completely different: price received 1.4$/lbs 

@add_method(MOEA)
def calc_fitness(variables, gs=None, factor=None):
    pres = Prescription(variables=variables, field=field, factor=factor, organic=True, yield_predictor=yp, applicator_cost=12, yield_price=20)
    if gs:
        pres.set_fitness(global_solution=gs.variables, cont_bool=True)
    else:
        pres.set_fitness(cont_bool=True)
    return pres.objective_values


for j in range(5):
    for population in population_sizes:
        for ga_run in ga_runs:
            start = time.time()
            filename = path + '/results/prescriptions/optimized/SPEA2_millview_trial_organic_objectives_ga_runs_' + str(ga_run) + '_population_' + str(population) + time.strftime('_%d%m%H%M%S') + '.pickle'
            # feamoo = MOFEA(fea_runs, ga_run, pop_size=population, factor_architecture=FA, base_alg=nsga, dimensions=len(field.cell_list),
            #                value_range=[0,upper_bound], ref_point=[1, 1])  # , combinatorial_options=field.nitrogen_list)
            # feamoo.run()
            nsga = SPEA2(population_size=population, ea_runs=ga_run, dimensions=len(field.cell_list),
                         value_range=[0,upper_bound], reference_point=[1, 1])
            nsga.run()
            end = time.time()
            pickle.dump(nsga, open(filename, "wb"))
            elapsed = end-start
            print("NSGA with ga runs %d and population %d took %s"%(ga_run, population, str(timedelta(seconds=elapsed))))