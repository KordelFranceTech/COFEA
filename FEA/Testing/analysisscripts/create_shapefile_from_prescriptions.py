from copy import deepcopy
import pickle, fiona, os
from fiona.crs import from_epsg
from shapely.geometry import Polygon, shape, mapping, MultiPolygon
import numpy as np

from utilities import fileIO
from utilities.util import PopulationMember

file_ccea = "/media/amy/WD Drive/Prescriptions/optimal/CCEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_50_0804190813.pickle" 
file_nsga = "/media/amy/WD Drive/Prescriptions/optimal/NSGA_Sec35Mid_strip_trial_3_objectives_ga_runs_100_population_25_1104102923.pickle"
fieldfile = "/home/amy/projects/FEA/utilities/saved_fields/sec35mid.pickle"

field = pickle.load(open(fieldfile, 'rb'))
ccea = pickle.load(open(file_ccea, 'rb'))
nsga = pickle.load(open(file_nsga, 'rb'))

#self.jumps, self.fertilizer_rate, self.net_return
net_return_fitnesses = np.array([np.array(x.fitness[-1]) for x in ccea.nondom_archive])
net_return_sol = [x for y, x in sorted(zip(net_return_fitnesses, ccea.nondom_archive))][0]

print([x.fitness for x in ccea.nondom_archive])
print([x.fitness for y, x in sorted(zip(net_return_fitnesses, ccea.nondom_archive))])

fertilizer_fitnesses = np.array([np.array(x.fitness[0]) for x in ccea.nondom_archive])
fertilizer_sol = [x for y, x in sorted(zip(fertilizer_fitnesses, ccea.nondom_archive))][0]


fert_cells = []
nr_cells = []
for i, cell in enumerate(field.cell_list):
    fert_cell = deepcopy(cell)
    fert_cell.nitrogen = fertilizer_sol.variables[i]
    #print(fertilizer_sol.variables[i])
    fert_cells.append(fert_cell)
    nr_cell = deepcopy(cell)
    nr_cell.nitrogen = net_return_sol.variables[i]
    #print(net_return_sol.variables[i])
    nr_cells.append(nr_cell)

schema = {'ID' : 'int', 'AvgYield':'float:9.6', 'AvgProtein':'float:9.6', 'Nitrogen':'float:9.6'}

fileIO.ShapeFiles.write_field_to_shape_file(nr_cells, field, shapefile_schema=schema, filename="/home/amy/Documents/Work/OFPE/optimal_maps/ccea_net_return")
fileIO.ShapeFiles.write_field_to_shape_file(fert_cells, field, shapefile_schema=schema, filename="/home/amy/Documents/Work/OFPE/optimal_maps/ccea_fertilizer")