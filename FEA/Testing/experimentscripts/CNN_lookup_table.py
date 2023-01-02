from predictionalgorithms.CNN_yieldpredictor.Predictor.YieldMapPredictor import *
from predictionalgorithms.yieldprediction import *
from optimizationproblems.prescription import Prescription
from utilities.util import *
import pandas as pd
import numpy as np
import pickle, random, utm, re

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path_ = path.group()

np.set_printoptions(suppress=True)
field = pickle.load(open(path_ + '/utilities/saved_fields/sec35mid.pickle', 'rb'))
field.field_name = 'sec35middle'

"""
Initialize CNN model
"""
cnn = YieldMapPredictor(filename='/home/amy/Documents/Work/OFPE/Data/broyles_10m_yldDat_with_sentinel.csv',
                        field='sec35middle', pred_year=2020, training_years=[2016, 2018])
# Load prediction data (it will be saved in cnn.data)
cnn.load_pred_data(objective='yld')
# Load model weights
cnn.model = cnn.init_model(modelType='Hyper3DNet')
path_weights = '/home/amy/projects/OFPETool-master/static/uploads/Model-Hyper3DNet-sec35middle--Objective-yld/Hyper3DNet' + "-sec35middle--Objective-yld"
cnn.model.loadModel(path=path_weights)