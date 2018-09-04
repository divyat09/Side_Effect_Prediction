'''
Created on Aug 8, 2016

@author: he8819197
'''

import scipy.sparse as sp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.data, self.labels1, self.labels2 = self.load_rating_file_as_matrix(path)
        
    def load_rating_file_as_matrix(self, filename):
        
        with open(filename) as data_file:
            side_effect_data= json.load(data_file)    

        data_temp = [] 
        labels_temp_1 = []   
        labels_temp_2 = []    
        for patient in side_effect_data.keys():
            for drug in side_effect_data[patient].keys():
                temp = []
                temp.append(int(patient))
                temp.append(int(drug))

                data_temp.append(temp)
                labels_temp_1.append( side_effect_data[patient][drug]["Side Effect Rating"])
                labels_temp_2.append( side_effect_data[patient][drug]["Side Effects"])

        data = pd.DataFrame(data_temp).values
        labels1 = labels_temp_1       
        labels2 = labels_temp_2     
        
        return data, labels1, labels2