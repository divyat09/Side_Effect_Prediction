'''
Created on Aug 9, 2016

@author: he8819197
'''
import numpy as np
import math
from sklearn.metrics import mean_squared_error

import theano as th
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.constraints import unit_norm,maxnorm,non_neg
from sklearn.utils import class_weight
from Dataset import Dataset
from time import time
import sys
import numpy as np
import pandas as pd

def get_model(num_users, num_items, num_side_effects,  mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0, enable_dropout=False):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    mlp_dim = int(layers[0]/2)
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                   W_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  W_regularizer = l2(reg_mf), input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = mlp_dim, name = "mlp_embedding_user",
                                  W_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = mlp_dim, name = 'mlp_embedding_item',
                                  W_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
    for idx in range(1, num_layer):
        layer = Dense(layers[idx],activation='tanh', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)
        if enable_dropout:
            mlp_vector = Dropout(0.3)(mlp_vector)
        #mlp_vector = BatchNormalization()(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    #predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
    
    # Final prediction layer
    pred_1 = Dense(1, name = "pred_1", activation = 'relu')(mlp_vector)
    pred_2 = Dense(1, name = "pred_2", activation = 'relu')(mf_vector)
    pred_3 = Dense(1, name = "pred_3", activation = 'relu')(mf_user_latent)
    pred_4 = Dense(1, name = "pred_4", activation = 'relu')(mf_item_latent)
    
    predict_vector = merge([pred_1, pred_2], mode = 'concat')

    prediction1 = Dense(1, name = "prediction1", activation = 'relu', kernel_constraint=non_neg())(predict_vector)

    prediction2 = Dense(100,activation = 'relu')(mlp_vector)
    prediction2 = Dropout(0.5)(prediction2)
    prediction2 = BatchNormalization()(prediction2)
    #prediction2 = Dense(1000,activation = 'relu')(prediction2)
    #prediction2 = Dropout(0.5)(prediction2)
    #prediction2 = BatchNormalization()(prediction2)
    prediction2 = Dense(num_side_effects, activation = "sigmoid", name = "prediction2")(prediction2)
    
    model = Model(input=[user_input, item_input], 
                  output=[ prediction1, prediction2] )

    print(model.summary())
    return model


def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    return c1/c2


def recall(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    return c1/c3

def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def precision_loss(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    return 1- c1/c2


def recall_loss(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    return 1- c1/c3

def f1_score_loss(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def mae(y_true, y_pred):
    return K.mean(np.absolute(y_pred-y_true))

def rmse(y_true, y_pred):
    return np.sqrt(K.mean((y_pred-y_true)**2))

if __name__ == '__main__':
    dataset_name = "Librec_Side_Effects_Data.txt"
    mf_dim = 64    #embedding size
    layers = eval("[128,64]")
    reg_layers = eval("[0,0]")
    reg_mf = 0
    learner = "Adam"
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 256
    verbose = 1
    enable_dropout = False
    mf_pretrain = ''
    mlp_pretrain = ''
    
            
    evaluation_threads = 1#mp.cpu_count()
    print("NeuMF(%s) Dropout %s: mf_dim=%d, layers=%s, regs=%s, reg_mf=%.1e, learning_rate=%.1e, num_epochs=%d, batch_size=%d, verbose=%d"
          %(learner, enable_dropout, mf_dim, layers, reg_layers, reg_mf, learning_rate, num_epochs, batch_size, verbose))
        
    # Loading data
    dataset = Dataset("data/"+dataset_name)
    data, labels1, labels2 = dataset.data, dataset.labels1, dataset.labels2
    data1 = data[:,0].tolist()
    data2 = data[:,1].tolist()

    t1 = time()
    cv = 5
    for i in range(0, cv):
        scale = int(len(data)/cv) 
        start = i*scale
        end = (i+1)*scale
        
        X_train_user = np.array( data1[: start] + data1[end :] )
        X_train_item = np.array( data2[: start] + data2[end :] )
        y_train_1 = np.array( labels1[:start] + labels1[end:] )
        y_train_2 = np.array( labels2[:start] + labels2[end:] )

        X_test_user = np.array( data1[start : end] )
        X_test_item = np.array( data2[start : end] )
        y_test_1 = np.array( labels1[start : end] )   
        y_test_2 = np.array( labels2[start : end] )


        num_users = max(data1) + 1
        num_items = max(data2) + 1
        num_side_effects = y_train_2.shape[1]
        # Testing
        print("Total Number of Patients")
        print(num_users)
        print("Total Number of Drugs")
        print(num_items)        
        print("Total Number of Evaluations")
        print(len(labels1))
        print("Training size:")
        print(len(y_train_1))
        print("Test shape:")
        print(y_test_1.shape)
        print("Test_2 shape:")
        print(y_test_2.shape)

        class_weights = class_weight.compute_class_weight('balanced', [0,1], y_train_2.flatten())
        print("Class Weights: ",class_weight) 
        # Build model
        model = get_model(num_users, num_items, num_side_effects,  mf_dim, layers, reg_layers, reg_mf, enable_dropout)
        model.compile(optimizer=Adam(lr=learning_rate), loss={'prediction1':'mean_absolute_error','prediction2':'binary_crossentropy'},
         loss_weights=[1.0,0.3], metrics = {'prediction1': rmse, 'prediction2':[ precision, recall, f1_score]}) 
        val_checkpoint = ModelCheckpoint('bestval.h5','val_prediction1_rmse', 1, True)
        cur_checkpoint = ModelCheckpoint('current.h5')
        early_stop = EarlyStopping(monitor='val_prediction1_rmse', patience=5, verbose=1)

        print('Model compiled.')
     
        # Training
        model.fit([X_train_user, X_train_item], #input
                             [y_train_1, y_train_2], # labels
                             validation_data=[[X_test_user,X_test_item],[y_test_1,y_test_2]], class_weight=[None,class_weights],
                             batch_size=batch_size, epochs=num_epochs, verbose=2,
                             callbacks=[val_checkpoint,early_stop], shuffle=True)

        prediction_weights = model.get_layer("prediction1").get_weights()
        print("Prediction Weights: ", prediction_weights)
        print("Done with current fold of training data")
