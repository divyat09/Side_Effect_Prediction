import theano.tensor as T
import tensorflow as tf
from theano import shared
import keras
from keras import backend as K

y_true = T.constant([[1,0,0],[1,1,1],[0,0,1],[1,0,1]])
y_pred = T.constant([[0.8,0.7,0.3],[0.9,0.2,0.98],[0.7,0.2,0.1],[0.9,0.2,0.8]])

print tf.nn.top_k(y_pred,2)

print y_true.max()

ytrue_np = y_true.eval()
ypred_np = y_pred.eval()

for i in range(0, len(ypred_np)):
	sorted_index= ypred_np[i].argsort()[-2:]
	ypred_np[i] = 0	
	ypred_np[i][sorted_index] = 1	
	
	print ytrue_np[i][sorted_index] 
	print ypred_np[i][sorted_index]

print ypred_np

y_true = shared(ytrue_np)
y_pred = shared(ypred_np)

y_true = K.cast(y_true, 'float32')
y_pred = K.round(y_pred)

TP = K.sum(y_true*y_pred)
TN = K.sum((1-y_true)*(1-y_pred))
FP = K.sum(y_pred*(1-y_true))
FN = K.sum((1-y_pred)*y_true)

accuracy = (TP+TN)/(TP + TN + FP + FN)
precision = TP/(TP + FP)
recall = TP/(TP + TN) 

print accuracy.eval()
print precision.eval()