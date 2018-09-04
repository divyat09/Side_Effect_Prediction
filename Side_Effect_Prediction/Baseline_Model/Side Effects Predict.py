import json
import math
from mlxtend.preprocessing import OnehotTransactions
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt


# Loading the final patient feature and id files
with open('Data/Final_Datasets/Total_Patients_Personal_Details_Features_File.txt') as data_file:    
    patient_data = json.load(data_file)   
    
with open('Data/Final_Datasets/Selected_Patient_Id.txt') as data_file:    
    patient_id = json.load(data_file)   
    
print len(patient_data)
print "Loading Files Done"

# Testing for Datasets

# Checking is still some case of no Side Effect being mentioned
for patient in patient_data.keys():
    for drug in patient_data[patient]["Treatments List"].keys():
        if patient_data[patient]["Treatments List"][drug]["Side Effects"] == -1:
            print "Error: Some cases of Side Effect not mentioned still present"

print "Done with the Test"
total_eval_count = 0

for patient in patient_data.keys():
    for drug in patient_data[patient]["Treatments List"].keys():
            total_eval_count = total_eval_count + 1

print "Total Number of evaluations",
print total_eval_count

# Generating the Training and Test Data Sets

matrix =[]
label = []

patient_counter = 0

for patient in patient_data.keys():
    
	patient_counter = patient_counter + 1

	base_feature = []
	base_feature.append( patient_counter )
	base_feature.append( patient_data[patient]["Age"] )
	base_feature.append( patient_data[patient]["Gender"] )
	base_feature.append( patient_data[patient]["Location"] )
	base_feature.append( patient_data[patient]["Primary Condition"] )

	#if patient_data[patient]["Other conditions"] == -1:
	#	base_feature.append(-1)
	#else:
	#	for item in patient_data[patient]["Other conditions"]:
	#		base_feature.append(item)

	time_list =[]
	for treatment in patient_data[patient]["Treatments List"].keys():
		time_list.append( patient_data[patient]["Treatments List"][treatment]["Date of Evaluation"])
    
	for treatment in patient_data[patient]["Treatments List"].keys():
		drug_feature = []
		drug_feature.append(int(treatment))
		drug_feature.append( patient_data[patient]["Treatments List"][treatment]["Purpose"])
		drug_feature.append( patient_data[patient]["Treatments List"][treatment]["Adherence"])
		drug_feature.append( patient_data[patient]["Treatments List"][treatment]["Cost"])
		drug_feature.append( patient_data[patient]["Treatments List"][treatment]["Date of Evaluation"])
		drug_feature.append( patient_data[patient]["Treatments List"][treatment]["Other_Drug"])
                            
		#for other in patient_data[patient]["Treatments List"][treatment]["Other Treatments Before"]:
			#drug_feature.append(other)
			#drug_feature.append(patient_data[patient]["Treatments List"][str(other)]["Purpose"])
			#drug_feature.append(patient_data[patient]["Treatments List"][str(other)]["Adherence"])
			#drug_feature.append(patient_data[patient]["Treatments List"][str(other)]["Cost"])
			#drug_feature.append(patient_data[patient]["Treatments List"][str(other)]["Side Effect Rating"])

		feature = base_feature + drug_feature
		matrix.append(feature)			
		label.append(patient_data[patient]["Treatments List"][treatment]["Side Effect Rating"])		
print "Done"

# Generating 5-datasets for 5-fold cross validation

#matrix1 = pd.DataFrame(matrix).fillna(-1)
#print matrix1.shape[0]

from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits=5)

train_matrix= []
test_matrix = []
train_label = []
test_label = []

scale =  len(matrix)/5
for i in range(0, 5):
    test_matrix.append( matrix[i*scale : (i+1)*scale] )
    train_matrix.append( matrix[:i*scale] + matrix[(i+1)*scale:] )

    test_label.append( label[i*scale : (i+1)*scale] )
    train_label.append( label[:i*scale] + label[(i+1)*scale:] )

print "Done"

# Linear Regression Model

mse = 0
rmse = 0
for i in range(0,5):
    clf= linear_model.LinearRegression()
    clf.fit(train_matrix[i], train_label[i])
    predict_label = clf.predict(test_matrix[i])
    
    mse = mse + mean_absolute_error(test_label[i], predict_label)
    rmse = rmse + math.sqrt(mean_squared_error(test_label[i], predict_label))
    
print 'MSE'
print mse/5.0
print 'RMSE'
print rmse/5.0

# Random Forest Regressor

for case in range(1,6):
    mse = 0
    rmse = 0
    imp = np.zeros(11)

    for i in range(0,5):
        clf= RandomForestRegressor(n_estimators=case*100, min_samples_leaf= 70)
        clf.fit(train_matrix[i], train_label[i])
        predict_label = clf.predict(test_matrix[i])

        imp = imp + np.array(clf.feature_importances_)
        mse = mse + mean_absolute_error(test_label[i], predict_label)
        rmse = rmse + math.sqrt(mean_squared_error(test_label[i], predict_label))
    
    print 'Case'
    print case
    print 'MSE'
    print mse/5.0
    print 'RMSE'
    print rmse/5.0
    print 'Importance'
    print imp/5.0

    Labels =["P_Id","Age","Gen","Loc","Cond","D_Id","Purp","Ad","Cost","Time","Oth"] 
    x_axis = [1,2,3,4,5,6,7,8,9,10,11]

    #Labels =["P_Id","Age","Gen","Loc","Cond","D_Id","Purp","Ad","Cost","Time"] 
    #x_axis = [1,2,3,4,5,6,7,8,9,10]

    #Labels =["P_Id","Age","Gen","Cond","D_Id","Purp","Ad","Cost","Time"] 
    #x_axis = [1,2,3,4,5,6,7,8,9]

    #Labels =["P_Id","Age","Gen","Cond","D_Id","Purp","Time"] 
    #x_axis = [1,2,3,4,5,6,7]

    #Labels =["P_Id","D_Id"] 
    #x_axis = [1,2]

    #Labels =["P_Id","Age","Gen","Cond","D_Id","Purp","Ad","Cost","Time","Oth"] 
    #x_axis = [1,2,3,4,5,6,7,8,9,10]

    #Labels =["P_Id","Age","Gen","Cond","D_Id","Purp","Time","Oth"] 
    #x_axis = [1,2,3,4,5,6,7,8]

    #Labels =["P_Id","Cond","D_Id","Purp","Time","Oth"] 
    #x_axis = [1,2,3,4,5,6]

    #Labels =["P_Id","D_Id","Purp","Ad","Cost","Time","Oth"] 
    #x_axis = [1,2,3,4,5,6,7]

    #Labels =["P_Id","D_Id","Purp"] 
    #x_axis = [1,2,3]

    plt.bar(x_axis , imp, align='center')
    plt.xticks(x_axis, Labels)
    plt.show()
    
# Decision Tree Model

mse = 0
rmse = 0
imp = np.zeros(7)

for i in range(0,5):
    clf= DecisionTreeRegressor(min_samples_leaf= 100)
    clf.fit(train_matrix[i], train_label[i])
    predict_label = clf.predict(test_matrix[i])
    
    imp = imp + np.array(clf.feature_importances_)
    mse = mse + mean_absolute_error(test_label[i], predict_label)
    rmse = rmse + math.sqrt(mean_squared_error(test_label[i], predict_label))
    
print 'MSE'
print mse/5.0
print 'RMSE'
print rmse/5.0
print 'Importance'
print imp/5.0

#Labels =["P_Id","Age","Gen","Loc","Cond","D_Id","Purp","Ad","Cost","Time"] 
#x_axis = [1,2,3,4,5,6,7,8,9,10]

#Labels =["P_Id","Age","Gen","Cond","D_Id","Purp","Ad","Cost","Time"] 
#x_axis = [1,2,3,4,5,6,7,8,9]

#Labels =["P_Id","Age","Gen","Cond","D_Id","Purp","Time"] 
#x_axis = [1,2,3,4,5,6,7]

#Labels =["P_Id","Age","Gen","Cond","D_Id","Purp","Time","Oth"] 
#x_axis = [1,2,3,4,5,6,7,8]

#plt.bar(x_axis , imp, align = 'center' )
#plt.xticks(x_axis, Labels)
#plt.show()    

# Gaussian Naive Bayes Model

mse = 0
rmse = 0
for i in range(0,5):
    clf= GaussianNB()
    clf.fit(train_matrix[i], train_label[i])
    predict_label = clf.predict(test_matrix[i])
    
    mse = mse + mean_absolute_error(test_label[i], predict_label)
    rmse = rmse + math.sqrt(mean_squared_error(test_label[i], predict_label))
    
print 'MSE'
print mse/5.0
print 'RMSE'

print rmse/5.0