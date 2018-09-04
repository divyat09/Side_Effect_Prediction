# Reading the Patient Feature and Id file
import json

with open('Data/Final_Datasets/Total_Patients_Personal_Details_Features_File.txt') as data_file:    
    patient_data = json.load(data_file)   # yaml.safe_loads produces strings rather than unicode strings as in json.load

with open('Data/Final_Datasets/Selected_Patient_Id.txt') as data_file:    
    patient_id = json.load(data_file)   # yaml.safe_loads produces strings rather than unicode strings as in json.load
    
print len(patient_data)
print len(patient_id)
print "Loading Files Done"

# Total Evaluations for Datasets

total_eval_count = 0
for patient in patient_data.keys():
    for drug in patient_data[patient]["Treatments List"].keys():
            total_eval_count = total_eval_count + 1

print "Total Number of evaluations",
print total_eval_count

# Checking is still some case of no Side Effect being mentioned
for patient in patient_data.keys():
    for drug in patient_data[patient]["Treatments List"].keys():
        if patient_data[patient]["Treatments List"][drug]["Side Effects"] == -1:
            print "Error: Some cases where Side Effect not reported"

print "Done with the Test"

# Checking shape and evaluations for the librec files and vectorised files

import pandas as pd

dataframe = pd.read_csv("Data/Final_Datasets/Librec Side Effects Rating.txt", sep = ' ')       
print "Length of Rating File"
print dataframe.shape

with open('Data/Final_Datasets/Complete_Vectorised_Features_Data.txt') as data_file:    
    patient_data_vec = json.load(data_file)   # yaml.safe_loads produces strings rather than unicode strings as in json.load

print "Length of Vectorised File"
print len(patient_data_vec)

count = 0
for patient in patient_data_vec:
    for drug in patient_data_vec[patient].keys():
        count =count + 1

print "Total evaluations in Vectorised File"
print count

# Generating some statistics for features

other_condition = []
primary_condition = []
location = []
age =[]
gender = []
purpose =[]
side_effect = []

for patient in patient_data.keys():
    
    # Primary Conditions
    feature = patient_data[patient]["Primary Condition"]
    if feature !=-1:
        if feature not in primary_condition:  
            primary_condition.append(feature)
            
    # Location
    feature = patient_data[patient]["Location"]
    if feature !=-1:
        if feature not in location:  
            location.append(feature)
    
    # Other Conditions
    feature = patient_data[patient]["Other conditions"]
    if feature !=-1:
        for item in feature:
            if item not in other_condition:  
                other_condition.append(item)
    # Age            
    feature = patient_data[patient]["Age"]
    if feature !=-1:
        if feature not in age:  
            age.append(feature)
    
    # Gender        
    feature = patient_data[patient]["Gender"]
    if feature !=-1:
        if feature not in gender:  
            gender.append(feature)
    
    # Purpose and Side Effect
    for drug in patient_data[patient]["Treatments List"].keys():
        
        feature = patient_data[patient]["Treatments List"][drug]["Purpose"]
        if feature != -1:
            if feature not in purpose:
                purpose.append(feature)
                
        feature = patient_data[patient]["Treatments List"][drug]["Side Effects"]
        for item in feature:
            if item not in side_effect:
                side_effect.append(item)
                    
print len(primary_condition)
print len(other_condition)
print len(location)
print len(age)
print len(gender)
print len(purpose)
print len(side_effect)