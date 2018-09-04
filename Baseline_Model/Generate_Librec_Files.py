import json
from mlxtend.preprocessing import OnehotTransactions

# Reading the patient feature and id file.

with open('Data/Final_Datasets/Total_Patients_Personal_Details_Features_File.txt') as data_file:    
    patient_data = json.load(data_file)   # yaml.safe_loads produces strings rather than unicode strings as in json.load

with open('Data/Final_Datasets/Id_Mapping_Files/Selected_Patient_Id.txt') as data_file:    
    patient_id = json.load(data_file)   # yaml.safe_loads produces strings rather than unicode strings as in json.load
    
print len(patient_data)
print "Loading Files Done"

# Testing for Datasets

# Checking is still some case of no Side Effect being mentioned
for patient in patient_data.keys():
    for drug in patient_data[patient]["Treatments List"].keys():
        if patient_data[patient]["Treatments List"][drug]["Side Effects"] == -1:
            print "You are Screwed"

print "Done with the Test"
total_eval_count = 0

for patient in patient_data.keys():
    for drug in patient_data[patient]["Treatments List"].keys():
            total_eval_count = total_eval_count + 1

print "Total Number of evaluations",
print total_eval_count

# Making a list of all side effects: side_effect_repeated

import matplotlib.pyplot as plt

side_effect_repeated_list = []
for patient in patient_data.keys():
    for drug in patient_data[patient]["Treatments List"].keys():
        feature = patient_data[patient]["Treatments List"][drug]["Side Effects"]
        for item in feature:
            side_effect_repeated_list.append(item)

# Making a dictionary of side effect: side_effect_dict: key as side effect and value as total number of times it gets repeated in dataset            
side_effect_dict= {}
for item in side_effect_repeated_list:
    if item not in side_effect_dict.keys():
        side_effect_dict[item] = side_effect_repeated_list.count(item)

print "Done"

# Writing the Side Effect Dict into file

print len(side_effect_dict.keys())
f = open("Data/Final_Datasets/Id_Mapping_Files/Side_effects_Distribution.txt","w")
f.write(json.dumps(side_effect_dict, indent=3))
f.close()

#IMPORTANT
# This below commented code should only be run if you wish to remove certain reported side effects based on their frequency.
# The code here will remvoe all those side effects that do not ge repeated more than twice in the whole list of side 
# effects reported by all patients, hence the name Cutoff-2
# DO NOT run if you do not wish to have any cutoffs.
# You can change the number to have cutoffs accordingly.

'''
with open('Data/Final_Datasets/Side_Effects_Id.txt') as data_file:    
    side_effect_id = json.load(data_file)   
    
print len(side_effect_dict.keys())

# Taking Cutoff 2: Removing those side effects that do not get repeated more than 2 time
useless_side_effect = []
for key in side_effect_id.keys():
    _id =  side_effect_id[key]
    if side_effect_dict[_id] <= 2:        # Case of side effect repeating once: Add it to useless category
        useless_side_effect.append(key)

for item in useless_side_effect:
    f = open("Data/Final_Datasets/Useless/Useless_Side_Effect_2.txt",'a')
    f.write("%s\n"% item)

print len(useless_side_effect)
print "Carefull About APPENDING"

# Cutoff-2
# Removing those evaluations which reported side effect that were repeated not more than twice
# Creating a new patient feature file that only have evaluations which staisfy the constraint of Cutoff-2

for patient in patient_data.keys():
    for drug in patient_data[patient]["Treatments List"].keys():
        feature = patient_data[patient]["Treatments List"][drug]["Side Effects"]
        temp = []
        for item in feature:
            if side_effect_dict[item] >2:
                temp.append(item)        
        if len(temp) == 0:
            patient_data[patient]["Treatments List"].pop(drug, None)
        else:
            patient_data[patient]["Treatments List"][drug]["Side Effects"] = temp
     
    if len(patient_data[patient]["Treatments List"].keys()) == 0:
        patient_data.pop(patient)

print len(patient_data.keys())

# Testing whether the above constraint has been implemented successsfully

for patient in patient_data.keys():
    if len(patient_data[patient]["Treatments List"].keys()) == 0:
        print "You are screwed"
        
    for drug in patient_data[patient]["Treatments List"].keys():
        feature = patient_data[patient]["Treatments List"][drug]["Side Effects"]
        if len(feature) == 0:
            print "You are screwed"
        for item in feature:
            if side_effect_dict[item] <= 2:
                print "You are screwed"

print "Done"

f=open("Data/Final_Datasets/Cutoff_2/Total_Patients_Personal_Details_Features_File_2.txt","w")
f.write( json.dumps(patient_data, indent=3, sort_keys=True) )
f.close()
'''

train_data = {}
patient_condition_data = {}

# Making a vocabulary of all the conditions: primary and other listed by all the patients in dataset.
# This vocabulary is stored in train_data. It is a dictionary with key as condition name and value as 1.0 if condition is 
# from a class of primary condition of patients and 0.5 if the condition is from class of other conditions.

# Concatenating both primary and secodary conditions for each patient and assiging the concatenated list of conditions 
# for each patient in patient_condition_data. Key is patient and value is the concatenated list.
# While making the concatenated list for each patient, we add "P_" at start of primary condition id 
# and "O_" for other condition id. This is to differentiate as there would a primary condition with id 1 and also an 
# other condition with id 1, so to differentiate between same id primary and other conditions, add these strings.

for patient in patient_data.keys():
    condition_list = []
    feature = patient_data[patient]["Primary Condition"]
    if feature != -1:
        condition_list.append("P_" + str(feature) )
        train_data[ "P_" + str(feature) ] = 1             # Assigning score 1 for primary condition
    
    feature = patient_data[patient]["Other conditions"]
    if feature != -1:
        for item in feature:
            condition_list.append( "O_" + str(item))
            train_data[ "O_" + str(item) ] = 0.5          # Assigning score 0.5 for other conditions
 
    patient_condition_data[ patient ] = condition_list

f = open("Data/Final_Datasets/Condition_Features/Patient_Condition_Data.txt","w")
f.write( json.dumps(patient_condition_data, indent=3, sort_keys=True) )
f.close()

f = open("Data/Final_Datasets/Condition_Features/Conditions_Vocaulary.txt", "w")
f.write( json.dumps(train_data, indent=3, sort_keys=True) )
f.close()

print len(patient_condition_data)
print len(train_data)

# Generating a vectorised representation for each patient's concatenated condition list
# Vector is of size of length of vocabualary i.e. len(train_data.keys())
# Every condition from patient's list of conditions i.e. patient_condition_data[patient] is compared 
# with list of conditions in vocabulary. 
# For those conditions of vocab present in patient's list condition, assign the score 1 or 0.5 depending on whether its primary or other condition
# Else assign zero for that component of vector.

vectorised_data ={}
for patient in patient_condition_data:
    temp = []
    for condition in train_data.keys():
        if condition in patient_condition_data[patient]:
            temp.append( train_data[condition] )
        else:
            temp.append(0)
    vectorised_data[ patient ] = temp

f = open("Data/Final_Datasets/Condition_Features/Vectorised_Patient_Condition.txt", "w")
f.write( json.dumps(vectorised_data, indent=3, sort_keys=True) )
f.close()    

# Generating files for Librec and the vectorised feature file for Deep Learning Model: 

total_eval_count = 0
for patient in patient_data.keys():
    content1 = patient_id[patient]
    for drug in patient_data[patient]["Treatments List"].keys():
        content2 = int(drug)
        content3 = patient_data[patient]["Treatments List"][drug]["Side Effect Rating"]

        f = open("Data/Final_Datasets/Librec Side Effects Rating New.txt",'a')
        f.write("%s "% content1 )
        f.write("%s "% content2 )
        f.write("%s\n"% content3 )
        total_eval_count = total_eval_count + 1

print "Total Number of evaluations",
print total_eval_count

# Make a list of all the Side Effects, so that we can convert it to One Hot Notation
side_effect_list = []
for patient in patient_data.keys():
    for drug in patient_data[patient]["Treatments List"].keys():
        side_effect_list.append( patient_data[patient]["Treatments List"][drug]["Side Effects"] )
                    
# Converting to One Hot Vector
convert= OnehotTransactions()
side_effect_list = convert.fit( side_effect_list ).transform( side_effect_list )

# Making the Librec Dictionary i.e. the vectorised feature dataset 
side_effect_counter = 0
librec_dict = {}

for patient in patient_data.keys():
    patient_counter = patient_id[patient]
    librec_dict[patient_counter] = {}
    
    for drug in patient_data[patient]["Treatments List"].keys():
        key = int(drug)
        librec_dict[patient_counter][key] = {}
        librec_dict[patient_counter][key]["Side Effect Rating"] = patient_data[patient]["Treatments List"][drug]["Side Effect Rating"]    
        librec_dict[patient_counter][key]["Primary Condition"] = patient_data[patient]["Primary Condition"]
        librec_dict[patient_counter][key]["Side Effects"] = side_effect_list[ side_effect_counter ].tolist() 
        librec_dict[patient_counter][key]["Condition"] = vectorised_data[patient]
            
        side_effect_counter = side_effect_counter + 1

print "Total Number of Evaluations",
print side_effect_counter

f= open("Data/Final_Datasets/Complete_Vectorised_Features_Data.txt","w")
json.dump(librec_dict, f)
f.close()

print "Done with File Generation for Librec and Deep Learning Model"