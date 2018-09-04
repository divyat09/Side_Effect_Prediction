# Reading the drug evaluations data file
import json
import re

with open('Data/Treatment_Drugs_Evaluation_Detail_Scaled_0_2000_Meaningful.txt') as data_file:    
    eval_data = json.load(data_file)   # yaml.safe_loads produces strings rather than unicode strings as in json.load

# Reading patients personal details file
with open('Data/Total_Patients_Personal_Details.txt') as data_file:    
    patient_data = json.load(data_file)   # yaml.safe_loads produces strings rather than unicode strings as in json.load

# Initialising
for patient in patient_data:
    patient_data[patient]["Treatments List"] = {}    


# Assigning the drugs with their respective id's. Constraint is imposed here that every patient must have evaluated 
# atleast 3 different drugs and every drug must have non zero evaluations.

drug_id = {}
drug_counter = 0

for key in eval_data.keys():
	
	drug_eval_counter = 0
	for item in eval_data[key]:	
		patient = item["Patient Name"]

# Maintains the constraint that patients are under at least 3 different treatments
		if patient in patient_data.keys():			
#			drug_eval_counter = drug_eval_counter + len(item["evaluations"].keys()) #This leads to a patient having multiple evals, but we only consider the latest one
			drug_eval_counter = drug_eval_counter + 1 
            
	# Only those drugs are assigned id's now which have more than 1 evaluations: Makes sense, no evaluations for a drug means it useless 
	if drug_eval_counter >=1 and key not in drug_id.keys():
		drug_id[key] = drug_counter
		drug_counter = drug_counter + 1

print "Total Number of Drugs"
print len(drug_id)

# Writing the final new data for Patient Details
f=open("Data/Final_Datasets/Id_Mapping_Files/Selected_Drug_Id.txt","w")
f.write( json.dumps(drug_id, indent=3, sort_keys=True) )
f.close()

# Checking if above constraint was implemented successfully

for key in drug_id.keys():
    drug_eval_counter = 0
    for item in eval_data[key]:
        patient = item["Patient Name"]
        if patient in patient_data.keys():
            #drug_eval_counter = drug_eval_counter + len(item["evaluations"].keys()) 
            drug_eval_counter = drug_eval_counter + 1 
    if drug_eval_counter < 1:
        print "Error"

for key in drug_id.keys():
    for item in eval_data[key]:
        patient= item["Patient Name"]
        if patient in patient_data.keys() and len(item["evaluations"].keys()) <=0:
            print "Error"

from datetime import datetime

# Generating Drug Specific Features
for key in drug_id.keys():
	
	for item in eval_data[key]:	
		patient = item["Patient Name"]
# if statement maintains the check that each patient is under at least 3 different drugs constraint 
		if patient in patient_data.keys():			
				# Convert string into datetime format
				date_list =[]
				for date in item["evaluations"].keys():
						date_list.append( datetime.strptime( date, '%d/%m/%Y') )                    
				# Find the latest date
				date = item["evaluations"].keys()[ date_list.index( max(date_list) ) ]
				
				temp_dict = {}
				# Assign the data of the latest evaluation for that patient
				temp_dict["Date of Evaluation"] = max(date_list)
				temp_dict["Effectiveness Rating"] = item["evaluations"][date]["Effectiveness"]
				temp_dict["Side Effect Rating"] = item["evaluations"][date]["Side Effect Rating"]
				temp_dict["Advice & Tips"] = item["evaluations"][date]["Advice & Tips"]
				temp_dict["Adherence"] = item["evaluations"][date]["Adherence"]
				temp_dict["Burden"] = item["evaluations"][date]["Burden"]
				temp_dict["Cost"] = item["evaluations"][date]["Cost"]
				temp_dict["Purpose"] = item["evaluations"][date]["Purpose"]
				temp_dict["Side Effects"] = item["evaluations"][date]["Side effects"]
				patient_data[patient]["Treatments List"][drug_id[key]] = temp_dict
				if temp_dict["Side Effects"] == -1: #Checking if still there are cases that do not have side effects reported                
					print "You are Screwed"

# Removing patients that do not have evaluted any drugs now because of some constraints.
counter_useless = 0
for patient in patient_data.keys():
	if not len(patient_data[patient]["Treatments List"].keys()):
		counter_useless = counter_useless + 1
		patient_data.pop(patient, None)

print counter_useless

# Normalising the date of evalations of different drugs for one patient to 0 to 1
counter_useless = 0
for patient in patient_data.keys():

	# Making a list of all dates for that patient    
	treatment_dict = patient_data[patient]["Treatments List"]
	date_list = []
	for drug in treatment_dict.keys():
		date_list.append( treatment_dict[drug]["Date of Evaluation"])
        
	# Initialising          
	max_date = max(date_list)   
	min_date = min(date_list)
	scale = (max_date - min_date).days
	count = 0
    
	# Normalising all the dates to 0 and 1 and assigning the normalised value in the patient dictionary
	days_list = []    
	if scale !=0:
		for drug in treatment_dict.keys():
			treatment_dict[drug]["Date of Evaluation"] = ( date_list[count] - min_date ).days / float(scale)
			days_list.append( ( date_list[count] - min_date ).days )
			count = count + 1
	else:
		for drug in treatment_dict.keys():
			days_list.append( ( date_list[count] - min_date ).days )
			treatment_dict[drug]["Date of Evaluation"] = 1
        
	# Making a list which has value 0 if no drug previous to it was taken in span of 30 days else 1    
	other_drug_list = []
	days_list.sort()

	for count in range(0, len(days_list)):
		other_drug = 0
		for sub_count in range(0,count):
			if days_list[count] - days_list[sub_count] <= 30:
				other_drug = 1
		other_drug_list.append(other_drug)
    
	# Assigning the binary other drug value to patient dictionary   
	count = 0
	for drug in treatment_dict.keys():
		treatment_dict[drug]["Other_Drug"] = other_drug_list[count]
		count = count + 1
     
	patient_data[patient]["Treatments List"] = treatment_dict 

print "Done"

other_condition = []
primary_condition = []
sentence_list = []
location = []

# Generating Patient_Specific Features

primary_condition_list = {}
other_condition_list = {}
location_list = {}

counter1 = 0
counter2 = 0
counter3 = 0
counter4 = 0

for patient in patient_data.keys():
    
	# Removing Unnecessary Data
	patient_data[patient].pop("Symptoms", None)

	# Feature Generation for Patient Condition: Assigning every different condition an integer id.
	feature = patient_data[patient]["Primary Condition"]
	if feature !=-1:
		if feature not in primary_condition_list.keys():  
			
			primary_condition_list[ feature ] = counter1
			patient_data[patient]["Primary Condition"] = counter1
			counter1 = counter1 + 1

		else:
			patient_data[patient]["Primary Condition"] = primary_condition_list[ feature ]		
	
	# Feature Generation for Other Conditions		
	feature = patient_data[patient]["Other conditions"]
	if feature !=-1:	
		temp = []
		for condition in feature:	

			if condition not in other_condition_list.keys():
				other_condition_list[condition] = counter2
				temp.append(counter2)
				counter2 =counter2 + 1 

			else:
				temp.append(other_condition_list[ condition ])
		patient_data[patient]["Other conditions"] = temp		
				

	# Feature generation for Location
	feature = patient_data[patient]["Location"]
    
# There was some issue that empty strings were being assigned id as they had not been assigned -1 label. Did that here   
	if feature == "":
		patient_data[patient]["Location"] = -1
		feature = -1
        
	if feature != -1:    
		if feature not in location_list.keys() :
			location_list[ feature ] = counter3
			patient_data[patient]["Location"] = counter3
			counter3 =counter3 +1 		
	
		else:
			patient_data[patient]["Location"] = location_list[ feature ]
	
	# Feature generation for Gender
	special_gender_list = ["Genderqueer", "Agender", "Asexual", "Gender Neutral", "TransFemale/Transwoman", "TransMale/Transman", "Genderfluid", "genderfluid", "Non Binary", "NonBinary"]
	if patient_data[patient]["Gender"] == "Female":
		patient_data[patient]["Gender"] = 1

	elif patient_data[patient]["Gender"] == "Male":
		patient_data[patient]["Gender"] = 0
        
	elif patient_data[patient]["Gender"] in special_gender_list:
		patient_data[patient]["Gender"] = 2
	else:
		counter4 = counter4 + 1
		patient_data[patient]["Gender"] = -1 

	# Feature Generation for Age

	if 0<= patient_data[patient]["Age"] and patient_data[patient]["Age"] <= 20 :
		patient_data[patient]["Age"] = 0

	elif 20< patient_data[patient]["Age"] and patient_data[patient]["Age"] <= 40 :
		patient_data[patient]["Age"] = 1

	elif 40< patient_data[patient]["Age"] and patient_data[patient]["Age"] <= 60 :
		patient_data[patient]["Age"] = 2

	elif 60< patient_data[patient]["Age"] and patient_data[patient]["Age"] <= 80 :
		patient_data[patient]["Age"] = 3

	elif 80< patient_data[patient]["Age"] and patient_data[patient]["Age"] <= 100 :
		patient_data[patient]["Age"] = 4

	elif 100< patient_data[patient]["Age"]:
		patient_data[patient]["Age"] = 5
        
print len(primary_condition_list)
print len(other_condition_list)
print len(location_list)

# Writing the mapping files
f=open("Data/Final_Datasets/Id_Mapping_Files/Primary_Condition_Id.txt","w")
f.write( json.dumps(primary_condition_list, indent=3, sort_keys=True) )
f.close()

f=open("Data/Final_Datasets/Id_Mapping_Files/Other_Condition_Id.txt","w")
f.write( json.dumps(other_condition_list, indent=3, sort_keys=True) )
f.close()

f=open("Data/Final_Datasets/Id_Mapping_Files/Location_Id.txt","w")
f.write( json.dumps(location_list, indent=3, sort_keys=True) )
f.close()

# Generating Features for Side Effects and Purpose

purpose ={}
side_effect = {}
count1 = 0
count2 = 0

for patient in patient_data.keys():
    
    # Purpose and Side Effect Feature Generation: Assiging each purpose and side effect an integer id
    for drug in patient_data[patient]["Treatments List"].keys():
        
        # Purpose
        feature = patient_data[patient]["Treatments List"][drug]["Purpose"]
        if feature != -1:
            temp = 0
            if feature not in purpose.keys():
                purpose[ feature ] = count1
                temp = count1
                count1 = count1 + 1
            else:
                temp = purpose[feature]

            patient_data[patient]["Treatments List"][drug]["Purpose"] = temp
        
        # Side Effect
        feature = patient_data[patient]["Treatments List"][drug]["Side Effects"]
        temp = []
        for item in feature:
            item = item.replace(" ","").lower()

            if item == "":
                item = "None"

            if item not in side_effect.keys():
                side_effect[item] = count2
                temp.append(count2)
                count2 = count2 + 1
            else:
                temp.append( side_effect[item] )
                
        patient_data[patient]["Treatments List"][drug]["Side Effects"] = temp
        
print len(purpose)
print len(side_effect)

f=open("Data/Final_Datasets/Id_Mapping_Files/Purpose_Id.txt","w")
f.write( json.dumps(purpose, indent=3, sort_keys=True) )
f.close()

f=open("Data/Final_Datasets/Id_Mapping_Files/Side_Effects_Id.txt","w")
f.write( json.dumps(side_effect, indent=3, sort_keys=True) )
f.close()

patient_counter =1
patient_id ={}

# Creating integer id's for patients
for patient in patient_data.keys():
    patient_id[patient] = patient_counter
    patient_counter = patient_counter + 1
            
# Writing the final new data for Patient Details
print len(patient_data)

f=open("Data/Final_Datasets/Id_Mapping_Files/Selected_Patient_Id.txt","w")
f.write( json.dumps(patient_id, indent=3, sort_keys=True) )
f.close()

f=open("Data/Final_Datasets/Total_Patients_Personal_Details_Features_File.txt","w")
f.write( json.dumps(patient_data, indent=3, sort_keys=True) )
f.close()