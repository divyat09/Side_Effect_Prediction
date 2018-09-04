# Report:
Link to access report : https://www.overleaf.com/read/xwjpsnqjyqfr

# Code:

### Baseline_Model:

 * *Generate_Features.ipynb*
 
   This notebook generates the final master dataset by converting all the patient's personal features and drug evaluation features and stores the result in Total_Patients_Personal_Details_Features_File.txt. It reads from two input files: Total_Patients_Personal_Details.txt and Treatment_Drugs_Evaluation_Detail_Scaled_0_2000_Meaningful.txt

   This also generates mapping files that assign Id to Patients, Durgs, Side Effects, Purpose, Primary Condition, Other Condition and Location and stores them in Id_Mapping_Files directory.

* *Side Effects Predict.ipynb*
 
   This notebook reads the feaures from Total_Patients_Personal_Details_Features_File.txt, generates the training and test sets by doing 5-fold cross validation and uses various Regression Algorithms to predict Side Effect Rating for the test set.
 
 * *Generate_Librec_Files.ipynb*
 
   This notebook reads the master dataset from Total_Patients_Personal_Details_Features_File.txt and generates different types of sub datasets required for Librec and Deep Learning Model. For librec, it generates the dataset and outputs it to Librec Side Effects Rating.txt and for Deep Learning Model it generates dataset and outputs it to Librec_Side_Effects_Data.txt
 
 * *Generate Statistics.ipynb*
 
   This notebook generates statistics like Total Patient, Drugs, Evaluations, etc for the master dataset and the sub datasets for Librec and Deep Learning Model. 

### neuralCF

This contains all the code for the Deep Learning Model.

 * *Dataset.py*
 
   This script generates a dataset class object to be used for the deep learning model. It accepts a filename of the dataset in json format and generates a class for it. It is called inside the deep learning model's code. 
   
 * *NeuMF_basic.py* 
 
   This script generates the basic Deep Learning model where the prediction variable is Side_Effect_Rating.

 * *NeuMF.py* 
 
   This script generates the Deep Learning model where the prediction variables are both Side_Effect_Rating and the Side Effect labels.

