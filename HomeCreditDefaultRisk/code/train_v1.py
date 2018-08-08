from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import svm

def generateXandY(features_file, target_file, rec_count):
	dataset = pd.read_csv(features_file)
	dataset_target = np.resize(pd.read_csv(target_file).values, rec_count)

	# Define which columns should be encoded vs scaled
	columns_to_encode = ['name_contract_type', 'code_gender', 'flag_own_car', 'flag_own_realty', 'name_income_type', 'name_education_type', 'name_family_status', 'name_housing_type', 'flag_emp_phone', 'flag_work_phone', 'flag_cont_mobile', 'flag_phone']
	# columns_to_encode = ['name_contract_type', 'code_gender', 'flag_own_car', 'flag_own_realty', 'name_type_suite']
	columns_to_scale  = ['cnt_children','amt_income_total', 'amt_credit', 'amt_annuity', 'amt_goods_price', 'region_population_relative', 'days_birth', 'days_employed','days_registration','days_id_publish', 'cnt_fam_members','region_rating_client','region_rating_client_w_city']

	# Instantiate encoder/scaler
	scaler = StandardScaler()
	ohe    = OneHotEncoder(sparse=False)

	# Scale and Encode Separate Columns
	scaled_columns  = scaler.fit_transform(dataset[columns_to_scale]) 
	encoded_columns =    ohe.fit_transform(dataset[columns_to_encode])

	# Concatenate (Column-Bind) Processed Columns Back Together
	processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1)	

	return processed_data, dataset_target

def main():
	processed_data_train, dataset_target_train = generateXandY('../Data/train_data.csv', '../Data/train_data_target.csv', 2000)

	# print(processed_data_train.shape)
	# print(dataset_target_train.shape)
	# print(processed_data_train[2])
	# print(type(dataset_target_train))

	clf = svm.SVC(C=0.5, kernel='linear')
	training_output = clf.fit(processed_data_train, dataset_target_train)  
	print(training_output)

	processed_data_test, dataset_target_test = generateXandY('../Data/test_data.csv', '../Data/test_data_target.csv', 400)	
	# processed_data_test, dataset_target_test = generateXandY('../Data/train_data.csv', '../Data/train_data_target.csv',2000)	
	testing_output = clf.predict(processed_data_test)

	hit = 0.0
	for real, predicted in zip(dataset_target_test, testing_output):
		if(real == predicted):
			hit = hit + 1.0

	print('hit: ', hit)
	print('len testing_output: ', len(testing_output))
	print('Accuracy: ', hit/len(testing_output))                              	

	# print(processed_data_test.shape)
	# print(dataset_target_test.shape)
	# print(processed_data_test[2])
	# print(type(dataset_target_test))


if __name__ == "__main__":
    main()	       		