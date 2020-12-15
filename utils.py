'''
Google Stock Price Prediction Project
Utility functions

author: Ugurcan Cakal
201209
'''
import torch
import pandas as pd
import os 

import numpy as np

from torch.autograd import Variable
from datetime import datetime, timedelta


def adjust_split(root_data, csv_filename, rate=2.002, date='2014-03-27'):
	'''
	Adjust closing price values according to the last split decision
	after the given date by applying the given transformation.
	Save the file in the given root directory with '_(Adjusted).csv' suffix.

		Arguments:
			
			root_data(str):
				root for the directory containing the data

			csv_filename(str):
				csv file to be adjusted

			rate(float):
				split rate 
				Default : 2.002

			date_before(str):
				string indicating the date before apply the transformation 
				in '%Y-%m-%d' form
				Default : '2014-03-26'

		Returns:

			filename(str):
				filename of the output file
	'''
	trans=lambda x: round(x/rate, 2)
	date_before= datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)
	date_after= datetime.strptime(date, '%Y-%m-%d')
	input_path = os.path.join(root_data,csv_filename)

	dataframe = pd.read_csv(input_path)
	dataframe.Date = pd.to_datetime(dataframe.Date)
	dataframe.set_index('Date', inplace=True)
	dataframe = dataframe.replace(',','', regex=True)

	lower_half = pd.DataFrame(dataframe.loc[:date_before])
	upper_half = dataframe.loc[date_after:]

	lower_half.Close = lower_half.Close.astype(float).apply(trans)

	dataframe = pd.concat((lower_half,upper_half))

	filename = csv_filename.replace('.csv', f'_{date}_({rate}).csv')
	output_path = os.path.join(root_data, filename)
	dataframe.to_csv(output_path)

	print(f"Stock split asjustment done on {csv_filename} with rate {rate} before {date}!\n")

	return filename

def normalize_func(array):
	'''
	Function for normalizing the array.
	Consider each column independently.

		Arguments:

			array(np.ndarray):
				the array whose elements to be normalized.

		Returns:
			normalized_array(np.ndarray):
				the normalized array

	'''
	_min = np.min(array,axis=0)
	diff = np.max(array,axis=0)-_min
	return (array-_min)/diff

def create_variable(tensor):
	'''
	Helper function to create a variable from a tensor 
	considering CUDA availability. 

		Arguments:
			tensor(torch.Tensor):
				pytorch tensor to be used

		Returns:
			tensor(torch.Variable):
				pytorch variable(CUDA if available)
	'''
	if torch.cuda.is_available():
		return Variable(tensor.cuda())
	else:
		return Variable(tensor)

def model_parallel(model, log=False):
	'''
	Helper function to parallelize the model in the case 
	there is at least one GPU(or more).

		Arguments:
			
			model(torch.model):
				network model of interest

			log(bool):
				print the model if True

		Returns:
			model(torch.model):
				parallelized model(if available)
	'''
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		# dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
		model = nn.DataParallel(model)

	if torch.cuda.is_available():
		print("Let's use GPU!")
		model.cuda()

	if (log):
		print(model)

	return model