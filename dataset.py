'''
Google Stock Price Prediction Project

Custom pythorch dataset definition 
and related utility functions.

author: ugurc
201209
'''

import os

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from utils import normalize_func

class StockPrice(Dataset):
	'''
	Custom pytorch dataset for Google Stock Price records.
	Can be used to construct memory efficient dataloader. 
	'''
	def __init__(self, csv_filepath, target=['Open'], timesteps=60, transform=None, normalize=True):
		'''
		Initializer for StockPrice dataset object.
		Since csv dataset allocate memory in the order of kBs, 
		raw data is stored as a np.ndarray

		Normalize if required

		Check if:
			-csv_filepath exist
			-target columns are provided as list or None
			-target columns list is a subset of available columns

		In each case, it prints an error message and return None

			Arguments:

				csv_filepath(str): 
					Filepath to the csv dataset to be used

				target(list of str):
					target columns to be considered as feature
					Default: ['Open']

				timesteps(int):
					the number of stock records before 
					the financial day of interest to be used in the dataset	
					Default: 60

				transform(function): 
					a transformation function to operate on the data before getting an item
					Default: None

				normalize(bool):
					Normalize the dataset ((x-min)/(max-min)) if True
					Default: True
		'''
		if not os.path.isfile(csv_filepath):
			print(f"ERROR : File {csv_filepath} does not exist!\n")
			return None

		if not isinstance(target, list)	and target is not None:
			print(f"ERROR : The target columns '{target}' need to be provided as list!\n")
			return None 

		data = pd.read_csv(csv_filepath)
		columns = set(data.columns)

		if target and not set(target).issubset(columns):
			print(f"ERROR: '{target}' is not a subset of available columns.\nAvailable keys:\n {columns}")
			return None 

		if not target:
			target = list(data.columns[1:])

		self.target = target
		self.dataset = np.asarray(data[self.target].values,dtype=np.float32)
		self.timesteps = timesteps
		self.transform = transform

		if normalize:
			self.dataset = normalize_func(self.dataset)

		print(f'StockPrice dataset object is constructed succesfully using data in {csv_filepath} !\n')

	def __len__(self):
		'''
		Overwrites the len method to get the length value as len(dataset)
			Returns:
				__len__(int): number of available samples in the dataset.
		'''
		return len(self.dataset)-self.timesteps

	def __getitem__(self,idx):
		'''
		Create a data structure with timesteps number of previous stock
		records and and the stock records in the day of interest. 
		Apply transformation if a transform function is defined. 

			Arguments:

				idx(int):	
					The index of the sample to be returned
				
			Returns:

				prev(2D np.ndarray of type float32):
					{timesteps} stock records before the financial day of interest
					axis=0 -> features
					axis=1 -> timesteps

				the_day(1D np.ndarray of type float32):
					stock records in the given day. idx = 0 returns the
					record in the day {timesteps} because we need {timesteps}
					previous records before the day.
		'''

		prev = self.dataset[idx:idx+self.timesteps]
		the_day = self.dataset[idx+self.timesteps]

		if self.transform:
			prev = self.transform(prev)
			the_day = self.transform(the_day)

		return prev, the_day
