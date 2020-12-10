'''
Dataloader
author: ugurc
201201
'''
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler

class StockPrice(Dataset):
	'''
	'''
	def __init__(self, csv_filepath, target='Open', timestep=60, transform=None, normalize=True):
		'''
		# DOWNLOAD IF NOT EXIST
		# PRINT OUT THE PROCESS if log
		# MAKE MULTIPLE TARGETS

		# ONLY ONE INDICATOR = OPEN for now
		'''
		data = pd.read_csv(csv_filepath)
		self.columns = list(data.columns)
		try:
			i = self.columns.index(target)
		except:
			print(f"Could not find '{target}' in the list.\nAvailable keys:\n {self.columns}")
			return
		self.dataset = np.asarray(data.iloc[:, i:i+1].values,dtype=np.float32)

		self.timestep = timestep
		self.transform = transform

		if normalize:
			self.dataset = normalize_func(self.dataset)


	def __len__(self):
		return len(self.dataset)-self.timestep

	def __getitem__(self,idx):
		'''
		Create a data structure with timestep timesteps and one output
		FLATTEN IF REQUIRED
		# RESHAPE IF REQUIRED
			Returns:
				item(np.ndarray of type float32)
		'''
		if torch.is_tensor(idx):
				idx = idx.tolist()

		item = self.dataset[idx:idx+self.timestep]
		item_next = self.dataset[idx+self.timestep]

		if self.transform:
			item = self.transform(item)

		return item, item_next


def normalize_func(array):
	'''
	Normalize the array
	'''
	_min = np.min(array)
	_max = np.max(array)
	diff = _max-_min
	_normalize = lambda t: ((t-_min)/diff)
	_normalize_func = np.vectorize(_normalize)
	return _normalize_func(array)


