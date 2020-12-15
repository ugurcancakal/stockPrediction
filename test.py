'''
Unittest
author: ugurc
201201
'''
import sys
sys.path.insert(0,'..')

import unittest
import visualisation as vis 
import dataset as sd
import regressor as reg
import prediction as pred
import utils as u

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time
import os

root_data = os.path.join(os.getcwd(), 'data')
root_figure = os.path.join(os.getcwd(), 'figure')

class TestStockPrice(unittest.TestCase):

	# def test1(self):
	# 	file = 'Google_Stock_Price_Train.csv'
	# 	file = u.adjust_split(root_data, file, rate=2.002, date='2014-03-27')
	# 	file = u.adjust_split(root_data, file, rate=1.0027455, date='2015-04-27')

	# def test2(self):
	# 	vis.plt_config(figsize=(16,9),transparent=False)
	# 	vis.visualise_data(root_data)

	

	# def test_dataset(self):
	# 	'''
	# 	Hardcode the 0th and the last
	# 	'''
	# 	filepath = os.path.join(root_data,'Google_Stock_Price_Train_2014-03-27_(2.002)_2015-04-27_(1.0027455).csv')
	# 	timesteps = 60
	# 	dataset = sd.StockPrice(filepath, target=None, timesteps = timesteps, normalize=True)
	# 	self.assertEqual(len(dataset),1258-timesteps)
	# 	print(dataset[0])

	# def test_network(self):
	# 	indicators = 1
	# 	timestep = 60
	# 	model = reg.LSTM(indicators,timestep)
	# 	print(model)

	# def test_train(self):
	# 	filepath = 'Google_Stock_Price_Train.csv'
	# 	batch = 1
	# 	indicators = 1
	# 	timestep = 60
	# 	epochs = 1

	# 	dataset = sd.StockPrice(filepath, timestep = timestep, normalize=True)

	# 	train_dl = DataLoader(dataset=dataset,
	# 											  batch_size=batch,
	# 											  shuffle=True)

	# 	model = reg.LSTM(indicators,timestep)
	# 	loss_fn = nn.MSELoss()

	# 	for epoch in range(epochs):
	# 		# Train
	# 		model.train() # Are you training or evalutating 
	# 		for xb, yb in train_dl:
	# 			# out = model(xb)
	# 			train_loss, _, _ = pred.loss_batch(model, loss_fn, xb, yb)
	# 			break


	# def test10(self):
	# 	train_path = os.path.join(root_data,'Google_Stock_Price_Train_2014-03-27_(2.002)_2015-04-27_(1.0027455).csv')
	# 	test_path = os.path.join(root_data,'Google_Stock_Price_Test.csv')
	# 	batch = 1
	# 	indicators = 1
	# 	timesteps = 60
	# 	epochs = 2

	# 	train_set = sd.StockPrice(train_path, timesteps = timesteps)
	# 	test_set = sd.StockPrice(test_path, timesteps = 1)

	# 	train_dl = DataLoader(dataset=train_set,
	# 												batch_size=batch,
	# 												shuffle=True)

	# 	valid_dl = DataLoader(dataset=test_set,
	# 												batch_size=batch,
	# 												shuffle=True)

	# 	model = reg.LSTM(indicators,timesteps)
	# 	loss_function = nn.MSELoss()

	# 	if torch.cuda.device_count() > 1:
	# 		print("Let's use", torch.cuda.device_count(), "GPUs!")
	# 		# dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
	# 		model = nn.DataParallel(model)

	# 	if torch.cuda.is_available():
	# 		print("Let's use GPU!")
	# 		model.cuda()

	# 	print(model)
	# 	start = time.time()
	# 	print("Training for %d epochs..." % epochs)
		
	# 	pred.train(epochs,model,loss_function,train_dl,valid_dl,lr=0.001)

	def test11(self):
		features = ['Open']
		timesteps = 60
		model = reg.LSTM(len(features),timesteps)

		exp1 = {'root_dir' : root_data,
						'train_path' : 'Google_Stock_Price_Train_2014-03-27_(2.002)_2015-04-27_(1.0027455).csv',
						'test_path' : 'Google_Stock_Price_Test.csv',
						'batch' : 1,
						'features' : features,
						'timesteps' : timesteps,
						'epochs' : 4,
						'model' : model,
						'loss_function' : nn.MSELoss, 
						'opt_fn' : torch.optim.Adam, 
						'lr' : 0.001, 
						'metric' : None }

		pred.experiment(**exp1)
if __name__=='__main__':
	unittest.main()