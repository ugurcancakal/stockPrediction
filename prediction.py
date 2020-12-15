'''
Google Stock Price Prediction Project
author: Ugurcan Cakal
201209
'''
import time
import torch
import os 

import numpy as np
import pandas as pd
import dataset as sd
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import create_variable, model_parallel


# As a metric
# https://colab.research.google.com/github/ugurcancakal/network_pytorch/blob/master/CNN_GPU_CIFAR10.ipynb#scrollTo=6os4k2PeJLEI
# def accuracy(outputs, labels):
# 	_, preds = torch.max(outputs, dim=1)
# 	return torch.sum(preds == labels).item() / len(preds)

def evaluate(model, loss_fn, valid_dl, metric=None):
	with torch.no_grad():
		results = [loss_batch(model, loss_fn, xb, yb, metric = metric) \
							 for xb, yb in valid_dl]
		losses, nums, metrics = zip(*results)

		# Total size of the dataset
		total = np.sum(nums)
		avg_loss = np.sum(np.multiply(losses, nums)) / total
		avg_metric = None
		if metric:
			avg_metric = np.sum(np.multiply(metrics, nums)) / total
	return avg_loss, total, avg_metric

def loss_batch(model, loss_func, xb, yb, opt = None, metric = None):
	# Generate Predictions
	preds = model(xb)
	# Calculate the loss
	loss = loss_func(preds, yb)

	if opt:
		# Calculate gradient
		loss.backward()
		# Update parameters
		opt.step()
		# Reset gradients
		opt.zero_grad()

	metric_result = None
	if metric:
		# Compute the metric
		metric_result = metric(preds, yb)

	return loss.item(), len(xb), metric_result

def train(epochs, model, loss_fn, train_dl, valid_dl,	opt_fn = torch.optim.Adam, lr = None, metric = None):

	train_losses, val_losses, val_metrics = [], [], []

	# Instantiate the optimizer 
	opt = opt_fn(model.parameters(), lr =lr)

	for epoch in range(epochs):
		# Train
		model.train() # Are you training or evalutating 
		for xb, yb in train_dl:
			xb, yb = create_variable(xb), create_variable(yb)
			train_loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt, metric)

		# Evaluate 
		model.eval()
		val_loss, total, val_metric = evaluate(model, loss_fn, valid_dl, metric)

		# Record the loss & metric
		train_losses.append(train_loss)
		val_losses.append(val_loss)
		val_metrics.append(val_metric)

		# Print progress
		if metric:
			print('Epoch: [{}/{}], train_loss: {:.4f}, valid_loss: {:.4f}, {}:{:.4f}'\
						.format(epoch+1, epochs, train_loss, val_loss, \
										metric.__name__, val_metric))
		else:
			print('Epoch: [{}/{}], train_loss: {:.4f}, valid_loss: {:.4f}'\
						.format(epoch+1, epochs, train_loss, val_loss))
			
	return train_losses, val_losses, val_metrics

def experiment(root_dir, train_path, test_path, batch, features, timesteps, epochs, model, loss_function, opt_fn, lr, metric):
	
	# Paths
	train_path = os.path.join(root_dir,train_path)
	test_path = os.path.join(root_dir,test_path)

	# Datasets
	train_set = sd.StockPrice(train_path, target=features, timesteps = timesteps)
	test_set = sd.StockPrice(test_path, target=features, timesteps = 1)

	# Dataloader
	train_dl = DataLoader(dataset=train_set, batch_size=batch, shuffle=True)
	valid_dl = DataLoader(dataset=test_set,	batch_size=1,	shuffle=True)

	# Model Generation
	model = model_parallel(model)
	loss_function = nn.MSELoss()

	# Training
	start = time.time()
	print("Training for %d epochs..." % epochs)
	train(epochs,model,loss_function,train_dl,valid_dl, opt_fn, lr, metric)
