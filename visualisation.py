'''
Google Stock Price Prediction Project
Data Visualisation


author: Ugurcan Cakal
201209
'''
import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
import os

def plt_config(figsize=(16,8), dpi=300, linewidth=2, fontsize = 16, transparent=True):
	'''
	Configure visual aspects of figures by changing default rc parameters.
	Arguments and hardcoded parameters can be increased if required.
	Detailed information about available rc parameters here :
	https://matplotlib.org/3.3.3/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file
	
		Arguments:
			figsize(int tuple): 
				figure size in inches
				Default : (12,6)

			dpi(int): 
				dots(pixels) per inch. The more dpi chosen the more resolution in the image
				Default : 300

			linewidth(int):
				linewidth of the lines to be plotted
				Default : 2

			fontsize(int):
				fontsize for labels, ticks and anything else on the figure
				Default : 16

			transparent(bool):
				determines if the background will be transparent or not
				Default : True

	'''
	lines = {'linewidth' : linewidth}

	font = {'family' : 'sans-serif',
					'style'  : 'normal',
					'weight' : 'normal',
					'size'   : fontsize}

	axes = {'facecolor'		: 'white',  # axes background color
					'edgecolor'		: 'black',  # axes edge color
					'linewidth'		: 1,  			# edge linewidth
					'grid'				: True,   	# display grid or not
					'titlesize'		: 'large',  
					'titleweight'	: 'bold',  
					'titlepad'		: 6.0,      # pad between axes and title in points
					'labelsize'		: 'medium',
					'labelpad'		: 5.0,     	# space between label and axis
					'labelweight'	: 'bold',  	# weight of the x and y labels
					'xmargin'			: .05,
					'ymargin'			: .05}

	grid = {'color'			: 'b0b0b0',  	# grid color
					'linestyle'	: 'dashed',  	# solid
					'linewidth'	: .8,     	 	# in points
					'alpha'			: .5}     	 	# transparency, between 0.0 and 1.0

	legend = {'loc'					: 'best',
						'frameon'			: 'True',  # if True, draw the legend on a background patch
						'framealpha'	: 0.8,     # legend patch transparency
						'fancybox'		: False,   # if True, use a rounded box for the legend background, else a rectangle
						'fontsize'		: 'medium'}

	figure = {'figsize'	: figsize,  # figure size in inches
						'dpi'			: dpi}      # figure dots per inch

	savefig = {'dpi'				: dpi,       # figure dots per inch or 'figure'
						 'format'			: 'png',
						 'bbox'				: 'tight',   # {tight, standard}
						 'pad_inches'	: .3,    		 # Padding to be used when bbox is set to 'tight'
						 'transparent': transparent}
	
	config = {'lines' 	: lines, 
						'font' 		: font, 
						'axes' 		: axes, 
						'grid' 		: grid, 
						'legend' 	: legend, 
						'figure' 	: figure, 
						'savefig' : savefig}

	for key, val in config.items():
		rc(key, **val)

def plot_confidence(price, time, upper, lower, 
										ylabel=None, xlabel=None, title=None, 
										num_ticks=10, rotation=45, savefig=False, filepath='figure'):
	''' 
	Plots prices depending on time with confidence interval

		Arguments:
			price : np.array (length N)
				data points representing the price values (y_axis)

			time : np.array (length N)
				time sequence of the prices (x_axis) 

			upper : np.array (length N)
				upper bound for the area to be filled in

			lower : np.array (length N)
				lower bound for the area to be filled in

			ylabel : string (default = None)
				label for y axis 

			xlabel : string (default = None)
				label for x axis 

			title : string (default = None)
				title for the figure 

			num_ticks : int (default = 10)
				number of ticks on the y axis 

			rotation : int (default = 45)
				angle of rotation for x ticks 

			savefig : bool (default = False)
				save figure if True, show figure if False
	'''
	plt.figure()
	plt.fill_between(time, upper, lower, color='orange', alpha=.3, label='Range')
	plt.plot(time,price, color = 'red', label = r'Price')
	plt.xticks(range(0,len(time),len(time)//num_ticks),rotation=rotation)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.autoscale(axis='x',tight=True)
	plt.legend()
	if(savefig):
		plt.savefig(filepath)
	else:
		plt.show()
	plt.close()

def investigate_openning(input_path, output_path):
	''' 
	Given csv data, extract openning price, the highest and the lowest values
	then plots the data using plot_confidence() function

		Arguments:
			filename : string
				csv filename to be investigated

		Example Usage:
			investigate_file('Google_Stock_Price_Train.csv')
	'''
	title = (output_path.split(os.path.sep)[-1]).replace('_',' ')
	dataset = pd.read_csv(input_path)
	cols = dataset.columns
	x_axis = dataset.iloc[:,0].values
	openning = dataset.iloc[:,1].values
	upper = dataset.iloc[:,2].values
	lower = dataset.iloc[:,3].values
	plot_confidence(openning, x_axis, upper, lower, 'Opening Price', cols[0], 
									title=title,
									savefig=True, filepath = output_path)

def visualise_data(root_dir):
	''' Visuzlise the data given the list of filenames
		Args:
			filenames : list/tuple of strings
				csv filenames to be investigated
		Example:
			visualise_data(('Google_Stock_Price_Train.csv', 
											'Google_Stock_Price_Test.csv'))
	'''
	figure_root = os.path.join(os.getcwd(), 'figure')

	if not os.path.exists(figure_root):
		os.makedirs(figure_root)

	filenames = os.listdir(root_dir)
	filenames = [filename for filename in filenames if filename.endswith('.csv')]

	input_paths = [os.path.join(root_dir,filename) for filename in filenames]
	output_paths = [os.path.join(figure_root,filename.replace('.csv', '')) for filename in filenames]

	for _in, _out in zip(input_paths, output_paths):
		investigate_openning(_in, _out)
