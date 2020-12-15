'''
Google Stock Price Prediction Project
Data Visualisation related functions

author: Ugurcan Cakal
201209
'''
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import os

def plt_config(figsize=(16,9), dpi=300, linewidth=2, fontsize = 16, transparent=True):
	'''
	Configure visual aspects of figures by changing default rc parameters.
	Arguments and hardcoded parameters can be increased if required.
	Detailed information about available rc parameters here :
	https://matplotlib.org/3.3.3/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file
	
		Arguments:
			figsize(int tuple): 
				figure size in inches
				Default : (16,9)

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
					'labelpad'		: 7.0,     	# space between label and axis
					'labelweight'	: 'bold',  	# weight of the x and y labels
					'xmargin'			: .05,
					'ymargin'			: .05}
	
	xtick = {'major.pad' : 10.0}

	ytick = {'major.pad' : 10.0}
	
	grid = {'color'			: 'b0b0b0',  	# grid color
					'linestyle'	: 'dashed',  	# solid
					'linewidth'	: .8,     	 	# in points
					'alpha'			: .5}     	 	# transparency, between 0.0 and 1.0

	legend = {'loc'					: 'upper left',
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
						'xtick' 	: xtick,
						'ytick'		: ytick,
						'grid' 		: grid, 
						'legend' 	: legend, 
						'figure' 	: figure, 
						'savefig' : savefig}

	for key, val in config.items():
		rc(key, **val)

def visualise_data(root_dir):
	''' 
	Visuzlise the data given the root directory including .csv files
	Creates a figure directory to store the results in the working directory 
	if it does not exist.

		Arguments:
			
			root_dir(str) : 
				path to the directory containing the csv files to be investigated
		
		Example:
			visualise_data(/home/data)
	'''

	# Output Directory
	figure_root = os.path.join(os.getcwd(), 'figure')
	if not os.path.exists(figure_root):
		os.makedirs(figure_root)

	# Check and get input paths and create output paths
	filenames = os.listdir(root_dir)
	filenames = [filename for filename in filenames if filename.endswith('.csv')]
	input_paths = [os.path.join(root_dir,filename) for filename in filenames]
	output_paths = [os.path.join(figure_root,filename.replace('.csv', '.png')) for filename in filenames]

	for _in, _out in zip(input_paths, output_paths):
		set_first_days = False if 'Train' in _in else True
		investigate_data(_in, _out, set_first_days)

def investigate_data(input_path, output_path, set_first_days=False):
	'''
	Given csv data with ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
	columns, preprocess the data in the .csv file to visualise 
	all features in one figure. 

		Arguments:

			input_path(str) :
				csv filepath to be investigated

			output_path(str) :
				output path to save the png file

			set_first_days(bool):
				set first days of the week as proper xticks.

		Example Usage:
			investigate_file('/home/data/Google_Stock_Price_Train.csv',/home/figure/train.png)
	'''

	title = (output_path.split(os.path.sep)[-1]).replace('_',' ').replace('.png', '')

	dataframe = pd.read_csv(input_path)
	assert dataframe.columns.tolist() == ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
	
	dataframe = dataframe.replace(',','', regex=True)
	dataframe.Date = pd.to_datetime(dataframe.Date)
	dataframe.set_index('Date', inplace=True)

	xticks = None
	if set_first_days:
		xticks = get_firstdays(dataframe.index.to_frame())
	plot_all(dataframe, title=title, xticks=xticks, filepath=output_path)

def plot_all(dataframe, title=None, xlabel='Date', xticks = None, ylabel_left='Price (USD)', ylabel_right='Volume (Millions)', savefig=True, filepath='figure'):
	'''
	Plot the stock records checked and prepoceseed by investigate_data() function.
	'Open', 'High', 'Low', 'Close', columns depended on the left hand side y axis,
	'Volume' column depended on the right hand side y axis and
	'Date' column depended on the x axis.

	Create two identical subfigures on top of each other to show volume and price
	features in the same figure. Offset prices to upwards to avoid intersections.

	Arguments:

			dataframe(pandas.Dataframe) : 
				dataframe to be visualised with 
				['Date'(index), 'Open', 'High', 'Low', 'Close', 'Volume'] columns

			title(str) :
				title of the figure(on top)
				Default : None
			
			xlabel(str) :
				x axis label for the figure
				Default : 'Date'

			xticks(list of float):
				list of indexes converted from dates using matplotlib.date.date2num() 
				functions indicating the xticks to be printed. If None, left this job 
				for the matplotlib.pyplot backend. 
				Default : None

			ylabel_left(str):
				left y axis label for the figure
				Default : 'Price (USD)'

			ylabel_right(str):
				right y axis label for the figure
				Default : 'Volume (Millions)'

			savefig(bool):
				save the figure if True, show the figure if False
				Default : True

			filepath(str):
				path for the figure to be saved in the case that savefig=True
				Default : 'figure'
	'''

	fig, ax_left = plt.subplots()
	ax_right = ax_left.twinx()  # set up the 2nd axis
	
	# Prices
	_max = max(dataframe.High.max(), dataframe.Close.astype(float).max())
	diff = _max - dataframe.Low.min()
	ax_left.fill_between(dataframe.index, dataframe.High.astype(float), dataframe.Low.astype(float), color='orange', alpha=.3, label='Range')
	ax_left.plot(dataframe.index, dataframe.Open.astype(float), color = 'red', label='Open')
	ax_left.plot(dataframe.index, dataframe.Close.astype(float), color = 'blue', label='Close')
	ax_left.set_ylabel(ylabel_left)
	ax_left.set_ylim(dataframe.Low.min() - diff, _max+diff*0.05)

	# Volume
	ax_right.bar(dataframe.index, dataframe.Volume.astype(int)/1e6, color = 'yellowgreen',label='Volume')
	ax_right.grid(b=False)
	ax_right.set_ylabel(ylabel_right)
	ax_right.set_ylim(0,dataframe.Volume.astype(int).max()/.5e6)

	if xticks:
		ax_left.axes.set_xticks(xticks)

	# Combine labels
	lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
	lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

	plt.sca(ax_left)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.xticks(ha="left")
	plt.legend(lines, labels)
	plt.autoscale(axis='x',tight=False)

	if(savefig):
		plt.savefig(filepath)
		print(f'{title} stock record plot saved in {filepath}\n')
	else:
		plt.show()

	plt.close()

def get_firstdays(dataframe):
	'''
	Get the indexes of the first days of the weeks to be used to declare 
	proper xticks in a pyplot figure. 

		Arguments:

			dataframe(pandas.DataFrame):
				1 column dataframe containing numpy.datetime64 objects

		Returns:

			day_list(list of float):
				list of floats indicating the float representaitons of the 
				dates generated by matplotlib.dates.date2num() function.
	'''

	mask = dataframe.applymap(lambda x: x.isocalendar()[2]).iloc[:,0].values
	datenum = dataframe.applymap(lambda x: mpl.dates.date2num(x)).iloc[:,0].values

	m_prev = 10;
	day_list = []
	for m,d in zip(mask,datenum):
		if(m < m_prev):
			day_list.append(d)
		m_prev = m;

	return day_list