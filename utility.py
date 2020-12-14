'''
Google Stock Price Prediction Project
Utility functions

author: Ugurcan Cakal
201209
'''
from datetime import datetime, timedelta
import pandas as pd
import os 

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

	return filename;