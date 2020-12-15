# Google Stock Price Prediction

Stock market is the aggregation of buyers and sellers of stocks which represents ownership claims on businesses. In this project the aim is to build and compare multiple recurrently connected neural network models to forecast stock price values (Open, Close, High, Low) and trade volume.

## Dataset
Google stock records is used to train and test the networks. The dataset includes 1258 workday records from January 2012 to December 2016 to train; and 20 workday records in January 2017 to test.

Let's have a look at the csv files provided. The data is presented as follows: 

| Date     | Open 	| High 	 | Low 	  | Close  | Volume  |
| -------- | ------ | ------ | ------ | ------ | ------- | 
| 1/3/2017 | 778.81 | 789.63 | 775.8  | 786.14 | 1657300 |
| 1/4/2017 | 788.36	| 791.34 | 783.16 | 786.9  | 1073000 | 
| 1/5/2017 | 786.08	| 794.48 | 785.02 | 794.02 | 1335200 | 
| ...	   | ...   	| ...	 | ...	  |...	   |...	     |

Brief explanations for the features:

* **Date** : The day that the stock statistics is recorded.
* **Open** : The price of the first trade, the daily openning price
* **High** : The highest price at which the stock is traded during the course of the day
* **Low** : The lowest price at which the stock is traded during the course of the day
* **Close** : The price of the last trade during a regular trading session 
* **Volume** : The number of shares traded during a given period of time

*Available here:* *https://www.kaggle.com/medharawat/google-stock-price*

## Visualisation

First, let's take look at the training dataset. One of the interesting thing is that in 2014, there is a great decline at the closing price. The actual day that happened is 27 March 2014. The reason of that is, a stock split took place with a rate of 2002/1000 at that day. It has resulted in that the closing price has decreased by half. Actually there is another split in 27 April 2015 but the rate was 10027455/10000000, and the results are not that obvious to visually inspect. 

![Train Set](https://github.com/ugurcancakal/stockPrediction/blob/main/figure/Google_Stock_Price_Train.png)

The effects of the splits can be eliminated by scaling the closing prices to the effect of last decision. That is, the closing prices before 27 March 2014, can be divided by 2.002 and the closing prices before 27 April 2015 can be divided by 1.0027455. Doing the split adjustments accordingly, the resulting plot is like:

![Train Set (Adjusted)](https://github.com/ugurcancakal/stockPrediction/blob/main/figure/Google_Stock_Price_Train_2014-03-27_(2.002)_2015-04-27_(1.0027455).png)

Lastly in the test dataset, all aspect can be investigated clearly.

![Testset](https://github.com/ugurcancakal/stockPrediction/blob/main/figure/Google_Stock_Price_Test.png)

Further information about stock split:

*What is a Stock Split:* 

*https://www.investopedia.com/ask/answers/what-stock-split-why-do-stocks-split/#:~:text=A%20stock%20split%20is%20a,share%20held%20by%20a%20shareholder.*

*Google Stock Split History:* 

*https://www.splithistory.com/goog/#:~:text=The%20first%20split%20for%20GOOG,shareholder%20now%20owned%202002%20shares.*

## The Network

## Results
