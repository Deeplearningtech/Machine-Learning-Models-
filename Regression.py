#pip install numpy

#pip install scipy

#pip install scikit-learn

#pip install matplotlib

#pip install quandl


import pandas as pd
import quandl

df = quandl.get("WIKI/GOOGL")

print(df.head())
#Thus, let's go ahead and pair down our original dataframe a bit:
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

# Let's go ahead and transform our data next:
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0

#Next, we'll do daily percent change:
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.


#Now we will define a new dataframe as:

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())


#Regression - Features and Labels
import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression



#math.ceil(x)
#Return the ceiling of x as a float,
#the smallest integer value greater than or equal to x.

#Forcost, fill missing data and return ceiling as float, smallest integer values
#greater than or equal to x
#In our case, we've decided the features are a bunch of the current values, and the label shall be the price, in the future, where the future is 1% of the entire length of the dataset out. We'll assume all current columns are our features,
#so we'll add a new column with a simple pandas operation:
#We're saying we want to forecast out 1% of the entire length of the dataset
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))


#In our case, we've decided the features are a bunch of the current values,
#and the label shall be the price, in the future, where the future is 1% of the entire length of the dataset out.
#We'll assume all current columns are our features, so we'll add a new column with a simple pandas operation:

df['label'] = df[forecast_col].shift(-forecast_out)


#Regression - Training and Testing
#We'll then drop any still NaN information from the dataframe:







#df.dropna(inplace=True)       
#X = np.array(df.drop(['label'], 1))
#y = np.array(df['label'])
#X = preprocessing.scale(X)
#y = np.array(df['label'])
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#clf = svm.SVR()
#clf = LinearRegression()
#clf.fit(X_train, y_train)
#confidence = clf.score(X_test, y_test)
#print(confidence)
#clf = LinearRegression()
#clf = LinearRegression(n_jobs=-1)
#for k in ['linear','poly','rbf','sigmoid']:
#    clf = svm.SVR(kernel=k)
#    clf.fit(X_train, y_train)
#   confidence = clf.score(X_test, y_test)
#    print(k,confidence)
#Predict
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

forecast_set = clf.predict(X_lately)
print(forecast_set, confidence, forecast_out)


#visualizing this information
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
df['Forecast'] = np.nan


last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()




