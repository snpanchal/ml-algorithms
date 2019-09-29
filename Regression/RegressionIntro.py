import math
import datetime
import numpy as np
import quandl
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# import googl stocks dataset
df = quandl.get("WIKI/GOOGL", api_key="d5D-Njk3ayKKTND1ctJx")

# limit features used
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# define new columns using data we have
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# limit data frame (df) to only necessary columns
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# save name of forecast to another variable
forecast_col = 'Adj. Close'

# fill NaN in data with some outlier value (e.g. -99999)
df.fillna(-99999, inplace=True)

# predict 10% out of df (10% ahead of period of df)
forecast_out = int(math.ceil(0.1 * len(df)))

# label column for each row will predict close price (Adj. Close)
# 10% of df length in future
df['label'] = df[forecast_col].shift(-forecast_out)

# features are df without 'label' converted to np array
X = np.array(df.drop(['label'], 1))
# scale X to get it in range of -1 and 1
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


# drop NaN values
df.dropna(inplace=True)
y = np.array(df['label'])

# select random training data and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# use linear regression classifier
clf = LinearRegression(n_jobs=-1)
# train classifier using training data
clf.fit(X_train, y_train)

# serialize trained classifier (clf) by saving it in
# 'linearregression.pickle' file
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

# get serialized classifier from 'linearregression.pickle'
# file again to load in classifier
# (don't need to train classifier every time it's used)
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

# test on testing data
accuracy = clf.score(X_test, y_test)
# predict using sample values
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
# fill 'Forecast' column with NaN
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
