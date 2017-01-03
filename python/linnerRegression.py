import quandl
import math
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing,cross_validation
import pandas

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Close'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
df.fillna(value=-9999999,inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)
forecast_col = 'Adj. Close'
df['label'] = df[forecast_col].shift(-forecast_out)
print(df.head())
df.dropna(inplace=True)
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
y = np.array(df['label'])
x_train,x_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.2)
clf = linear_model.LinearRegression()
clf.fit(x_train,y_train)
accuracy = clf.score(x_test,y_test)
print(accuracy)