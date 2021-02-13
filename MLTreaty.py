#1: Importing libraries
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
import pandas as pd

""
# Things to do:
# 1. debug error "ValueError: Input contains NaN, infinity or a value too large for dtype('float32')." when running the script
# 2. Investigate number '4' when coverting y and n to 1 and 0. Look at the testing output for y.
""

# Clean missing values + "special" values from the imported csv file
missingvals = ['NA','OAUcharter','LoN', 'EEC', 'ILO', '"Amazon Cooperation"','Multilateral Agreement for the Establishment of an International Think Tank for Landlocked Developing Countries','55 (a)', '20 (a)', '20 (b)']
csv_data = pd.read_csv (r'/Users/kraynguyen1/Desktop/MLtreaty/FinalTreatyCombo.csv',na_values = missingvals) 

#csv_data.fillna(2)

# make a subset of necessary variables
df = pd.DataFrame(csv_data, columns= ['treatyNum','prec1','prec4'])

# capitalize y and n, replace with 1 for yes, 0 for no. Input for training model only accepts float data type
df['prec1'] = df['prec1'].str.capitalize()
df['prec1'] = df['prec1'].replace(['Y','N'],['1','0'])

df['prec4'] = df['prec4'].str.capitalize()
df['prec4'] = df['prec4'].replace(['Y','N'],['1','0'])

# cleaning 
df['treatyNum'] = df['treatyNum'].replace(['266-I-3822','I-54669'],['3822','54669'])

# x is treaty number and prec 1, y is prec 4
X = df.iloc[:, [0,1]].values

y = df.iloc[:, 2].values

# print treaty number, prec 1, prec4 column for testing purpose 
print(X,y)


# splits into train and test. training 80% testing 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train linear regression model
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()  
# regressor.fit(X_train, y_train)

# train random forest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 

#y_pred = regressor.predict(X_test)

# prediction
y_pred = classifier.predict(X_test)

# output testing 
dfoutput = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

print(dfoutput)


