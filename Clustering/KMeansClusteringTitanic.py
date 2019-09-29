# dataset: https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

style.use('ggplot')

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic-data.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}  # dictionary to map text value to an int id

        def convert_to_int(val):  # extracts id for a value 'val'
            return text_digit_vals[val]

        # if it is text-based data
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()  # get all values for specific column in a list
            unique_elements = set(column_contents)  # get set of unique elements from column values list
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))  # map converted values to df[column]
    return df


df = handle_non_numerical_data(df)
# print(df.head())

df.drop(['boat'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1). astype(float))  # drop survived column
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_data = np.array(X[i].astype(float))
    predict_data = predict_data.reshape(-1, len(predict_data))
    prediction = clf.predict(predict_data)
    if prediction[0] == y[i]:
        correct += 1

print(correct / len(X))
