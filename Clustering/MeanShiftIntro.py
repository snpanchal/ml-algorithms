# dataset: https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing, model_selection
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
original_df = pd.DataFrame.copy(df)
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

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centres = clf.cluster_centers_

original_df['cluster_group'] = np.nan

# add labels to "cluster_group" column
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]
n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]  # get all feature sets with current cluster group
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate
print(survival_rates)
