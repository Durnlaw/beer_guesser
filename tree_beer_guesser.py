import pandas as pd
pd.set_option('display.max_columns', 30)

#> Script Purpose: See if we can find any connection btw Alcohol by Volume and US Region
#> on the category (general type) of beer produced.

rawGrains = pd.read_csv('open-beer-database (mod).csv', delimiter = ';')
rawGrains =rawGrains.rename(columns={'Alcohol By Volume': 'abv'})
print('Shape Check', rawGrains.shape)
print('-----------------------------------------------------')
print(rawGrains.groupby('Country')['id'].nunique())

#. I saw in the orig data source that there were beers with ABV =0, outside of the US, and
#. with spurrious cat_id's. I quickly checked how many there were of each and cut them out
#. as need be. Also cut out an ABV of 99. That's silly.

abv_check = rawGrains.groupby('abv')['id'].nunique()
print('-----------------------------------------------------')
print('ABV List:', abv_check)
country_check = rawGrains.groupby('Country')['id'].nunique()
print('-----------------------------------------------------')
print('Country List:', country_check)
cat_check = rawGrains.groupby('cat_id')['id'].nunique()
print('-----------------------------------------------------')
print('Category List:', cat_check)
#. Do a quick verification that there is an unlabeled "Category" field that is "-1" in cat_id
cat_nm_check = rawGrains.groupby('Category')['id'].nunique()
print('-----------------------------------------------------')
print('Category Names:', cat_nm_check)

#. Now actually remove the things that we checked for in the above.
allUSAlc = rawGrains[rawGrains.abv != 0]
allUSAlc = allUSAlc[allUSAlc.abv < 99]
allUSAlc = allUSAlc[allUSAlc.Country == 'United States']
allUSAlc = allUSAlc[allUSAlc.cat_id != -1]
print('-----------------------------------------------------')
print('Final Raw Shape:', allUSAlc.shape)



#> Originally I considered limiting the model to look at states, not regions, but
#> I quickly realized that not all states have the same # of beers. Not close.
#. Here we prove the above point.
#. Start with removing white spaces as precaution
allUSAlc['State'].str.strip()

#. Do a group by and give it a visual once over.
state_list = allUSAlc.groupby('State')['id'].nunique()
print('-----------------------------------------------------')
print('Are States to limiting?')
print(state_list)

#. We conclude that it doesn't look great for just states. Bring in the regional data.
#! This file I maniupaled outside of pandas as I had to manually create the data file. 
#! Used this resource "https://apps.bea.gov/regional/docs/regions.cfm"
regions = pd.read_csv('region_list.csv', delimiter=',')


#! Above, we noticed that states were stored both as abbreviations as well as
#! simple names. Let's account for that now and make sure we can join to region regardless.

#. First we are going to join the regions table on State and State columns. With an inner
#. join this assures we are left with only full state names.
regByState = allUSAlc.merge(regions, left_on = 'State', right_on = 'State', how = 'inner')
print('-----------------------------------------------------')
#. Quick check here to see how many were full state names.
print(regByState.count())
state_check = regByState.groupby('State')['id'].nunique()
print('-----------------------------------------------------')
#. A quick check here shows that we have full names only.
print('state_check')
print (state_check)
print(list(regByState.columns))


#. Next we are joining the regions table on State(from beer) and Abbrev (from regions).
#. Keep an inner join here for the same reason above
regByAbbrv = allUSAlc.merge(regions, left_on = 'State', right_on = 'Abbrev', how = 'inner')
#. We also drop the old state column and change the new state column to "State"
regByAbbrv.drop(['State_x'], axis =1, inplace =True)
regByAbbrv.rename(columns={'State_y': 'State'}, inplace = True)
print('-----------------------------------------------------')
#. Quick check here to see how many were abbreviation names.
print(regByAbbrv.count())
abbrev_check = regByAbbrv.groupby('Abbrev')['id'].nunique()
print('-----------------------------------------------------')
#. A quick check here shows that we have abbrevs only.
print('abbrev_check')
print(abbrev_check)
print(list(regByAbbrv.columns))


#. Concat the dataframes back together. Count the number of rows here vs two fames
#. above
frames = [regByState, regByAbbrv]
cleanedData = pd.concat(frames, sort=False)
print('-----------------------------------------------------')
print(cleanedData.count())
print(list(cleanedData.columns))



#> Data is cleaned up. Let's get to the Decision Tree
#! Disclosure: I heavily made use of this individuals Decision Tree walkthrough
#! "https://www.geeksforgeeks.org/decision-tree-implementation-python/"

#. Should only need these three columns for the moment
treeData = cleanedData[['abv', 'region_cd', 'cat_id']]
print('-----------------------------------------------------')
print(treeData.groupby('cat_id').count())

#. Trying Trees
import sklearn
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def splitDataset(cleanedData):
    #. Separating target variable
    X = treeData.values[:, 0:2]
    Y = treeData.values[:, 2]

    #. Splitting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test

#. Build a func to perform training with gini
def gini_train(X_train, X_test, y_train):
    #. Classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                random_state=100, max_depth=3, min_samples_leaf=5)
    #. Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

#. Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):

    #. Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    #. Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


#. Function to make predictions
def prediction(X_test, clf_object):

    #. Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print('-----------------------------------------------------')
    print("Predicted values:")
    print(y_pred)
    return y_pred

#. Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print('-----------------------------------------------------')
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))
    print('-----------------------------------------------------')
    print("Accuracy : ",
          accuracy_score(y_test, y_pred)*100)
    print('-----------------------------------------------------')
    print("Report : ",
          classification_report(y_test, y_pred))

#. Driver code
def main():

    #. Building Phase
    X, Y, X_train, X_test, y_train, y_test = splitDataset(treeData)
    clf_gini = gini_train(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

    #. Operational Phase
    print('-----------------------------------------------------')
    print("Results Using Gini Index:")

    #. Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    print('-----------------------------------------------------')
    print("Results Using Entropy:")
    #. Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


#. Calling main function
if __name__ == "__main__":
    main()



#? Things to be improved: I think we should pick a better model, tune the model
#? more, or perhaps restructure the data to better express the nature of ABV
#? as it is mostly continuous while cat_id and region_cd are discrete. Ranges of
#? ABV might have been better.

#? Another thought here: I would like to add other data sources in to be used
#? in the future. For instance, pH of the local water in the state. Or even
#? better would be getting zip codes for each address through a google geocode
#? API set up.
