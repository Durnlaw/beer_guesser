import pandas as pd
pd.set_option('display.max_columns', 30)

rawGrains = pd.read_csv('open-beer-database (mod).csv', delimiter = ';')
rawGrains =rawGrains.rename(columns={'Alcohol By Volume': 'abv'})

#. Cut down to just US, abv, and categoried beers
allUSAlc = rawGrains[rawGrains.abv != 0]
allUSAlc = allUSAlc[allUSAlc.Country == 'United States']
allUSAlc = allUSAlc[allUSAlc.cat_id != -1]
#. Remove white space from states
allUSAlc['State'].str.strip()
# print(allUSAlc.count())
test0 = allUSAlc.groupby('State')['id'].nunique()
# print('test0')
# print(test0)

regions = pd.read_csv('region_list.csv', delimiter=',')

#. Join the regions in two ways.
regByState = allUSAlc.merge(regions, left_on = 'State', right_on = 'State', how = 'inner')
# print(regByState.count())
test1 = regByState.groupby('State')['id'].nunique()
# print('test1')
# print (test1)
# print(list(regByState.columns))


regByAbbrv = allUSAlc.merge(regions, left_on = 'State', right_on = 'Abbrev', how = 'inner')
regByAbbrv.drop(['State_x'], axis =1, inplace =True)
regByAbbrv.rename(columns={'State_y': 'State'}, inplace = True)
# print(regByAbbrv.count())
test2 = regByAbbrv.groupby('Abbrev')['id'].nunique()
# print('test2')
# print(test2)
# print(list(regByAbbrv.columns))


#. Concat the dataframes back together. Drop the former State column, rename the new State column
frames = [regByState, regByAbbrv]
cleanedData = pd.concat(frames, sort=False)
print(cleanedData.head(2))
print(list(cleanedData.columns))

#> Data is cleaned up. Let's grab some ML models

treeData = cleanedData[['abv', 'region_cd', 'cat_id']]

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

#. Build a func to perform training
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
    print("Predicted values:")
    print(y_pred)
    return y_pred

#. Function to calculate accuracy
def cal_accuracy(y_test, y_pred):

    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred)*100)

    print("Report : ",
          classification_report(y_test, y_pred))

#. Driver code
def main():

    #. Building Phase
    X, Y, X_train, X_test, y_train, y_test = splitDataset(treeData)
    clf_gini = gini_train(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

    #. Operational Phase
    print("Results Using Gini Index:")

    #. Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("Results Using Entropy:")
    #. Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


#. Calling main function
if __name__ == "__main__":
    main()
