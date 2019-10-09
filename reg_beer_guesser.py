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
# print(cleanedData.head(2))
print(list(cleanedData.columns))

#> Data is cleaned up. Let's grab some ML models

treeData = cleanedData[['abv', 'region_cd', 'cat_id']]

print(treeData.groupby('cat_id').count())

import numpy as np
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# %matplotlib inline

#. Make a test plot for general trends
treeData.plot(x='cat_id', y='abv', style='o')
plt.title('cat_id vs abv')
plt.xlabel('cat_id')
plt.ylabel('abv')
# plt.show()

#. Set up indep and dep variables
X = treeData['cat_id'].values.reshape(-1, 1)
y = treeData['abv'].values.reshape(-1, 1)

#. Train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#. Train the alg
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#. To retrieve the intercept:
print(regressor.intercept_)  
#. For retrieving the slope:
print(regressor.coef_)

#. Prediction time
y_pred = regressor.predict(X_test)

#. Checker
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

#. Graph check
df1 = df.head(25)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


#. Scatter plot with best fit
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))