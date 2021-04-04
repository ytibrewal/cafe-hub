#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import calendar
from imblearn.over_sampling import SMOTE


# In[16]:


dataset = pd.read_csv('sample_data.csv')
dataset = dataset.dropna()
dataset = dataset[['Year', 'Month', 'SST', 'SSC', 'AT', 'RH', 'SLP', 'TC', 'TOTALOIL', 'Label']]

dataset.head()

# Plot all the data as lines
do=dataset.loc[dataset.Label =='PFZ'] 

dx=dataset.loc[dataset.Label =='NPFZ'] 

df=dataset.loc[dataset.Year ==2010] 
plt.plot(df['Month'], df['SST'], 'b-', label  = '2010', alpha = 1.0)
df=dataset.loc[dataset.Year ==2011] 
plt.plot(df['Month'], df['SST'], 'y-', label  = '2011', alpha = 1.0)
df=dataset.loc[dataset.Year ==2012] 
plt.plot(df['Month'], df['SST'], 'k-', label  = '2012', alpha = 0.8)
df=dataset.loc[dataset.Year ==2013] 
plt.plot(df['Month'], df['SST'], 'r-', label  = '2013', alpha = 0.3)
plt.plot(do['Month'], do['SST'], 'o',
             label="Preferred Fishing Zone")
plt.plot(dx['Month'], dx['SST'], 'x',
             label="Non-Preferred Fishing Zone")
plt.legend(); plt.xticks(rotation = '60');
plt.xlabel('Month_Number'); plt.ylabel('SST'); plt.title('SST values over the years');


# In[29]:


df=dataset.loc[dataset.Year ==2010] 
plt.plot(df['Month'], df['TOTALOIL'], 'b-', label  = '2010', alpha = 1.0)
df=dataset.loc[dataset.Year ==2011] 
plt.plot(df['Month'], df['TOTALOIL'], 'y-', label  = '2011', alpha = 1.0)
df=dataset.loc[dataset.Year ==2012] 
plt.plot(df['Month'], df['TOTALOIL'], 'k-', label  = '2012', alpha = 0.8)
df=dataset.loc[dataset.Year ==2013] 
plt.plot(df['Month'], df['TOTALOIL'], 'r-', label  = '2013', alpha = 0.3)
plt.plot(do['Month'], do['TOTALOIL'], 'o',
             label="Preferred Fishing Zone")
plt.plot(dx['Month'], dx['TOTALOIL'], 'x',
             label="Non-Preferred Fishing Zone")
plt.legend(); plt.xticks(rotation = '60');
plt.xlabel('Month_Number'); plt.ylabel('TOTAL OIL'); plt.title('Total Oil on the surface of the Water over the years');


# 

# In[17]:


#Converting the Label col to integer - 0 or 1 
#Normalization
dataset.loc[dataset.Label =="PFZ", 'Label'] = 1  
dataset.loc[dataset.Label =="NPFZ", 'Label'] = 0  
dataset.head()
dataset.describe()


# In[18]:


#Splitting of dataset
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

split = int(len(dataset)*0.6)
X_train, X_test, y_train, y_tests = X[:split], X[split:], y[:split], y[split:]

#Up-sampling
sm = SMOTE(random_state=12, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

#checking the shapes of the data 
print('Training data :', X_train_res.shape)
print('Training label:', y_train_res.shape)
print('Test data:', X_test.shape)
print('Test label:', y_tests.shape)
#Effect of upsampling
print (np.bincount(y_train))
print (np.bincount(y_train_res))


# In[19]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10, random_state = 40)
# Train the model on training data
rf.fit(X_train_res, y_train_res);


# In[24]:


# Use the forest's predict method on the test data
predictions = (rf.predict(X_test))
predictions =np.round (predictions)
# Calculate the absolute errors
errors = abs(predictions - y_tests)
# Print out the mean absolute error (mae) and accyrecy
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print (100-( errors.sum()/y_tests.sum()*100)  ,"%" )


# In[21]:


#For confussion matrix
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

conf_mat = confusion_matrix(y_true=y_tests, y_pred=predictions)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()


# In[22]:


#Most Important feature is calculated out
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',                                                                 
                                    ascending=False)
display (feature_importances)


# In[ ]:




