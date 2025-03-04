#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import python libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


# for the python script to work the dataset csv file must be in the same folder as the current file
# load dataset to python notebook
dataSet = pd.read_csv('Data_Set.csv')

# drop ID as it is not needed 
dataSet = dataSet.drop(['ID'], axis=1)


# In[5]:


# print dataset
print(dataSet)


# In[6]:


# print summary 
dataSet.head()


# In[7]:


# Exploratory Data Analysis (EDA)
dataSet.shape


# In[8]:


# columns and data types 
dataSet.info()


# In[9]:


# summary of statistics 
dataSet.describe()


# In[10]:


# data distribution
plt.figure(figsize=(30,30))
for i, col in enumerate(dataSet.columns):
    plt.subplot(4,6,i+1)
    plt.hist(dataSet[col], alpha=0.3, color='b', density=True)
    plt.title(col)
    plt.xticks(rotation='vertical')


# In[11]:


# box plot to examine target life expectancy 
plt.boxplot(dataSet['TARGET_LifeExpectancy'])
plt.title('Target Life Expectancy')
plt.show()


# In[12]:


# relationship between target life expectancy and variables 
import seaborn as sns
plt.figure(figsize=(20,20))
for i, col in enumerate(dataSet.columns):
    plt.subplot(4,6,i+1)
    sns.scatterplot(data=dataSet, x=col, y='TARGET_LifeExpectancy')
    plt.title(col)

plt.xticks(rotation='vertical')
plt.show()


# In[13]:


# correlation between variables 
import seaborn as sns

f, ax = plt.subplots(figsize=(11, 9))
corr = dataSet.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
);


# In[14]:


# doing some data cleaning and removing outliers
for i in ['Device_returen']:
    dataSet.loc[dataSet[i] > 1000,i] = np.nan

for i in ['Obsolescence']:
    dataSet.loc[dataSet[i] > 1000,i] = np.nan

for i in ['Engine_failure']:
    dataSet.loc[dataSet[i] > 1000,i] = np.nan

dataSet = dataSet.dropna(axis = 0)


# In[15]:


dataSet_x = dataSet.drop(['TARGET_LifeExpectancy'], axis=1)
dataSet_y = dataSet['TARGET_LifeExpectancy']


# In[16]:


# randomly splitting data
from sklearn.model_selection import train_test_split

with pd.option_context('mode.chained_assignment', None):
    dataSet_x_train, dataSet_x_test, dataSet_y_train, dataSet_y_test = train_test_split(dataSet_x, dataSet_y, test_size=0.2, shuffle=True)
    
with pd.option_context('mode.chained_assignment', None):
    dataSet_x_train, dataSet_x_val, dataSet_y_train, dataSet_y_val = train_test_split(dataSet_x_train, dataSet_y_train, test_size=0.25, shuffle=True)


# In[17]:


print("Instances in the original dataset: {}\nInstances after splitting train: {}\nInstances after splitting test: {}\nInstances after splitting validation: {}"
      .format(dataSet.shape[0], dataSet_x_train.shape[0], dataSet_x_test.shape[0], dataSet_x_val.shape[0]))


# In[18]:


# checking splits to see if identical 
plt.figure(figsize=(30,30))
for i, col in enumerate(dataSet_x_train.columns):
    plt.subplot(4,6,i+1)
    plt.hist(dataSet_x_train[col], alpha=0.3, color='b', density=True)
    plt.title(col)
    plt.xticks(rotation='vertical')
    
plt.figure(figsize=(30,30))
for i, col in enumerate(dataSet_x_test.columns):
    plt.subplot(4,6,i+1)
    plt.hist(dataSet_x_train[col], alpha=0.3, color='r', density=True)
    plt.title(col)
    plt.xticks(rotation='vertical')


# In[19]:


# feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer

logNorm_attributes = ['Gas_Pressure','TotalExpenditure','IncomeCompositionOfResources','RD','Company_Confidence','Device_confidence']
minmax_attributes = list(set(dataSet_x.columns).difference(set(logNorm_attributes)))

dataSet_x_train_scaled = dataSet_x_train.copy()
dataSet_x_test_scaled = dataSet_x_test.copy()
dataSet_x_val_scaled = dataSet_x_val.copy()


minmaxscaler = MinMaxScaler().fit(dataSet_x_train_scaled.loc[:, minmax_attributes])
dataSet_x_train_scaled.loc[:, minmax_attributes] = minmaxscaler.transform(dataSet_x_train_scaled.loc[:, minmax_attributes])
dataSet_x_test_scaled.loc[:, minmax_attributes] = minmaxscaler.transform(dataSet_x_test_scaled.loc[:, minmax_attributes])
dataSet_x_val_scaled.loc[:, minmax_attributes] = minmaxscaler.transform(dataSet_x_val_scaled.loc[:, minmax_attributes])

powertransformer = PowerTransformer(method='yeo-johnson', standardize=False).fit(dataSet_x_train.loc[:, logNorm_attributes])
dataSet_x_train_scaled.loc[:, logNorm_attributes] = powertransformer.transform(dataSet_x_train.loc[:, logNorm_attributes])
dataSet_x_test_scaled.loc[:, logNorm_attributes] = powertransformer.transform(dataSet_x_test.loc[:, logNorm_attributes])
dataSet_x_val_scaled.loc[:, logNorm_attributes] = powertransformer.transform(dataSet_x_val.loc[:, logNorm_attributes])

minmaxscaler_pt = MinMaxScaler().fit(dataSet_x_train_scaled.loc[:, logNorm_attributes])
dataSet_x_train_scaled.loc[:, logNorm_attributes] = minmaxscaler_pt.transform(dataSet_x_train_scaled.loc[:, logNorm_attributes])
dataSet_x_test_scaled.loc[:, logNorm_attributes] = minmaxscaler_pt.transform(dataSet_x_test_scaled.loc[:, logNorm_attributes])
dataSet_x_val_scaled.loc[:, logNorm_attributes] = minmaxscaler_pt.transform(dataSet_x_val_scaled.loc[:, logNorm_attributes])

# code referenced from Lab Week 4, 30/03/23


# In[20]:


# comparing plots before and after scaling
plt.figure(figsize=(20,20))
for i, col in enumerate(dataSet_x_train_scaled.columns):
    plt.subplot(4,6,i+1)
    plt.hist(dataSet_x_train_scaled[col], alpha=0.3, color='b', density=True)
    plt.hist(dataSet_x_test_scaled[col], alpha=0.3, color='r', density=True)
    plt.title(col)
    plt.xticks(rotation='vertical')
    
# code referenced from Lab Week 4, 30/03/23


# In[21]:


# build a baseline regression model
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression().fit(dataSet_x_train_scaled, dataSet_y_train)


# In[22]:


# print the parameter of the fitted model
print("Parameter of the Linear model: ", model_lr.coef_)
print("Intercept of the Linear model: ", model_lr.intercept_)


# In[23]:


# write predictions to csv file 
dataSet_y_test_pred = model_lr.predict(dataSet_x_test_scaled)

prediction = pd.DataFrame(dataSet_y_test_pred, columns=['predictions']).to_csv('prediction.csv')


# In[24]:


# compare predictions vs actual
fig, ax = plt.subplots()
ax.scatter(dataSet_y_test, dataSet_y_test_pred, s=25, cmap=plt.cm.coolwarm, zorder=10)

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.plot(lims, [np.mean(dataSet_y_train),]*2, 'r--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')

plt.show()

# code referenced from Lab Week 4, 30/03/23


# In[25]:


# r^2 value to measure how well model fits
from sklearn.metrics import r2_score

r2_lr = r2_score(dataSet_y_test, dataSet_y_test_pred)
print('The R^2 value for the linear regression model is: {:.3f}'.format(r2_lr))


# In[26]:


# test r^2 value on unscaled data
model_us_lr = LinearRegression().fit(dataSet_x_train, dataSet_y_train)
dataSet_y_test_us_pred = model_us_lr.predict(dataSet_x_test)

r2_us_lr = r2_score(dataSet_y_test, dataSet_y_test_us_pred)
print('The R^2 value for the linear regression model without feature scaling is: {:.3f}'.format(r2_us_lr))


# In[27]:


# check deviation from actual value
fig, ax = plt.subplots()
ax.scatter(dataSet_y_test, dataSet_y_test-dataSet_y_test_pred, s=25, cmap=plt.cm.coolwarm, zorder=10)

xlims = ax.get_xlim()
ax.plot(xlims, [0.0,]*2, 'k--', alpha=0.75, zorder=0)
ax.set_xlim(xlims)

plt.xlabel('Actual Life Expectancy')
plt.ylabel('Residual')

plt.show()

# code referenced from Lab Week 4, 30/03/23


# In[28]:


# examine feature importance by looking at model coefficients
coefs = pd.DataFrame(
    model_lr.coef_  * dataSet_x_train_scaled.std(axis=0),
    columns=['Coefficient importance'], index=dataSet_x_train_scaled.columns
)
coefs.sort_values(by=['Coefficient importance']).plot(kind='barh', figsize=(9, 7))
plt.title('Ridge model, small regularization')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)

# code referenced from Lab Week 4, 30/03/23


# In[29]:


from sklearn.inspection import permutation_importance

r = permutation_importance(model_lr, dataSet_x_test_scaled, dataSet_y_test, n_repeats=30)
inx = np.argsort(r.importances_mean)

plt.barh(dataSet_x_test_scaled.columns[inx], r.importances_mean[inx])
plt.xticks(rotation='vertical')
plt.show()

# code referenced from Lab Week 4, 30/03/23


# In[30]:


# apply regularisation
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

# L2 (ridge) regularisation
# create ridge regression object
ridge_reg = Ridge()

# fit ridge regression model to training data
ridge_reg.fit(dataSet_x_train_scaled, dataSet_y_train)

# make predictions on the test set using the ridge model
ridge_y_pred = ridge_reg.predict(dataSet_x_test_scaled)

# calculate the mean squared error of the ridge model predictions
ridge_mse = mean_squared_error(dataSet_y_test, ridge_y_pred)
print("Ridge model MSE: ", ridge_mse)

# L1 (lasso) regularisation
# create lasso regression object
lasso_reg = Lasso()

# fit lasso regression model to training data
lasso_reg.fit(dataSet_x_train_scaled, dataSet_y_train)

# make predictions on the test set using the lasso model
lasso_y_pred = lasso_reg.predict(dataSet_x_test_scaled)

# calculate the mean squared error of the lasso model predictions
lasso_mse = mean_squared_error(dataSet_y_test, lasso_y_pred)
print("Lasso model MSE: ", lasso_mse)


# In[31]:


# apply hyperparameter tuning  
from sklearn.model_selection import GridSearchCV

# set up hyperparameters 
params = {'alpha': [0, 0.01, 0.1, 1, 10]}

# create grid search object
grid_search = GridSearchCV(estimator=ridge_reg, param_grid=params, scoring='neg_mean_squared_error', cv=5)

# fit grid search object to training data
grid_search.fit(dataSet_x_train_scaled, dataSet_y_train)

# make predictions on the test set using the best model found by grid search
gs_y_pred = grid_search.predict(dataSet_x_test_scaled)

# calculate the mean squared error of the best model prediction
gs_mse = mean_squared_error(dataSet_y_test, gs_y_pred)
print("Best model MSE: ", gs_mse)

# print best hyperparameters found by grid search
print("Best hyperparameters: ", grid_search.best_params_)

# print best score found by grid search
print("Best score: ", grid_search.best_score_)


# In[32]:


# mse of baseline model
mse = mean_squared_error(dataSet_y_test, dataSet_y_test_pred)

# compare mse of all models
print("Lasso model MSE: ", lasso_mse)
print("Ridge model MSE: ", ridge_mse)
print("Best model MSE: ", gs_mse)
print("Base model MSE: ", mse)


# In[33]:


# baseline model
lin_reg = LinearRegression().fit(dataSet_x_train_scaled, dataSet_y_train)
print(f"Linear Regression Training set score: {lin_reg.score(dataSet_x_train_scaled, dataSet_y_train):.2f}")
print(f"Linear Regression Test set score: {lin_reg.score(dataSet_x_test_scaled, dataSet_y_test):.2f}")

# L2 regularisation
ridge_reg = Ridge(alpha=0.01).fit(dataSet_x_train_scaled, dataSet_y_train)
print(f"Ridge Regression Training set score: {ridge_reg.score(dataSet_x_train_scaled, dataSet_y_train):.2f}")
print(f"Ridge Regression Test set score: {ridge_reg.score(dataSet_x_test_scaled, dataSet_y_test):.2f}")

# L1 regularisation
lasso_reg = Lasso(alpha=0.01).fit(dataSet_x_train_scaled, dataSet_y_train)
print(f"Lasso Regression Training set score: {lasso_reg.score(dataSet_x_train_scaled, dataSet_y_train):.2f}")
print(f"Lasso Regression Test set score: {lasso_reg.score(dataSet_x_test_scaled, dataSet_y_test):.2f}")

# code referenced from Mehdi Lotfinejad, Regularization in Machine Learning, 04/04/23


# In[34]:


# train the final model on the full dataset with the best hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(dataSet_x_train_scaled, dataSet_y_train)

# evaluate the model on a holdout dataset
best_model_pred = best_model.predict(dataSet_x_val_scaled)
val_mse = mean_squared_error(dataSet_y_val, best_model_pred)
print("Best MSE: ", val_mse)

base_model_pred = model_lr.predict(dataSet_x_val_scaled)
val_mse = mean_squared_error(dataSet_y_val, base_model_pred)
print("Base MSE: ", val_mse)

