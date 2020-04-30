#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, r2_score

import eli5
from eli5.sklearn import PermutationImportance


# <font size = 4>
# Data cleaning

# In[2]:


non_gk_df = pd.read_csv('non_gk_can_raw.csv')


# In[3]:


non_gk_df.head()


# In[4]:


pd.set_option('display.max_columns', 500)
non_gk_df = non_gk_df.drop('Unnamed: 0', axis = 1)


# In[5]:


non_gk_df.head()


# In[6]:


df = non_gk_df


# In[7]:


df.columns


# In[8]:


model_df = df[['Peak Overall','Age', 'overall', 'potential','height_cm', 'weight_kg',
               'international_reputation','weak_foot', 'skill_moves', 'pace', 'shooting',
               'passing', 'dribbling', 'defending', 'physic','attacking_crossing',
               'attacking_finishing', 'attacking_heading_accuracy',
               'attacking_short_passing', 'attacking_volleys', 'skill_dribbling',
               'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
               'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
               'movement_agility', 'movement_reactions', 'movement_balance',
               'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
               'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
               'mentality_positioning', 'mentality_vision', 'mentality_penalties',
               'defending_marking', 'defending_standing_tackle',
               'defending_sliding_tackle']]


# In[9]:


model_df = model_df.dropna()


# In[10]:


model_df


# In[11]:


chang_list = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
               'attacking_short_passing', 'attacking_volleys', 'skill_dribbling',
               'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
               'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
               'movement_agility', 'movement_reactions', 'movement_balance',
               'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
               'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
               'mentality_positioning', 'mentality_vision', 'mentality_penalties',
               'defending_marking', 'defending_standing_tackle',
               'defending_sliding_tackle']


# In[12]:


for i in chang_list:
    model_df[i] = model_df[i].str[:2]


# In[13]:


type_list = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
               'attacking_short_passing', 'attacking_volleys', 'skill_dribbling',
               'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
               'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
               'movement_agility', 'movement_reactions', 'movement_balance',
               'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
               'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
               'mentality_positioning', 'mentality_vision', 'mentality_penalties',
               'defending_marking', 'defending_standing_tackle',
               'defending_sliding_tackle']


# In[14]:


for i in type_list:
    model_df[i] = model_df[i].astype(float)


# In[15]:


model_df.dtypes


# <font size = 4>
# Preparation for models

# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


N_bs = 100


# In[18]:


X = model_df[['Age', 'overall', 'height_cm', 'weight_kg',
       'international_reputation', 'weak_foot', 'skill_moves', 'pace',
       'shooting', 'passing', 'dribbling', 'defending', 'physic',
       'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions',
       'mentality_positioning', 'mentality_vision', 'mentality_penalties',
       'defending_marking', 'defending_standing_tackle',
       'defending_sliding_tackle']]

y = model_df[['Peak Overall']]


# <font size = 4>
# Standardization of attributes

# In[19]:


from sklearn.preprocessing import StandardScaler


# ## Start Models horse racing

# In[19]:


from sklearn.linear_model import LinearRegression, SGDRegressor


# ### OLS Model

# In[20]:


MSE_vec_ols = np.zeros(N_bs)


# In[23]:


for bs_ind in range(N_bs):
    x_train, x_test, y_train, y_test =         train_test_split(X, y, test_size=0.25, random_state=bs_ind)
    
    scaler = StandardScaler()  
    scaler.fit(x_train)  
    x_train_stand = scaler.transform(x_train)  
    x_test_stand = scaler.transform(x_test)  

    scaler = StandardScaler()
    scaler.fit(y_train)
    y_train_stand = scaler.transform(y_train)
    y_test_stand = scaler.transform(y_test)
    
    LR = LinearRegression()
    lr = LR.fit(x_train_stand, y_train_stand)
    y_pred = lr.predict(x_test_stand)
    MSE_vec_ols[bs_ind] = mean_squared_error(y_test_stand, y_pred)
    print('MSE for test set', bs_ind, ' is', MSE_vec_ols[bs_ind])


# In[26]:


mse_min = 0.3
for i, mse in enumerate(MSE_vec_ols):
    if mse < mse_min:
        mse_min = mse
        i_min = i


# In[27]:


print('Linear Regression Model\'s MSE is', mse_min)
print('The random state for that MSE is', i_min)


# In[29]:


MSE_bs_ols = MSE_vec_ols.mean()
MSE_bs_ols_std = MSE_vec_ols.std()
print('test estimate MSE bootstrap=', MSE_bs_ols,
      'test estimate MSE standard err=', MSE_bs_ols_std)


# In[30]:


train, test = train_test_split(model_df, test_size=0.25, random_state=i_min)

x_train = train[['Age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]
y_train = train[['Peak Overall']]

x_test = test[['Age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]
y_test = test[['Peak Overall']]

y_expert = test[['potential']]

scaler = StandardScaler()  
scaler.fit(x_train)  
x_train_stand = scaler.transform(x_train)  
x_test_stand = scaler.transform(x_test)  

scaler = StandardScaler()
scaler.fit(y_train)
y_train_stand = scaler.transform(y_train)
y_test_stand = scaler.transform(y_test)
y_expert_stand = scaler.transform(y_expert)

LR = LinearRegression()
lr = LR.fit(x_train_stand, y_train_stand)
y_pred = lr.predict(x_test_stand)

m1 = mean_squared_error(y_test_stand, y_pred)
m2 = mean_squared_error(y_test_stand, y_expert_stand)
print('Linear Regression Model\'s MSE is', m1)
print('Expert Guess\'s MSE is', m2)


# In[31]:


perm = PermutationImportance(lr, random_state=i_min).fit(x_test_stand, y_test_stand)
eli5.show_weights(perm, feature_names = x_test.columns.tolist(), top=50)


# ### SGD Model

# In[32]:


MSE_vec_sgd = np.zeros(N_bs)


# In[34]:


for bs_ind in range(N_bs):
    x_train, x_test, y_train, y_test =         train_test_split(X, y, test_size=0.25, random_state=bs_ind)
    
    scaler = StandardScaler()  
    scaler.fit(x_train)  
    x_train_stand = scaler.transform(x_train)  
    x_test_stand = scaler.transform(x_test)  

    scaler = StandardScaler()
    scaler.fit(y_train)
    y_train_stand = scaler.transform(y_train)
    y_test_stand = scaler.transform(y_test)
    
    SGD = SGDRegressor(random_state=bs_ind)
    param_dist1 = {'penalty': ['l1', 'l2'], 'alpha': sp_uniform(1e-5, 10.0)}
    sgd_lr = RandomizedSearchCV(SGD, param_dist1, 
                n_iter=200, n_jobs=-1, cv=5, random_state=25, scoring='neg_mean_squared_error')
    sgd = sgd_lr.fit(x_train_stand, y_train_stand)
    
    y_pred = sgd.predict(x_test_stand)
    
    MSE_vec_sgd[bs_ind] = mean_squared_error(y_test_stand, y_pred)
    print('MSE for test set', bs_ind, ' is', MSE_vec_sgd[bs_ind])


# In[35]:


mse_min = 0.3
for i, mse in enumerate(MSE_vec_sgd):
    if mse < mse_min:
        mse_min = mse
        i_min = i


# In[36]:


print('SGD Model\'s MSE is', mse_min)
print('The random state for that MSE is', i_min)


# In[37]:


MSE_bs_sgd = MSE_vec_sgd.mean()
MSE_bs_sgd_std = MSE_vec_sgd.std()
print('test estimate MSE bootstrap=', MSE_bs_sgd,
      'test estimate MSE standard err=', MSE_bs_sgd_std)


# In[38]:


train, test = train_test_split(model_df, test_size=0.25, random_state=i_min)

x_train = train[['Age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]
y_train = train[['Peak Overall']]

x_test = test[['Age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]
y_test = test[['Peak Overall']]

y_expert = test[['potential']]

scaler = StandardScaler()  
scaler.fit(x_train)  
x_train_stand = scaler.transform(x_train)  
x_test_stand = scaler.transform(x_test)  

scaler = StandardScaler()
scaler.fit(y_train)
y_train_stand = scaler.transform(y_train)
y_test_stand = scaler.transform(y_test)
y_expert_stand = scaler.transform(y_expert)

SGD = SGDRegressor(random_state=i_min)
param_dist1 = {'penalty': ['l1', 'l2'], 'alpha': sp_uniform(1e-5, 10.0)}
sgd_lr = RandomizedSearchCV(SGD, param_dist1, 
            n_iter=200, n_jobs=-1, cv=5, random_state=25, scoring='neg_mean_squared_error')
sgd = sgd_lr.fit(x_train_stand, y_train_stand)

print('Optimal tuning parameter values:\n', sgd.best_params_)
print('MSE of the optimal results:', abs(sgd.best_score_))

y_pred = sgd.predict(x_test_stand)
m1 = mean_squared_error(y_test_stand, y_pred)
m2 = mean_squared_error(y_test_stand, y_expert_stand)
print('SGD Regression Model\'s MSE is', m1)
print('Expert Guess\'s MSE is', m2)


# In[39]:


perm = PermutationImportance(sgd, random_state=i_min).fit(x_test_stand, y_test_stand)
eli5.show_weights(perm, feature_names = x_test.columns.tolist(), top=50)


# ### Random Forest Regressor

# In[40]:


MSE_vec_rf = np.zeros(N_bs)


# In[41]:


param_dist2 = {'n_estimators': sp_randint(10, 200),
               'max_depth': sp_randint(2, 4),
               'min_samples_split': sp_randint(2, 20),
               'min_samples_leaf': sp_randint(2, 20),
               'max_features': sp_randint(1, 4)}


# In[42]:


for bs_ind in range(N_bs):
    x_train, x_test, y_train, y_test =         train_test_split(X, y, test_size=0.25, random_state=bs_ind)
    
    scaler = StandardScaler()  
    scaler.fit(x_train)  
    x_train_stand = scaler.transform(x_train)  
    x_test_stand = scaler.transform(x_test)  

    scaler = StandardScaler()
    scaler.fit(y_train)
    y_train_stand = scaler.transform(y_train)
    y_test_stand = scaler.transform(y_test)
    
    RFC = RandomForestRegressor(bootstrap=True,oob_score=True, random_state=bs_ind)
    rscv_rf = RandomizedSearchCV(RFC, param_dist2, 
             n_iter=200, n_jobs=-1, cv=5, random_state=25, scoring='neg_mean_squared_error')

    rf = rscv_rf.fit(x_train_stand, y_train_stand)
    
    y_pred = rf.predict(x_test_stand)
    
    MSE_vec_rf[bs_ind] = mean_squared_error(y_test_stand, y_pred)
    print('MSE for test set', bs_ind, ' is', MSE_vec_rf[bs_ind])


# In[44]:


mse_min = 0.5
for i, mse in enumerate(MSE_vec_rf):
    if mse < mse_min:
        mse_min = mse
        i_min = i
print('Random Forest Regression Model\'s MSE is', mse_min)
print('The random state for that MSE is', i_min)


# In[45]:


MSE_bs_rf = MSE_vec_rf.mean()
MSE_bs_rf_std = MSE_vec_rf.std()
print('test estimate MSE bootstrap=', MSE_bs_rf,
      'test estimate MSE standard err=', MSE_bs_rf_std)


# In[46]:


train, test = train_test_split(model_df, test_size=0.25, random_state=i_min)

x_train = train[['Age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]
y_train = train[['Peak Overall']]

x_test = test[['Age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]
y_test = test[['Peak Overall']]

y_expert = test[['potential']]

scaler = StandardScaler()  
scaler.fit(x_train)  
x_train_stand = scaler.transform(x_train)  
x_test_stand = scaler.transform(x_test)  

scaler = StandardScaler()
scaler.fit(y_train)
y_train_stand = scaler.transform(y_train)
y_test_stand = scaler.transform(y_test)
y_expert_stand = scaler.transform(y_expert)

RFC = RandomForestRegressor(bootstrap=True,oob_score=True, random_state=i_min)
rscv_rf = RandomizedSearchCV(RFC, param_dist2, 
         n_iter=200, n_jobs=-1, cv=5, random_state=25, scoring='neg_mean_squared_error')

rf = rscv_rf.fit(x_train_stand, y_train_stand)

print('Optimal tuning parameter values:\n', rf.best_params_)
print('MSE of the optimal results:', abs(rf.best_score_))

y_pred = rscv_rf.predict(x_test_stand)

m1 = mean_squared_error(y_test_stand, y_pred)
m2 = mean_squared_error(y_test_stand, y_expert_stand)
print('Random Forest Regression Model\'s MSE is', m1)
print('Expert Guess\'s MSE is', m2)


# In[47]:


perm = PermutationImportance(rf, random_state=i_min).fit(x_test_stand, y_test_stand)
eli5.show_weights(perm, feature_names = x_test.columns.tolist(), top=50)


# ### Support Vector Regression

# In[48]:


MSE_vec_svm = np.zeros(N_bs)


# In[49]:


param_dist3 = {'C': sp_uniform(loc=0.1, scale=10.0), 'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
               'gamma': ['scale', 'auto'],
               'shrinking': [True, False]}


# In[50]:


for bs_ind in range(N_bs):
    x_train, x_test, y_train, y_test =         train_test_split(X, y, test_size=0.25, random_state=bs_ind)
    
    scaler = StandardScaler()  
    scaler.fit(x_train)  
    x_train_stand = scaler.transform(x_train)  
    x_test_stand = scaler.transform(x_test)  

    scaler = StandardScaler()
    scaler.fit(y_train)
    y_train_stand = scaler.transform(y_train)
    y_test_stand = scaler.transform(y_test)
    
    svr = SVR()
    rscv_SVR = RandomizedSearchCV(svr, param_dist3, 
            n_iter=10, n_jobs=-1, cv=5, random_state=bs_ind, scoring='neg_mean_squared_error')

    random_SVR = rscv_SVR.fit(x_train_stand, y_train_stand)
    
    y_pred = random_SVR.predict(x_test_stand)
    
    MSE_vec_svm[bs_ind] = mean_squared_error(y_test_stand, y_pred)
    print('MSE for test set', bs_ind, ' is', MSE_vec_svm[bs_ind])


# In[52]:


mse_min = 0.5
for i, mse in enumerate(MSE_vec_svm):
    if mse < mse_min:
        mse_min = mse
        i_min = i
print('Support Vector Machine Model\'s MSE is', mse_min)
print('The random state for that MSE is', i_min)


# In[53]:


MSE_bs_svm = MSE_vec_svm.mean()
MSE_bs_svm_std = MSE_vec_svm.std()
print('test estimate MSE bootstrap=', MSE_bs_svm,
      'test estimate MSE standard err=', MSE_bs_svm_std)


# In[54]:


train, test = train_test_split(model_df, test_size=0.25, random_state=i_min)

x_train = train[['Age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]
y_train = train[['Peak Overall']]

x_test = test[['Age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]
y_test = test[['Peak Overall']]

y_expert = test[['potential']]

scaler = StandardScaler()  
scaler.fit(x_train)  
x_train_stand = scaler.transform(x_train)  
x_test_stand = scaler.transform(x_test)  

scaler = StandardScaler()
scaler.fit(y_train)
y_train_stand = scaler.transform(y_train)
y_test_stand = scaler.transform(y_test)
y_expert_stand = scaler.transform(y_expert)

svr = SVR()
rscv_SVR = RandomizedSearchCV(svr, param_dist3, 
            n_iter=10, n_jobs=-1, cv=5, random_state=i_min, scoring='neg_mean_squared_error')
random_SVR = rscv_SVR.fit(x_train_stand, y_train_stand)

print('Optimal tuning parameter values:\n', random_SVR.best_params_)
print('MSE of the optimal results:', abs(random_SVR.best_score_))

y_pred = random_SVR.predict(x_test_stand)

m1 = mean_squared_error(y_test_stand, y_pred)
m2 = mean_squared_error(y_test_stand, y_expert_stand)

print('Support Vector Machine Model\'s MSE is', m1)
print('Expert Guess\'s MSE is', m2)


# In[55]:


perm = PermutationImportance(random_SVR, random_state=i_min).fit(x_test_stand, y_test_stand)
eli5.show_weights(perm, feature_names = x_test.columns.tolist(), top=50)


# ### MLP Model

# In[20]:


from sklearn.neural_network import MLPRegressor


# In[57]:


MSE_vec_mlp = np.zeros(N_bs)


# In[21]:


param_dist4 = {'hidden_layer_sizes': sp_randint(1, 100),
               'activation': ['logistic', 'relu'],
               'alpha': sp_uniform(0.1, 10.0)}


# In[59]:


for bs_ind in range(N_bs):
    x_train, x_test, y_train, y_test =         train_test_split(X, y, test_size=0.25, random_state=bs_ind)
    
    scaler = StandardScaler()  
    scaler.fit(x_train)  
    x_train_stand = scaler.transform(x_train)  
    x_test_stand = scaler.transform(x_test)  

    scaler = StandardScaler()
    scaler.fit(y_train)
    y_train_stand = scaler.transform(y_train)
    y_test_stand = scaler.transform(y_test)
    
    mlp = MLPRegressor(activation='tanh', alpha=1, random_state=bs_ind)
    rscv_MLP = RandomizedSearchCV(mlp, param_dist4, n_iter=10, n_jobs=-1, cv=5, random_state=25, scoring='neg_mean_squared_error')
    random_MLP = rscv_MLP.fit(x_train_stand, y_train_stand)
    
    y_pred = random_MLP.predict(x_test_stand)
    
    MSE_vec_mlp[bs_ind] = mean_squared_error(y_test_stand, y_pred)
    print('MSE for test set', bs_ind, ' is', MSE_vec_mlp[bs_ind])


# In[60]:


mse_min = 0.5
for i, mse in enumerate(MSE_vec_mlp):
    if mse < mse_min:
        mse_min = mse
        i_min = i
print('MLP Model\'s MSE is', mse_min)
print('The random state for that MSE is', i_min)


# In[61]:


MSE_bs_mlp = MSE_vec_mlp.mean()
MSE_bs_mlp_std = MSE_vec_mlp.std()
print('test estimate MSE bootstrap=', MSE_bs_mlp,
      'test estimate MSE standard err=', MSE_bs_mlp_std)


# In[22]:


i_min = 34


# In[23]:


train, test = train_test_split(model_df, test_size=0.25, random_state=i_min)

x_train = train[['Age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]
y_train = train[['Peak Overall']]

x_test = test[['Age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]
y_test = test[['Peak Overall']]

y_expert = test[['potential']]

scaler = StandardScaler()  
scaler.fit(x_train)  
x_train_stand = scaler.transform(x_train)  
x_test_stand = scaler.transform(x_test)  

scaler = StandardScaler()
scaler.fit(y_train)
y_train_stand = scaler.transform(y_train)
y_test_stand = scaler.transform(y_test)
y_expert_stand = scaler.transform(y_expert)

mlp = MLPRegressor(activation='tanh', alpha=1, random_state=i_min)
rscv_MLP = RandomizedSearchCV(mlp, param_dist4, n_iter=10, n_jobs=-1, cv=5, random_state=i_min, scoring='neg_mean_squared_error')
random_MLP = rscv_MLP.fit(x_train_stand, y_train_stand)

print('Optimal tuning parameter values:\n', random_MLP.best_params_)
print('MSE of the optimal results:', abs(random_MLP.best_score_))

y_pred = random_MLP.predict(x_test_stand)

m1 = mean_squared_error(y_test_stand, y_pred)
m2 = mean_squared_error(y_test_stand, y_expert_stand)

print('MLP Model\'s MSE is', m1)
print('Expert Guess\'s MSE is', m2)


# In[63]:


perm = PermutationImportance(random_MLP, random_state=i_min).fit(x_test_stand, y_test_stand)
eli5.show_weights(perm, feature_names = x_test.columns.tolist(), top=50)


# ## Most undervalued Non-GK players by Expert

# In[24]:


x_whole = model_df[['Age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]


# In[25]:


y_expert_whole = model_df[['potential']]


# In[26]:


scaler = StandardScaler()  
scaler.fit(x_whole)  
x_mlp_stand = scaler.transform(x_whole)  

y_pred_mlp = random_MLP.predict(x_mlp_stand)

y_real = model_df[['Peak Overall']]
scaler = StandardScaler()  
scaler.fit(y_real)

mlp_predicted_potential = scaler.inverse_transform(y_pred_mlp)


# In[68]:


mlp_predicted_potential


# In[27]:


for i in range(3208):
    model_df.loc[i, 'gap_1'] = mlp_predicted_potential[i] - model_df.loc[i, 'Peak Overall']


# In[28]:


model_df['MLP Prediction'] = model_df['Peak Overall'] + model_df['gap_1']


# In[29]:


from scipy.stats import norm


# In[30]:


plt.figure(figsize=(12,8))
plt.hist(model_df['potential'], bins=20, normed=True, alpha=0.6, color='g')
plt.xlabel("Player\'s Potential Scores", fontsize = 16)
plt.ylabel('Number of players', fontsize = 16)
 
# Plot the probability density function for norm
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, model_df['potential'].mean(), model_df['potential'].std())
plt.plot(x, p, 'k', linewidth=2, color='r')
plt.title('Histogram of Experts\' guess of Non-goalkeepers\' Potential Scores', fontsize = 20)

plt.xticks(np.arange(50, 100, 5))

plt.show()


# In[33]:


plt.figure(figsize=(12,8))
plt.hist(model_df['Peak Overall'], bins=20, normed=True, alpha=0.6, color='purple')
plt.xlabel("Player\'s Peak Scores", fontsize = 16)
plt.ylabel('Number of players', fontsize = 16)
 
# Plot the probability density function for norm
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, model_df['Peak Overall'].mean(), model_df['Peak Overall'].std())
plt.plot(x, p, 'k', linewidth=2, color='b')
plt.title('Histogram of Non-goalkeepers\' Peak Overall Scores', fontsize = 20)

plt.xticks(np.arange(50, 100, 5))

plt.show()


# In[34]:


plt.figure(figsize=(12,8))
plt.hist(model_df['MLP Prediction'], bins=20, normed=True, alpha=0.6, color='b')
plt.xlabel("Player\'s MLP predicted Potential Scores", fontsize = 16)
plt.ylabel('Number of players', fontsize = 16)
 
# Plot the probability density function for norm
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, model_df['MLP Prediction'].mean(), model_df['MLP Prediction'].std())
plt.plot(x, p, 'k', linewidth=2, color='y')
plt.title('Histogram of MLP model\'s Prediction of Non-goalkeepers\' Potential Scores', fontsize = 20)

plt.xticks(np.arange(50, 100, 5))

plt.show()


# In[75]:


model_df['gap_2'] = model_df['potential'] - model_df['Peak Overall']


# In[80]:


model_df['gap_3'] = model_df['gap_1'] - model_df['gap_2']


# In[81]:


value_prediction_table = model_df.sort_values(by=['gap_3'], ascending=False).head(10)


# In[83]:


value_prediction_table.to_csv('underestimated_players.csv')


# ## Generate Ranking For Current Players

# In[91]:


df_2020 = pd.read_csv('players_20.csv')


# In[92]:


non_gk_2020 = df_2020[df_2020['player_positions'] != 'GK']


# In[93]:


x_2020 = non_gk_2020[['age', 'overall', 'height_cm', 'weight_kg',
   'international_reputation', 'weak_foot', 'skill_moves', 'pace',
   'shooting', 'passing', 'dribbling', 'defending', 'physic',
   'attacking_crossing', 'attacking_finishing',
   'attacking_heading_accuracy', 'attacking_short_passing',
   'attacking_volleys', 'skill_dribbling', 'skill_curve',
   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
   'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
   'movement_reactions', 'movement_balance', 'power_shot_power',
   'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
   'mentality_aggression', 'mentality_interceptions',
   'mentality_positioning', 'mentality_vision', 'mentality_penalties',
   'defending_marking', 'defending_standing_tackle',
   'defending_sliding_tackle']]


# In[94]:


y_2020 = non_gk_2020[['potential']]


# In[95]:


scaler = StandardScaler()  
scaler.fit(x_2020)  
x_2020_stand = scaler.transform(x_2020)  

y_2020_mlp = random_SVR.predict(x_2020_stand)

scaler = StandardScaler()  
scaler.fit(y_2020)

mlp_predicted_potential = scaler.inverse_transform(y_2020_mlp)


# In[96]:


non_gk_2020['MLP Potential'] = mlp_predicted_potential.round(2)


# In[97]:


non_gk_2020['Potential Gap'] =  non_gk_2020['MLP Potential'] - non_gk_2020['potential']


# In[100]:


non_gk_2020[non_gk_2020['age'] < 26].sort_values(by=['potential'], ascending=False).head(10).to_excel('nongk_expert_potential_rank.xlsx')


# In[102]:


non_gk_2020[non_gk_2020['age'] < 26].sort_values(by=['MLP Potential'], ascending=False).head(10).to_excel('nongk_mlp_potential_rank.xlsx')


# In[104]:


non_gk_2020[non_gk_2020['age'] < 26].sort_values(by=['overall'], ascending=False).head(10).to_excel('nongk_current_overall_rank.xlsx')


# In[106]:


non_gk_2020[non_gk_2020['age'] < 26].sort_values(by=['Potential Gap'], ascending=False).head(10).to_excel('nongk_undervalue_potential_rank.xlsx')


# In[ ]:




