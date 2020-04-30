#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


non_gk_MSE = [0.4180, 0.2602, 0.2675, 0.2881, 0.2667, 0.2573]
gk_MSE = [0.3244, 0.2936, 0.2753, 0.2955, 0.3080, 0.2692]
Name = ['Expert Prediciton', 'Linear Regression', 'SGD Regression', 'Random Forest Regressor', 'Support Vector Regression', 'BP MLP']


# In[5]:


plt.figure(figsize = (20, 10))
plt.bar(Name, non_gk_MSE)
plt.title('MSE of Non-Goalkeeper Models', fontsize = 20)
plt.xlabel('Model Names', fontsize = 16)
plt.ylabel('MSE Value', fontsize = 16)
for i in range(len(non_gk_MSE)):
    plt.text(x=i-0.15, y=non_gk_MSE[i] + 0.01, s=str(non_gk_MSE[i]), size = 15, color='r')
#plt.xticks(MSE)
plt.show()


# In[8]:


plt.figure(figsize = (20, 10))
plt.bar(Name, gk_MSE)
plt.title('MSE of Goalkeeper Models', fontsize = 20)
plt.xlabel('Model Names', fontsize = 16)
plt.ylabel('MSE Value', fontsize = 16)
for i in range(len(gk_MSE)):
    plt.text(x=i-0.15, y=gk_MSE[i] + 0.01, s=str(gk_MSE[i]), size = 15, color='r')
plt.show()


# In[ ]:




