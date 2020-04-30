#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data_2020 = pd.read_csv('players_20.csv')


# In[3]:


data_19 = pd.read_csv('players_19.csv')


# In[4]:


data_18 = pd.read_csv('players_18.csv')


# In[5]:


data_17 = pd.read_csv('players_17.csv')


# In[6]:


data_16 = pd.read_csv('players_16.csv')


# In[7]:


data_15 = pd.read_csv('players_15.csv')


# In[8]:


frames = [data_2020, data_19, data_18, data_17, data_16, data_15]


# In[9]:


df_whole = pd.concat(frames)


# In[11]:


import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


ax = sns.lineplot(x="age", y="overall", data=df_whole)
ax.set_title(label = 'Overall Curve through Sports Career', fontsize = 20)
ax.set_xlabel(xlabel = 'Age', fontsize = 16)
ax.set_ylabel(ylabel = 'Overall', fontsize = 16)
plt.show()


# In[19]:


df_qual = df_whole[df_whole['age'] < 41]


# In[20]:


df_qual = df_qual[df_qual['age'] > 17]


# In[21]:


ax = sns.lineplot(x="age", y="overall", data=df_qual)
ax.set_title(label = 'Overall Curve through Sports Career', fontsize = 20)
ax.set_xlabel(xlabel = 'Age', fontsize = 16)
ax.set_ylabel(ylabel = 'Overall', fontsize = 16)
plt.show()


# In[22]:


id_set = set()


# In[23]:


id_dict = dict()


# In[24]:


for i, (fifa_id, age) in data_2020[['sofifa_id', 'age']].iterrows():
    id_set.add(fifa_id)


# In[25]:


for i in id_set:
    id_dict[i] = 1


# In[26]:


for i, (fifa_id, age) in data_19[['sofifa_id', 'age']].iterrows():
    if fifa_id in id_dict:
        id_dict[fifa_id] += 1


# In[27]:


for i, (fifa_id, age) in data_18[['sofifa_id', 'age']].iterrows():
    if fifa_id in id_dict:
        id_dict[fifa_id] += 1


# In[28]:


for i, (fifa_id, age) in data_17[['sofifa_id', 'age']].iterrows():
    if fifa_id in id_dict:
        id_dict[fifa_id] += 1


# In[29]:


for i, (fifa_id, age) in data_16[['sofifa_id', 'age']].iterrows():
    if fifa_id in id_dict:
        id_dict[fifa_id] += 1


# In[30]:


for i, (fifa_id, age) in data_15[['sofifa_id', 'age']].iterrows():
    if fifa_id in id_dict:
        id_dict[fifa_id] += 1


# In[31]:


final_id = set()


# In[32]:


for key,value in id_dict.items():
    if value == 6:
        final_id.add(key)


# In[33]:


len(final_id)


# In[34]:


df_new_qual = df_whole.loc[df_whole['sofifa_id'].isin(final_id)]


# In[35]:


ax = sns.lineplot(x="age", y="overall", data=df_new_qual)
ax.set_title(label = 'Overall Curve through Sports Career', fontsize = 20)
ax.set_xlabel(xlabel = 'Age', fontsize = 16)
ax.set_ylabel(ylabel = 'Overall', fontsize = 16)
plt.show()


# In[37]:


df_new_qual = df_new_qual[df_new_qual['age'] < 37]


# In[43]:


ax = sns.lineplot(x="age", y="overall", data=df_new_qual)
ax.set_title(label = 'Overall Curve through Sports Career', fontsize = 20)
ax.set_xlabel(xlabel = 'Age', fontsize = 16)
ax.set_ylabel(ylabel = 'Overall', fontsize = 16)
plt.scatter([25.5], [71.4], marker='o',color = 'y')
plt.scatter([28.75], [72.5], marker='o',color = 'r')
plt.show()


# In[ ]:




