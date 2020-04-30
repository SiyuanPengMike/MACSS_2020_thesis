#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[55]:


lin_table = pd.read_excel('non_gk_ols_table.xlsx')


# In[56]:


mlp_table = pd.read_excel('non_gk_mlp_table.xlsx')


# In[57]:


mlp = mlp_table.set_index('Column')


# In[58]:


lin = lin_table.set_index('Column')


# In[59]:


two = lin.merge(mlp, left_index = True, right_index = True)


# In[60]:


sgd_table = pd.read_excel('non_gk_SGD_table.xlsx')


# In[61]:


sgd = sgd_table.set_index('Column')


# In[62]:


three = two.merge(sgd, left_index = True, right_index = True)


# In[63]:


rf = pd.read_excel('non_gk_RF_table.xlsx')


# In[64]:


rf = rf.set_index('Column')


# In[65]:


four = three.merge(rf, left_index = True, right_index = True)


# In[66]:


svm = pd.read_excel('non_gk_svm_table.xlsx')


# In[67]:


svm = svm.set_index('Column')


# In[68]:


result = four.merge(svm, left_index = True, right_index = True)


# In[69]:


results = result.reset_index()


# In[70]:


results.rename(columns={'Column':'var'}, inplace=True)


# In[71]:


results = results[['var', 'MLP', 'OLS', 'SGD', 'SVM', 'RF']]


# In[72]:


results


# In[30]:


import matplotlib.pyplot as plt


# In[75]:


import seaborn as sns
sns.set(style="whitegrid")


# Make the PairGrid
g = sns.PairGrid(results.sort_values('MLP', ascending=False),
                 x_vars=results.columns[1:], y_vars=['var'],
                 height=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
      palette="ch:s=1,r=-.1,h=1_r", linewidth=1, edgecolor="w")

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(-0.01, 0.1), xlabel="Weights", ylabel="")

# Use semantically meaningful titles for the columns
titles = ['MLP', 'OLS','SGD', 'SVM', 'RandomeForest']
for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
sns.despine(left=True, bottom=True)


# In[ ]:





# In[33]:


lin_table = pd.read_excel('gk_ols_table.xlsx')


# In[34]:


mlp_table = pd.read_excel('gk_mlp_table.xlsx')


# In[35]:


mlp = mlp_table.set_index('Column')


# In[36]:


lin = lin_table.set_index('Column')


# In[37]:


two = lin.merge(mlp, left_index = True, right_index = True)


# In[38]:


sgd_table = pd.read_excel('gk_SGD_table.xlsx')


# In[39]:


sgd = sgd_table.set_index('Column')


# In[40]:


three = two.merge(sgd, left_index = True, right_index = True)


# In[41]:


rf = pd.read_excel('gk_RF_table.xlsx')


# In[42]:


rf = rf.set_index('Column')


# In[43]:


four = three.merge(rf, left_index = True, right_index = True)


# In[44]:


svm = pd.read_excel('gk_svm_table.xlsx')


# In[45]:


svm = svm.set_index('Column')


# In[46]:


result = four.merge(svm, left_index = True, right_index = True)


# In[47]:


results = result.reset_index()


# In[48]:


results.rename(columns={'Column':'var'}, inplace=True)


# In[49]:


results = results[['var', 'MLP', 'OLS', 'SGD', 'SVM', 'RF']]


# In[50]:


results


# In[54]:


import seaborn as sns
sns.set(style="whitegrid")


# Make the PairGrid
g = sns.PairGrid(results.sort_values('MLP', ascending=False),
                 x_vars=results.columns[1:], y_vars=['var'],
                 height=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
      palette="ch:s=1,r=-.1,h=1_r", linewidth=1, edgecolor="w")

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(-0.02, 0.1), xlabel="Weights", ylabel="")

# Use semantically meaningful titles for the columns
titles = ['MLP', 'OLS','SGD', 'SVM', 'RandomeForest']
for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
sns.despine(left=True, bottom=True)


# In[ ]:




