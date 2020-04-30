#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('final_can_raw.csv')


# In[3]:


# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-deep')


# In[17]:


plt.figure(figsize = (13, 8))
ax = sns.countplot(x = 'height_cm', data = df, color = 'deepskyblue')
ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)
ax.set_xlabel(xlabel = 'Height in Centimeter', fontsize = 20)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 20)
plt.xticks(rotation=75)
plt.show()


# In[18]:


# To show Different body weight of the players participating in the FIFA 2019

plt.figure(figsize = (13, 8))
sns.countplot(x = 'weight_kg', data = df, color = 'deepskyblue')
plt.title('Count of players on Basis of Weight', fontsize = 20)
plt.xlabel('Weight in Kilogram', fontsize = 20)
plt.ylabel('Count of Players', fontsize = 20)
plt.xticks(rotation=75)
plt.show()


# In[19]:


df_des = pd.read_csv('clean_whole_candidate.csv')


# In[21]:


df_des['preferred_foot'] = df['preferred_foot']


# In[31]:


# plotting a clean correlation heatmap

plt.rcParams['figure.figsize'] = (30, 20)
sns.set(font_scale=1.8)
sns.heatmap(df_des[['Age', 'Peak Overall', 'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'defending_marking',
       'defending_sliding_tackle', 'defending_standing_tackle', 
       'goalkeeping_diving', 'goalkeeping_handling',
       'goalkeeping_kicking', 'goalkeeping_positioning',
       'goalkeeping_reflexes', 'height_cm', 'international_reputation',
       'mentality_aggression', 'mentality_interceptions',
       'mentality_penalties', 'mentality_positioning', 'mentality_vision',
       'movement_acceleration', 'movement_agility', 'movement_balance',
       'movement_reactions', 'movement_sprint_speed', 'overall', 
       'power_jumping', 'power_long_shots',
       'power_shot_power', 'power_stamina', 'power_strength', 
       'skill_ball_control', 'skill_curve', 'skill_dribbling',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_moves', 'weak_foot',
       'weight_kg', 'preferred_foot']].corr(), annot = True, annot_kws={"size": 12})

plt.title('Correlation Heatmap of the Dataset', fontsize = 30)
plt.show()
plt.savefig('heatmap.png')


# In[ ]:




