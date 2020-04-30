#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data_2020 = pd.read_csv('players_20.csv')


# In[3]:


data_2020.head()


# In[4]:


data_19 = pd.read_csv('players_19.csv')


# In[5]:


data_19.head()


# In[6]:


data_18 = pd.read_csv('players_18.csv')


# In[7]:


data_17 = pd.read_csv('players_17.csv')


# In[8]:


data_16 = pd.read_csv('players_16.csv')


# In[9]:


data_15 = pd.read_csv('players_15.csv')


# In[10]:


data_2020.dtypes


# In[11]:


id_set = set()


# In[12]:


id_dict = dict()


# In[15]:


for i, (fifa_id, age) in data_2020[['sofifa_id', 'age']].iterrows():
    id_set.add(fifa_id)


# In[17]:


for i in id_set:
    id_dict[i] = 1


# In[19]:


for i, (fifa_id, age) in data_19[['sofifa_id', 'age']].iterrows():
    if fifa_id in id_dict:
        id_dict[fifa_id] += 1


# In[21]:


for i, (fifa_id, age) in data_18[['sofifa_id', 'age']].iterrows():
    if fifa_id in id_dict:
        id_dict[fifa_id] += 1


# In[22]:


for i, (fifa_id, age) in data_17[['sofifa_id', 'age']].iterrows():
    if fifa_id in id_dict:
        id_dict[fifa_id] += 1


# In[23]:


for i, (fifa_id, age) in data_16[['sofifa_id', 'age']].iterrows():
    if fifa_id in id_dict:
        id_dict[fifa_id] += 1


# In[24]:


for i, (fifa_id, age) in data_15[['sofifa_id', 'age']].iterrows():
    if fifa_id in id_dict:
        id_dict[fifa_id] += 1


# In[26]:


final_id = set()


# In[28]:


for key,value in id_dict.items():
    if value == 6:
        final_id.add(key)


# In[30]:


len(final_id)


# In[31]:


pd.set_option('display.max_columns', None)


# In[32]:


data_15.head()


# In[34]:


list(data_15.columns)


# In[36]:


data_15.head(1)


# In[43]:


data_to_use_columns = ['sofifa_id','data_Year',
                       'Peak Overall', 'Peak Year', 'Peak Age','2020 Age', 'Age','overall', 'Expert Potential', 
                       'short_name', 'long_name', 'age', 'dob', 'height_cm', 'weight_kg','value_eur','wage_eur',
                       'player_positions','preferred_foot','international_reputation','weak_foot','skill_moves',
                       'work_rate','body_type','release_clause_eur','player_tags','team_position','team_jersey_number',
                       'loaned_from','joined','contract_valid_until','nation_position','nation_jersey_number',
                       'pace','shooting','passing', 'dribbling', 'defending', 'physic', 'gk_diving', 'gk_handling',
                       'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning', 'player_traits', 'attacking_crossing',
                       'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
                       'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
                       'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
                       'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
                       'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
                       'mentality_vision', 'mentality_penalties', 'defending_marking',
                       'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling',
                       'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']
data_to_use = pd.DataFrame(index = range(0, len(final_id)), columns = data_to_use_columns)


# In[44]:


data_to_use['sofifa_id'] = final_id


# In[45]:


data_to_use.head()


# In[46]:


for player_id in data_2020['sofifa_id']:
    data_to_use.loc[data_to_use['sofifa_id'] == player_id, '2020 Age'] = int(data_2020.loc[data_2020['sofifa_id'] == player_id, 'age'])


# In[48]:


data_to_use['Peak Overall'] = 0


# In[50]:


data_2020['Year'] = 2020


# In[52]:


data_19['Year'] = 2019


# In[53]:


data_18['Year'] = 2018
data_17['Year'] = 2017
data_16['Year'] = 2016
data_15['Year'] = 2015


# In[56]:


for player_id in data_to_use['sofifa_id']:
    for data_year in (data_2020, data_19, data_18, data_17, data_16, data_15):
        if data_year.loc[data_year['sofifa_id'] == player_id]['overall'].values.size > 0:
            current_overall = data_year.loc[data_year['sofifa_id'] == player_id]['overall'].values[0]
            peak_overall = data_to_use.loc[data_to_use['sofifa_id'] == player_id, 'Peak Overall'].values[0]
            if current_overall > peak_overall:
                data_to_use.loc[data_to_use['sofifa_id'] == player_id, 'Peak Overall'] = current_overall
                data_to_use.loc[data_to_use['sofifa_id'] == player_id, 'Peak Year'] = int(data_year.Year.mean())
        else:
            continue


# In[57]:


data_to_use.head()


# In[58]:


data_to_use['Peak Age'] = data_to_use.apply(lambda x: x['2020 Age'] - (2020 - x['Peak Year']), axis=1)


# In[59]:


data_to_use.head()


# In[60]:


data_to_use['Age'] = data_to_use.apply(lambda x: x['Peak Age'] - 3, axis=1)


# In[61]:


data_to_use['data_Year'] = data_to_use.apply(lambda x: x['Peak Year'] - 3, axis=1)


# In[62]:


data_to_use.head()


# In[63]:


data_to_use.rename(columns={'Expert Potential': 'potential'}, inplace=True)


# In[64]:


data_to_use.head()


# In[65]:


columns_to_fill = ['overall', 'potential', 
                       'short_name', 'long_name', 'age', 'dob', 'height_cm', 'weight_kg','value_eur','wage_eur',
                       'player_positions','preferred_foot','international_reputation','weak_foot','skill_moves',
                       'work_rate','body_type','release_clause_eur','player_tags','team_position','team_jersey_number',
                       'loaned_from','joined','contract_valid_until','nation_position','nation_jersey_number',
                       'pace','shooting','passing', 'dribbling', 'defending', 'physic', 'gk_diving', 'gk_handling',
                       'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning', 'player_traits', 'attacking_crossing',
                       'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
                       'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
                       'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
                       'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
                       'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
                       'mentality_vision', 'mentality_penalties', 'defending_marking',
                       'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling',
                       'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']


# In[67]:


final_candidate = data_to_use[data_to_use['data_Year'] > 2014]


# In[69]:


for player_id in final_candidate['sofifa_id']:
    year = final_candidate.loc[final_candidate['sofifa_id'] == player_id, 'data_Year'].values[0]
    if year == 2017:
        for col in columns_to_fill:
            if data_17.loc[data_17['sofifa_id'] == player_id, col].values.size > 0:
                final_candidate.loc[final_candidate['sofifa_id'] == player_id, col] = data_17.loc[data_17['sofifa_id'] == player_id, col].values[0]
    elif year == 2016:
        for col in columns_to_fill:
            if data_16.loc[data_16['sofifa_id'] == player_id, col].values.size > 0:
                final_candidate.loc[final_candidate['sofifa_id'] == player_id, col] = data_16.loc[data_16['sofifa_id'] == player_id, col].values[0]
    elif year == 2015:
        for col in columns_to_fill:
            if data_15.loc[data_15['sofifa_id'] == player_id, col].values.size > 0:
                final_candidate.loc[final_candidate['sofifa_id'] == player_id, col] = data_15.loc[data_15['sofifa_id'] == player_id, col].values[0]


# In[70]:


final_candidate


# In[71]:


final_candidate.to_csv('final_can_raw.csv', encoding='utf-8-sig')


# In[72]:


gk_data = final_candidate[final_candidate['player_positions'] == 'GK']


# In[73]:


gk_data


# In[74]:


gk_data.to_csv('gk_can_raw.csv', encoding='utf-8-sig')


# In[75]:


non_gk_data = final_candidate[final_candidate['player_positions'] != 'GK']


# In[76]:


non_gk_data


# In[77]:


non_gk_data.to_csv('non_gk_can_raw.csv', encoding='utf-8-sig')


# In[ ]:




