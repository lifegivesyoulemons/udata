
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as sts
import pandas as pd


# In[2]:


#1
def nandrop(data, axis=0):
    header = np.array(data.columns)
    data = np.array(data)
    drop_index = []
    if axis==0:
        #if nan
        for i in range(np.size(data,0)):
            for j in range(np.size(data,1)):
                if data[i][j]!=data[i][j]:
                    drop_index.append(i)
        data = np.delete(data, drop_index, 0)
        
    else:
        #if nan
        for i in range(np.size(data,0)):
            for j in range(np.size(data,1)):
                if data[i][j]!=data[i][j]:
                    drop_index.append(j)
        data = np.delete(data, drop_index, 1)
        header = np.delete(header, drop_index, 0)
    return pd.DataFrame(data, columns=header)


# In[3]:


#2
#mean
def nantomean(data, name_of_numeric_cols):
    header = name_of_numeric_cols
    data = np.array(data[name_of_numeric_cols])
    for i in range(np.size(data,0)):
            for j in range(np.size(data,1)):
                if data[i,j] != data[i,j]:
                    data[i,j] = np.nanmean(data[:,j])
    return pd.DataFrame(data, columns=header)

#median
def nantomed(data, name_of_numeric_cols):
    header = name_of_numeric_cols
    data = np.array(data[name_of_numeric_cols])
    for i in range(np.size(data,0)):
            for j in range(np.size(data,1)):
                if data[i,j] != data[i,j]:
                    data[i,j] = np.nanmedian(data[:,j])
    return pd.DataFrame(data, columns=header)

#mode
def nantomode(data, name_of_numeric_cols):
    header = name_of_numeric_cols
    data = np.array(data[name_of_numeric_cols])
    mode = sts.mode(data)[0][0]
    for i in range(np.size(data,0)):
            for j in range(np.size(data,1)):
                if data[i,j] != data[i,j]:
                    data[i,j] = mode[j]
    return pd.DataFrame(data, columns=header)


# In[4]:


#3
def nantolin(data, target):
    lr = LinearRegression()
    target_name = target.columns[0]
    data_drop = nandrop(pd.concat([data, target], axis = 1))
    X_train, y_train = data_drop[list(data_drop.columns)[:-1]], data_drop[list(data_drop.columns)[-1]]
    lr.fit(X_train, y_train)
    data = nantomean(data)
    data_copy = pd.concat([data, target], axis = 1)
    data_copy[target_name] = data_copy[target_name].fillna('?')
    for i in range(data_copy.shape[0]):
        if data_copy.ix[i, target_name] == '?':
            x_test = pd.DataFrame(data_copy.ix[i, :-1]).T
            data_copy.ix[i, target_name] = lr.predict(x_test)[0]
    return pd.to_numeric(data_copy[target_name])


# In[5]:


#4
def distance_matrix(data):
    m = data.shape[1]
    n = data.shape[0]  
    is_numeric = [all(isinstance(n, numbers.Number) for n in data.iloc[:, i]) for i, x in enumerate(data)]
    is_all_numeric = sum(is_numeric) == len(is_numeric)
    is_all_categorical = sum(is_numeric) == 0
    is_mixed_type = not is_all_categorical and not is_all_numeric   
    if is_mixed_type:
        number_of_numeric_var = sum(is_numeric)
        number_of_categorical_var = m - number_of_numeric_var
        data_numeric = data.iloc[:, is_numeric]
        data_numeric = (data_numeric - data_numeric.mean()) / (data_numeric.max() - data_numeric.min())
        data_categorical = data.iloc[:, [not x for x in is_numeric]]        
    if is_mixed_type:
        data_numeric = nantomean(data_numeric)
        for x in data_categorical:
            data_categorical[x] = nantomode(data_categorical[x])
    elif is_all_numeric:
        data = nantomean(data)
    else:
        for x in data:
            data[x] = nantomode(data[x])
    if not is_all_numeric :
        if is_mixed_type:
            data_categorical = pd.DataFrame([pd.factorize(data_categorical[x])[0] for x in data_categorical]).transpose()
        else:
            data = pd.DataFrame([pd.factorize(data[x])[0] for x in data]).transpose()
    if is_all_numeric:
        result_matrix = cdist(data, data, metric='euclidean')
    elif is_all_categorical:
        result_matrix = cdist(data, data, metric='hamming')
    else:
        result_numeric = cdist(data_numeric, data_numeric, metric='euclidean')
        result_categorical = cdist(data_categorical, data_categorical, metric='hamming')
        result_matrix = np.array([[1.0*(result_numeric[i, j] * number_of_numeric_var + result_categorical[i, j] * number_of_categorical_var) / m for j in range(n)] for i in range(n)])
        np.fill_diagonal(result_matrix, np.nan)
    return pd.DataFrame(result_matrix)

def knn_impute(target, attributes, k_neighbors):
    n = len(target)
    is_target_numeric = all(isinstance(n, numbers.Number) for n in target)
    target = pd.DataFrame(target)
    attributes = pd.DataFrame(attributes)
    distances = distance_matrix(attributes)
    for i, value in enumerate(target.iloc[:, 0]):
        if pd.isnull(value):
            order = distances.iloc[i,:].values.argsort()[:k_neighbors]
            closest_to_target = target.iloc[order, :]
            missing_neighbors = [x for x  in closest_to_target.isnull().iloc[:, 0]]
            target.iloc[i] = stats.mode(closest_to_target.dropna())[0][0]
    return target


# In[8]:


#5
def standardize(data, name_of_numeric_cols):
    header = name_of_numeric_cols
    data = np.array(data[name_of_numeric_cols])
    data_copy = data
    for i in range(np.size(data,0)):
            for j in range(np.size(data,1)):
                data_copy[i][j] = (data[i][j] - np.nanmean(data[:,j]))/np.nanstd(data[:,j])
    return pd.DataFrame(data_copy, columns=header)


# In[9]:


#6
def scale(data, name_of_numeric_cols):
    header = name_of_numeric_cols
    data = np.array(data[name_of_numeric_cols])
    data_copy = data
    for i in range(np.size(data,0)):
            for j in range(np.size(data,1)):
                data_copy[i][j] = (data[i][j] - np.nanmin(data[:,j]))/(np.nanmax(data[:,j]) - np.nanmin(data[:,j]))
    return pd.DataFrame(data_copy, columns=header)

