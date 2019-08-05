#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


# In[13]:


SFNT = pd.read_csv('F:/flight_delay_new/SFSNT.csv')


# In[3]:


SFNT.head(5)


# In[4]:


df =pd.read_csv("F:/flight_delay_new/AFSNT.csv",encoding = "ISO-8859-1")


# In[6]:


df.head(5)


# In[7]:


df.tail(5)


# In[8]:


df.shape


# In[9]:


df['SDT_YY'].unique()


# In[26]:


df['DLY'].unique()


# In[27]:


df['AOD'].unique()


# In[10]:


df['ARP'].unique()


# In[11]:


df['FLO'].unique()


# In[30]:


df['ODP'].unique()


# In[31]:


df['SDT_DY'].unique()


# In[10]:


df.columns


# In[117]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[12]:


df.head(5)


# In[121]:


X = df.drop(['SDT_DY','DLY'], axis=1)


# In[122]:


def convert_delay(s):
    if s.DLY == "Y":
        return 1
    else:
        return 0
y = df.apply(convert_delay, axis=1)


# In[123]:


y.head(5)


# In[124]:


#Label distribution
label_value_count = y.value_counts()
sns.barplot(x=label_value_count.index, y=label_value_count.values)


# ## Training the model

# In[129]:


import seaborn as sns
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[130]:


X.dtypes


# In[131]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[132]:


X.dtypes


# ### label encode the categorical values and convert them to numbers 

# In[134]:


X_test.shape


# In[135]:


X_train.shape


# In[136]:


X.shape


# In[137]:


y.shape


# In[138]:


y_train.shape


# In[139]:


X_train.head(5)


# In[140]:


y.head(5)


# In[145]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(X_train['ARP'].astype(str))
le.transform(X_train['ARP'].astype(str))

le.fit(X_train['ODP'].astype(str))
le.transform(X_train['ODP'].astype(str))

le.fit(X_train['FLO'].astype(str))
le.transform(X_train['FLO'].astype(str))


# # A more definitive form

# In[149]:


train_data =pd.read_csv("F:/flight_delay_new/AFSNT.csv",encoding = "ISO-8859-1")


# In[148]:


test_data= pd.read_csv("F:/flight_delay_new/AFSNT_DLY.csv",encoding = "ISO-8859-1")


# In[ ]:





# In[150]:


train_data.shape


# In[151]:


test_data.shape


# In[152]:


test_data.head(5)


# In[201]:


train_data.head(5)


# In[208]:


train_data= train_data.drop(['CNL'], axis=1)


# In[209]:


#label encoding for training data

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train_data ["ARP"] = labelencoder.fit_transform(train_data["ARP"])
train_data ["SDT_DY"] = labelencoder.fit_transform(train_data["SDT_DY"])
train_data ["ODP"] = labelencoder.fit_transform(train_data["ODP"])
train_data ["FLO"] = labelencoder.fit_transform(train_data["FLO"])
train_data ["FLT"] = labelencoder.fit_transform(train_data["FLT"])
#train_data ["REG"] = labelencoder.fit_transform(train_data["REG"])
train_data ["AOD"] = labelencoder.fit_transform(train_data["AOD"])
train_data ["IRR"] = labelencoder.fit_transform(train_data["IRR"])
train_data ["DLY"] = labelencoder.fit_transform(train_data["DLY"])


# In[211]:


train_data


# In[279]:


test_data.head(5)


# In[213]:


test_data.columns


# In[ ]:


#test_data= train_data.drop(['SDT_DY'], axis=1)


# In[215]:


#label encoding for test_data

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
test_data ["ARP"] = labelencoder.fit_transform(test_data["ARP"])
test_data ["SDT_DY"] = labelencoder.fit_transform(test_data["SDT_DY"])
test_data ["ODP"] = labelencoder.fit_transform(test_data["ODP"])
test_data ["FLO"] = labelencoder.fit_transform(test_data["FLO"])
test_data ["FLT"] = labelencoder.fit_transform(test_data["FLT"])
#train_data ["REG"] = labelencoder.fit_transform(train_data["REG"])
test_data ["AOD"] = labelencoder.fit_transform(test_data["AOD"])
#test_data ["DLY"] = labelencoder.fit_transform(train_data["DLY"])


# In[240]:


train_data['STT'] = pd.to_datetime(train_data['STT'] )
train_data['ATT'] = pd.to_datetime(train_data['ATT'])

test_data['STT'] = pd.to_datetime(train_data['STT'])


# In[222]:


train_data.dtypes


# In[233]:


def time_delay(s):
    s_hour = s['STT'].hour
    s_min = s['STT'].minute
    a_hour = s['ATT'].hour
    a_min = s['ATT'].minute
    
    time_dif = (a_hour - s_hour) * 60 + a_min - s_min
    return time_dif

train_data['TIME_DIF'] = train_data.apply(time_delay, axis=1)


# In[234]:


train_data.head()


# In[255]:


#train_data.CNR.fillna()


# In[244]:


train_data.CNR.value_counts()


# In[245]:


train_data


# In[250]:


train_data.isnull().sum()


# In[253]:


X.isnull().sum()


# In[254]:


y.isnull().sum()


# ## CLASSIFIER

# In[292]:


X= train_data.drop(['REG','STT','ATT','DLY','TIME_DIF'], axis=1)
y=train_data['DLY']


# In[293]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[294]:


clf = RandomForestClassifier(n_jobs=4, random_state=0)
clf.fit(X, y)


# In[296]:


y_pred = clf.predict(X_test)
print("Baseline accuracy:", accuracy_score(y_test, y_pred))


# In[297]:


feature_list = train_data.columns
importances = list(clf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:40} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[299]:


from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# overall accuracy
acc = clf.score(X_test,y_test)

# get roc/auc info
Y_score = clf.predict_proba(X_test)[:,1]
fpr = dict()
tpr = dict()
fpr, tpr, _ = roc_curve(y_test, Y_score)

roc_auc = dict()
roc_auc = auc(fpr, tpr)

# make the plot
plt.figure(figsize=(10,10))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.plot(fpr, tpr, label='AUC = {0}'.format(roc_auc))        
plt.legend(loc="lower right", shadow=True, fancybox =True) 
plt.show()


# In[ ]:




