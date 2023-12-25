#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing required packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn import svm
from sklearn.neural_network import MLPClassifier
# from sklearn.Linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


wine = pd.read_csv('D:\\winequality-red.csv')


# In[3]:


wine.head()


# In[4]:


wine.info()


# In[5]:


wine.isnull().sum()


# In[6]:


# preprocessing data
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
wine['quality'].unique()


# In[7]:


label_quality = LabelEncoder()


# In[8]:


wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[9]:


wine.head(10)


# In[10]:


wine['quality'].value_counts()


# In[11]:


sns.countplot(wine['quality'])


# In[12]:


# separating dataset as response variable & feature variables
X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[13]:


# train and test splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[14]:


#applying standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# random forest classifier

# In[15]:


rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[16]:


#how model performed
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


# SVM classifier

# In[17]:


clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)


# In[18]:


print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))


# neural network

# In[27]:


mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=1000)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)


# In[28]:


print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))


# In[29]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred_rfc)
print(cm)


# In[30]:


wine.head(10)


# In[31]:


Xnew = [[7.3, 0.58, 0.00, 2.0, 0.065, 15.0, 21.0, 0.9946, 3.36, 0.47, 10.0]]
Xnew = sc.transform(Xnew)
ynew = rfc.predict(Xnew)
ynew


# The quality of wine with given parameters is 0, or bad
