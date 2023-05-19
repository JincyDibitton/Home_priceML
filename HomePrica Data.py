#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import seaborn as sns


# In[4]:


df=pd.read_csv('House_Price.csv')


# In[5]:


df


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


sns.boxplot(y='n_hos_beds',data=df)


# In[9]:


sns.boxplot(y='n_hot_rooms',data=df)


# In[10]:


sns.jointplot(x='rainfall',y='Sold',data=df)


# In[11]:


sns.countplot(x="airport",data=df)


# In[12]:


sns.countplot(x="waterbody",data=df)


# In[13]:


sns.countplot(x="bus_ter",data=df)


# Observations
# 1.Missing values in n_hos_beds
# 2.bus_ter provides no additional information.
# 3.n_hot_rooms and rainfall have outliers.

# In[14]:


np.percentile(df.n_hot_rooms,[99][0])


# In[15]:


uv=np.percentile(df.n_hot_rooms,[99][0])


# In[16]:


df.n_hot_rooms[(df.n_hot_rooms)>3*uv]


# In[17]:


df[df.n_hot_rooms>uv]


# In[18]:


np.percentile(df.rainfall,[1][0])


# In[19]:


lv=np.percentile(df.rainfall,[1][0])


# In[20]:


df[df.rainfall<lv]


# In[21]:


df.rainfall[(df.rainfall<0.3*lv)]


# In[22]:


df.rainfall[(df.rainfall<0.3*lv)]=0.3*lv


# In[23]:


df.describe()


# In[24]:


df.info()


# In[25]:


df.n_hos_beds=df.n_hos_beds.fillna(df.n_hos_beds.mean())


# In[26]:


df.info()


# In[27]:


df['avg_dist']=(df.dist1+df.dist2+df.dist3+df.dist4)/4


# In[28]:


df.describe()


# In[29]:


del df['dist1']


# In[30]:


del df['dist2']
del df['dist3']
del df['dist4']


# In[31]:


df.describe()


# In[32]:


df=pd.get_dummies(df)


# In[33]:


df.head()


# In[34]:


del df['airport_NO']
del df['waterbody_None']


# In[35]:


df.head()


# In[36]:


# to find whether house sold in 3 months??
# need two variables x and y x-independent(price) ,y-dependent(sold)[single predictor]


# In[37]:


x=df[['price']]
y=df['Sold']


# In[38]:


x.head()


# In[39]:


y.head()


# In[40]:


# predict with models
# we use sklearn model includes 3 steps
# 1.create the classification object
# 2.with the ojects fit x and y
# 3.then predict with the classification


# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


clfs_lrs=LogisticRegression()


# In[43]:


clfs_lrs.fit(x,y)


# In[44]:


clfs_lrs.coef_


# In[45]:


clfs_lrs.intercept_


# In[46]:


import statsmodels.api as sn


# In[47]:


x_const=sn.add_constant(x)


# In[48]:


x_const.head()


# In[49]:


import statsmodels.discrete.discrete_model as sm


# In[50]:


logit=sm.Logit(y,x_const).fit()


# In[51]:


logit.summary()


# In[52]:


# multiple predictor


# In[53]:


x=df.loc[:,df.columns!='Sold']


# In[54]:


y=df['Sold']


# In[55]:


x.head()


# In[56]:


y.head()


# In[57]:


from sklearn.linear_model import LogisticRegression


# In[58]:


clfs_lrs=LogisticRegression()


# In[59]:


clfs_lrs.fit(x,y)


# In[60]:


clfs_lrs.coef_


# In[61]:


clfs_lrs.intercept_


# In[62]:


import statsmodels.api as sn


# In[63]:


x_const=sn.add_constant(x)


# In[64]:


x_const.head()


# In[65]:


import statsmodels.discrete.discrete_model


# In[66]:


logit=sm.Logit(y,x_const).fit()


# In[67]:


logit.summary()


# In[68]:


# Prediction and Confusion Maatrix


# In[69]:


clfs_lrs.predict_proba(x)


# In[70]:


y_pred=clfs_lrs.predict(x)


# In[71]:


y_pred


# In[72]:


# to find 0.3 prediction


# In[73]:


y_pred_03=(clfs_lrs.predict_proba(x)[:1]>=0.3).astype(bool)


# In[74]:


from sklearn.metrics import confusion_matrix


# In[75]:


confusion_matrix(y,y_pred)


# In[76]:


# LDA


# In[77]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[78]:


clf_lda=LinearDiscriminantAnalysis()


# In[79]:


clf_lda.fit(x,y)


# In[80]:


y_pred_lda=clf_lda.predict(x)


# In[81]:


y_pred_lda


# In[82]:


confusion_matrix(y,y_pred_lda)


# In[83]:


# training and test


# In[84]:


from sklearn .model_selection import train_test_split


# In[85]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[86]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[87]:


# creating model 


# In[88]:


clf_LR=LogisticRegression()


# In[89]:


clf_LR.fit(x_train,y_train)


# In[90]:


y_test_pred=clf_LR.predict(x_test)


# In[91]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[92]:


confusion_matrix(y_test,y_test_pred)


# In[93]:


accuracy_score(y_test,y_test_pred)


# In[95]:


from sklearn import preprocessing 


# In[115]:


scaler=preprocessing.StandardScaler().fit(x_train)
x_train_s=scaler.transform(x_train)


# In[116]:


scaler=preprocessing.StandardScaler().fit(x_test)
x_test_s=scaler.transform(x_test)


# In[117]:


x_test_s


# In[118]:


from sklearn.neighbors import KNeighborsClassifier


# In[119]:


clf_knn_1=KNeighborsClassifier(n_neighbors=1)


# In[120]:


clf_knn_1.fit(x_train_s,y_train)


# In[121]:


confusion_matrix(y_test,clf_knn_1.predict(x_test_s))


# In[122]:


accuracy_score(y_test,clf_knn_1.predict(x_test_s))


# In[123]:


# if we have 10 neighbors


# In[124]:


from sklearn.model_selection import GridSearchCV


# In[125]:


params={'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}


# In[130]:


grid_search_cv=GridSearchCV(KNeighborsClassifier(),params)


# In[131]:


grid_search_cv.fit(x_train_s,y_train)


# In[132]:


grid_search_cv.best_params_


# In[133]:


optimised_KNN=grid_search_cv.best_estimator_


# In[134]:


confusion_matrix(y_test,y_test_pred)


# In[135]:


accuracy_score(y_test,y_test_pred)


# In[ ]:




