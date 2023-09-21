#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats as stats

import warnings 
warnings. filterwarnings('ignore')


# # Q7

# In[2]:


df = pd.read_csv("Q7.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
sns.histplot(x="Score", data=df)

plt.subplot(1,3,2)
sns.histplot(x="Points", data=df)

plt.subplot(1,3,3)
sns.histplot(x="Weigh", data=df)


# In[7]:


plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
sns.distplot(df["Score"])

plt.subplot(1,3,2)
sns.distplot(df["Points"])

plt.subplot(1,3,3)
sns.distplot(df["Weigh"])


# #### distribution of data shows that score and weight contains outsiders while point give double peak curve

# # Q8

# In[8]:


patient_weights = pd.Series([108, 110, 123, 134, 135, 145, 167, 187, 199])


# In[9]:


patient_weights.mean()


# # Q9

# In[10]:


df1 = pd.read_csv("Q9_a.csv")


# In[11]:


df1.head()


# In[12]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.distplot(df1["speed"])

plt.subplot(1,2,2)
sns.distplot(df1["dist"])


# In[13]:


import scipy               #importing scipy lib for skew and kurtosis analyis


# In[14]:


scipy.stats.skew(df1["speed"])     #   Skewness < 0  ie -ve (more toward right)


# In[15]:


scipy.stats.kurtosis(df1["speed"])   


# In[16]:


scipy.stats.skew(df1["dist"])       #   Skewness > 0  ie +ve (more toward left)


# In[17]:


scipy.stats.kurtosis(df1["dist"])   


# In[18]:


df2 = pd.read_csv("Q9_b.csv")


# In[19]:


df2.head()


# In[20]:


scipy.stats.skew(df2["SP"])


# In[21]:


scipy.stats.skew(df2["WT"])


# In[22]:


scipy.stats.kurtosis(df2["SP"])


# In[23]:


scipy.stats.kurtosis(df2["WT"])


# In[24]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.distplot(df2["SP"])

plt.subplot(1,2,2)
sns.distplot(df2["WT"])


# # Q11  

# stats.norm.interval(% of confidence, mean, std deviation)

# In[25]:


z1=stats.norm.ppf(0.94)
z2=stats.norm.ppf(0.98)
z3=stats.norm.ppf(0.96)

np.round([z1,z2,z3],2)


# In[26]:


stats.norm.interval(0.94,200,30)


# In[27]:


stats.norm.interval(0.98,200,30)


# In[28]:


stats.norm.interval(0.96,200,30)


# # Q12

# In[29]:


df3=[34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
df3


# In[30]:


np.mean(df3)


# In[31]:


np.median(df3)


# In[32]:


np.var(df3,ddof=1)


# In[33]:


np.std(df3,ddof=1)


# In[34]:


sns.histplot(df3)


# # Q20

# In[35]:


from scipy.stats import norm


# In[36]:


df5 = pd.read_csv('Cars (1).csv')


# In[37]:


df5.tail()


# In[38]:


df5.info()


# In[39]:


# for case1 Probability of (mpg > 38)


mpg_38=df5[df5["MPG"]>38]            #finding index no (entries) of mpg > 38
len(mpg_38)


# In[40]:


px = len(mpg_38)/len(df5)
px


# In[41]:


# for case1 Probability of (mpg < 40)


mpg_40=df5[df5["MPG"]<40]  
len(mpg_40)


# In[42]:


py= len(mpg_40)/len(df5)
py


# In[43]:


# for case1 Probability of (20 < mpg < 50)

df6=df5[ df5["MPG"]<50]  
mpg_50_20=df6[df6["MPG"]>20]
len(mpg_50_20)


# In[44]:


pz= len(mpg_50_20)/len(df5)
pz


# # using p-norm method

# In[45]:


df10= df5["MPG"]


# In[46]:


#case1 MPG>38     (1-p(mpg<38))

prob_a=1- norm.cdf(38, df10.mean(), df10.std())
prob_a


# In[47]:


#case2 MPG<40

prob_b= norm.cdf(40, df10.mean(), df10.std())
prob_b


# In[48]:


#case3 20<MPG<50

prob_c= norm.cdf(50, df10.mean(), df10.std()) - norm.cdf(20, df10.mean(), df10.std())
prob_c


# # Q21

# In[49]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[50]:


df5.head()


# In[51]:


model1= smf.ols('MPG ~ WT+HP+SP+VOL',data=df5).fit()
model1.summary()


# In[52]:


qqplot = sm.qqplot(model1.resid,line="q")


# In[53]:


df21=pd.read_csv("wc-at.csv")


# In[54]:


df21.head()


# In[55]:


model2= smf.ols('Waist ~ AT',data=df21).fit()
model3= smf.ols('AT ~ Waist',data=df21).fit()


# In[56]:


qqplot1 = sm.qqplot(model2.resid,line="q")


# In[57]:


qqplot2 = sm.qqplot(model3.resid,line="q")


# # Q22

# In[58]:


stats.norm.ppf(0.60)


# In[59]:


stats.norm.ppf(0.90)


# In[60]:


stats.norm.ppf(0.94)


# # Q23

# In[61]:


#n=25 df=25-1=24 ci=(95%),(96%),(99%)


# In[65]:





# In[80]:


T1= stats.t.ppf((1+0.95)/2,24)
T2= stats.t.ppf((1+0.96)/2,24)
T3= stats.t.ppf((1+0.99)/2,24)

T_score=(T1,T2,T3)
T_score

