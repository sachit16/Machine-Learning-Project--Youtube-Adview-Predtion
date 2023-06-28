#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
#path="C:\Users\Sachit Bhor\Downloads\train.csv"
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


#path="\Desktop\train.csv++++"
DT = pd.read_csv(r'C:/Users/Sachit Bhor/Downloads/train.csv')



# In[21]:


DT


# In[22]:


DT.shape


# In[23]:


DT.head()


# In[24]:


DT.tail()


# In[25]:


DT.info


# In[26]:


DT.describe


# In[27]:


plt.hist(DT["category"],color='green')
plt.show()
plt.plot(DT["adview"],color='green')
plt.show()


# In[28]:


DT.isnull().any()


# In[29]:


DT =DT[DT["adview"] <2000000]


# In[30]:


DT


# In[31]:


plt.plot(DT["adview"],color='green')
plt.show()


# In[32]:


a=corelation =DT.corr()
print(a)


# In[33]:


import seaborn as sns
f, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(corelation, mask=np.zeros_like(corelation, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, annot=True)
plt.show()


# In[34]:


DT=DT[DT.views!='F']
DT=DT[DT.likes!='F']
DT=DT[DT.dislikes!='F']
DT=DT[DT.comment!='F']
DT.head()


# In[35]:


category={'A': 1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}
DT["category"]=DT["category"].map(category)
DT


# In[36]:


DT["views"] = pd.to_numeric(DT["views"])
DT["comment"] = pd.to_numeric(DT["comment"])
DT["likes"] = pd.to_numeric(DT["likes"])
DT["dislikes"] = pd.to_numeric(DT["dislikes"])
DT["adview"]=pd.to_numeric(DT["adview"])
column_vidid=DT['vidid']
DT


# In[37]:


from sklearn.preprocessing import LabelEncoder
DT['duration']=LabelEncoder().fit_transform(DT['duration'])
DT['vidid']=LabelEncoder().fit_transform(DT['vidid'])
DT['published']=LabelEncoder().fit_transform(DT['published'])
DT


# In[38]:


import datetime
import time

def checki(x):
 y = x[2:]
 h = ''
 m = ''
 s = ''
 mm = ''
 P = ['H','M','S']
 for i in y:
  if i not in P:
   mm+=i
  else:
   if (i=="H"):
    h = mm 
    mm = ''
   elif (i == "M"):
    m = mm
    mm = ''
   else:
    s = mm
    mm = ''
 if (h==''):
  h = '00'
 if (m == ''):
  m = '00'
 if (s==''):
  s='00'
 bp = h+':'+m+':'+s
 return bp


# In[39]:


train=pd.read_csv(r'C:/Users/Sachit Bhor/Downloads/train.csv')
mp = pd.read_csv(r'C:/Users/Sachit Bhor/Downloads/train.csv')["duration"]
time = mp.apply(checki)
def func_sec(time_string):
 h, m, s = time_string.split(':')
 return int(h) * 3600 + int(m) * 60 + int(s)
time1=time.apply(func_sec)
DT["duration"]=time1
DT


# In[41]:


corelation = DT.corr()


# In[42]:


import seaborn as sns
f, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(corelation, mask=np.zeros_like(corelation, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, annot=True)
plt.show()


# In[43]:


Y_train = pd.DataFrame(data = DT.iloc[:, 1].values, columns = ['target'])
DT=DT.drop(["adview"],axis=1)
DT=DT.drop(["vidid"],axis=1)
DT


# In[44]:


DT.head()


# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DT, Y_train, test_size=0.2, random_state=42)
X_train.shape


# In[46]:


DT.iloc[:,-2:8]
DT.iloc[:,2:8].values
DT=DT.dropna()
DT


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DT, Y_train, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[48]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
print(X_train.mean())


# In[49]:


from sklearn import metrics
mae=[0,0,0,0,0]
mse=[0,0,0,0,0]
rmse=[0,0,0,0,0]
def print_error(x_test, y_test, model_name):
  prediction = model_name.predict(x_test)
  a=metrics.mean_absolute_error(y_test, prediction)
  b=metrics.mean_squared_error(y_test, prediction)
  c=np.sqrt(b)
  print('Mean Absolute Error:', a)
  print('Mean Squared Error:', b)
  print('Root Mean Squared Error:', c)
  plt.figure(figsize=(10, 15))
  plt.scatter(range(len(prediction)), prediction, color='red', s=1)
  plt.scatter(range(len(y_test)), y_test, color='green', s=1)
  plt.show()
  return ([a,b,c])


# In[50]:


from sklearn import linear_model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
mae[0],mse[0],rmse[0]=print_error(X_test,y_test, linear_regression)


# In[51]:


from sklearn.svm import SVR
supportvector_regressor = SVR()
supportvector_regressor.fit(X_train,y_train.values.ravel())
mae[1],mse[1],rmse[1]=print_error(X_test,y_test, supportvector_regressor)


# In[53]:


from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
mae[2],mse[2],rmse[2]=print_error(X_test,y_test, decision_tree)


# In[54]:


from sklearn.ensemble import RandomForestRegressor
n_estimators = 200
max_depth = 25
min_samples_split = 15
min_samples_leaf = 2
random_forest = RandomForestRegressor(n_estimators = n_estimators , max_depth = max_depth, min_samples_split =  min_samples_split  , min_samples_leaf = min_samples_leaf )
random_forest.fit(X_train,y_train.values.ravel())
mae[3],mse[3],rmse[3]=print_error(X_test,y_test, random_forest)


# In[55]:


import keras
from keras.layers import Dense
ann = keras.models.Sequential([
Dense(6, activation="relu",
input_shape=X_train.shape[1:]),
Dense(6,activation="relu"),
Dense(1)
])
optimizer=keras.optimizers.Adam()
loss=keras.losses.mean_squared_error
ann.compile(optimizer=optimizer,loss=loss,metrics=["mean_squared_error"])
history=ann.fit(X_train,y_train,epochs=100)
ann.summary()
mae[4],mse[4],rmse[4]=print_error(X_test,y_test,ann)


# In[56]:


ann.summary()


# In[57]:


plt.plot(["LinearRegression","SVR","DecisionTree","RandomForest","ANN"],mae)
plt.show()


# In[58]:


plt.plot(["LinearRegression","SVR","DecisionTree","RandomForest","ANN"],mse)
plt.show()


# In[59]:


plt.plot(["LinearRegression","SVR","DecisionTree","RandomForest","ANN"],rmse)
plt.show()


# In[60]:


import joblib
joblib.dump(supportvector_regressor, "First_Model_SVR_Youtube_Adview.pkl")


# In[61]:


ann.save("First_Model_ANN_Youtube_Adview.h5")


# In[77]:


Test = pd.read_csv(r'C:/Users/Sachit Bhor/Downloads/train.csv')
Test


# In[78]:


Test.head()


# In[79]:


Test.tail()


# In[80]:


Test.info


# In[81]:


Test.describe


# In[82]:


Test.isnull().any()


# In[83]:


Test=Test[Test.views!='F']
Test=Test[Test.likes!='F']
Test=Test[Test.dislikes!='F']
Test=Test[Test.comment!='F']
Test


# In[84]:


category={'A': 1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}
Test["category"]=Test["category"].map(category)
Test


# In[85]:


Test["views"] = pd.to_numeric(Test["views"])
Test["comment"] = pd.to_numeric(Test["comment"])
Test["likes"] = pd.to_numeric(Test["likes"])
Test["dislikes"] = pd.to_numeric(Test["dislikes"])
Test["adview"]=pd.to_numeric(Test["adview"])
column_vidid_2=Test['vidid']
Test


# In[86]:


from sklearn.preprocessing import LabelEncoder
Test['duration']=LabelEncoder().fit_transform(Test['duration'])
Test['vidid']=LabelEncoder().fit_transform(Test['vidid'])
Test['published']=LabelEncoder().fit_transform(Test['published'])
Test


# In[88]:


import datetime
import time
def checki(x):
  y = x[2:]
  h = ''
  m = ''
  s = ''
  mm = ''
  P = ['H','M','S']
  for i in y:
    if i not in P:
      mm+=i
    else:
      if(i=="H"):
        h = mm
        mm = ''
      elif(i == "M"):
        m = mm
        mm = ''
      else:
        s = mm
        mm = ''
  if(h==''):
    h = '00'
  if(m == ''):
    m = '00'
  if(s==''):
    s='00'
  bp = h+':'+m+':'+s
  return bp
  train= pd.read_csv(r'C:/Users/Sachit Bhor/Downloads/train.csv')

mp =pd.read_csv(r'C:/Users/Sachit Bhor/Downloads/train.csv')["duration"]
time = mp.apply(checki)

def func_sec(time_string):
  h, m, s = time_string.split(':')
  return int(h) * 3600 + int(m) * 60 + int(s)

time1=time.apply(func_sec)

Test["duration"]=time1
Test


# In[89]:


Test.iloc[:,-2:8]
Test.iloc[:,2:8].values
Test=Test.dropna()
Test=Test.drop(["adview"],axis=1)
Test=Test.drop(["vidid"],axis=1)
Test


# In[92]:


import joblib
classifier = joblib.load("First_Model_SVR_Youtube_Adview.pkl")
prediction = classifier.predict(Test)


# In[93]:


prediction


# In[94]:


np.savetxt('Prediction.csv',prediction,delimiter=',')


# In[95]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_test = Test
X_test=scaler.fit_transform(X_test)


# In[97]:


from keras.models import load_model
model = load_model("First_Model_ANN_Youtube_Adview.h5")


# In[98]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_test = Test
X_test=scaler.fit_transform(X_test)


# In[99]:


predictions = model.predict(X_test)


# In[100]:


predictions=pd.DataFrame(predictions)
predictions.info()


# In[101]:


predictions = predictions.rename(columns={0: "Adview"})


# In[102]:


predictions.head()


# In[103]:


predictions.to_csv('predictions.csv')


# In[ ]:




