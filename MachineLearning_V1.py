#!/usr/bin/env python
# coding: utf-8

# # Assignment to build a Machine Learning model
# ### sample-data from the GitHub repo
# ### https://github.com/internbuddy/foster-app.git
# 
# 

# ### import required libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
sns.set()


# In[2]:



# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_1 = {
    'IAM_SERVICE_ID': 'iam-ServiceId-a931fcf3-736e-4f54-9804-c12648191d14',
    'IBM_API_KEY_ID': 'Ml0eUH_SkhJWfVljS96hgS75c6emX00vIuoQFX5DBsnE',
    'ENDPOINT': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'IBM_AUTH_ENDPOINT': 'https://iam.ng.bluemix.net/oidc/token',
    'BUCKET': 'machinelearning-donotdelete-pr-dqtmrfnzsgcjxt',
    'FILE': 'internbuddy_data_v1.xlsx'
}


# In[3]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
client_8225f5a1ab90496fb7b062e6e942874b = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='Ml0eUH_SkhJWfVljS96hgS75c6emX00vIuoQFX5DBsnE',
    ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_8225f5a1ab90496fb7b062e6e942874b.get_object(Bucket='machinelearning-donotdelete-pr-dqtmrfnzsgcjxt',Key='internbuddy_data_v1.xlsx')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

data = pd.read_excel(body)
data.head()


# ### Type of Given Data

# In[4]:


data.dtypes


# In[5]:


data.isnull().sum()


# In[6]:


pd.set_option('display.max_columns',None)


# In[7]:


data.head(4)


# In[8]:


# Shape of the data frame
data.shape


# # Data pre-processing

# In[9]:


data['Degree'].value_counts()


# In[10]:


data['Other skills'].value_counts().head(20)


# In[11]:


data['Stream'].value_counts().head(10)


# In[12]:


data['Current Year Of Graduation'].value_counts().head(15)


# In[13]:


data['Current City'].value_counts()


# ### Information of data set

# In[14]:


data.info()


# ### Data columns present in data set

# In[15]:


data.columns


# In[16]:


#Droping the Unnamed: 10 column.
data.drop(['Unnamed: 10'], axis =1,inplace=True)


# In[17]:


data.head()


# In[18]:


plt.subplots(figsize=(20,7))
sns.boxplot(data=data)


# In[19]:


data.head(1)


# In[20]:


data['Performance_PG']=data['Performance_PG'].transform(lambda x:x.fillna('No degree') )


# In[21]:


df1=data['Performance_PG']=data['Performance_PG'].str.split('/',expand=True)
df2=data['Performance_UG']=data['Performance_UG'].str.split('/',expand=True)
df3=data['Performance_12']=data['Performance_12'].str.split('/',expand=True)
df4=data['Performance_10']=data['Performance_10'].str.split('/',expand=True)


# In[22]:


df1.drop(1,axis=1,inplace=True)
df2.drop(1,axis=1,inplace=True)
df3.drop(1,axis=1,inplace=True)
df4.drop(1,axis=1,inplace=True)


# In[23]:


df1 = df1.rename(columns={0:'Performance_PG'})
df2 = df2.rename(columns={0:'Performance_UG'})
df3 = df3.rename(columns={0:'Performance_12'})
df4 = df4.rename(columns={0:'Performance_10'})


# In[24]:


score=pd.concat([df1,df2,df3,df4],axis=1)
score.head()


# In[25]:


data.drop(['Performance_PG','Performance_UG','Performance_12','Performance_10'],axis=1,inplace=True)


# In[26]:


data=pd.concat([data,score],axis=1)


# In[27]:


data.head()


# In[28]:


df2=data['Performance_UG']=data['Performance_UG'].astype(float)
df3=data['Performance_12']=data['Performance_12'].astype(float)
df4=data['Performance_10']=data['Performance_10'].astype(float)


# In[29]:


df2=data['Performance_UG']=data['Performance_UG'].transform(lambda x:x.fillna(data['Performance_UG'].mean()))
df3=data['Performance_12']=data['Performance_12'].transform(lambda x:x.fillna(data['Performance_12'].mean()))
df4=data['Performance_10']=data['Performance_10'].transform(lambda x:x.fillna(data['Performance_10'].mean()))


# In[30]:


plt.subplots(figsize=(11,11))
sns.heatmap(data.corr(),square=True,annot=True)


# In[31]:



from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


# In[32]:


std=StandardScaler()
lable=preprocessing.LabelEncoder()


# In[33]:


data['Other skills']=data['Other skills'].transform(lambda x:x.fillna('unknow'))
data['Degree']=data['Degree'].transform(lambda x:x.fillna('unknow'))
data['Stream']=data['Stream'].transform(lambda x:x.fillna('unknow'))


# In[34]:


data_labled=data.apply(lable.fit_transform)


# In[35]:


data_scaled=std.fit_transform(data_labled)


# In[36]:


data_scaled=pd.DataFrame(data_scaled,columns=data.columns)


# In[37]:


data_scaled.head()


# In[38]:


data_scaled.drop('Application_ID',axis=1,inplace=True)


# ## Principal component analysis helps to increase signal to noise ratio and also in feature extraction and dimensionality reduction

# In[39]:


from sklearn.decomposition import PCA


# In[40]:


pca=PCA()
pc=pca.fit_transform(data_scaled)


# In[41]:


features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


# ## We can do feature selection by taking principal components which covers 90% of the data

# In[42]:


pca = PCA(n_components=13)
pc = pca.fit_transform(data_scaled)


# In[43]:


features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


# In[44]:


PCA_components = pd.DataFrame(pc,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13'])


# In[45]:


PCA_components.head()


# In[46]:


plt.subplots(figsize=(10,10))
sns.heatmap(PCA_components.corr(),square=True,annot=True)


# ### PCA works great with multi colenearity and we can see multi colenearity is removed totally.
# ## Now we can start doing clustering by finding out optimal k value.

# In[47]:


from sklearn.cluster import KMeans


# In[48]:


cluster_range = range( 1, 15 )
cluster_errors = []
for num_clusters in cluster_range:
    clusters = KMeans( num_clusters,n_init = 15, random_state=2)
    clusters.fit(PCA_components)
    labels = clusters.labels_
    centroids = clusters.cluster_centers_
    cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df[0:15]


# ## There is great drop in cluster errors after 2. So 2 might be used as k value.

# In[49]:


ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[50]:


kmeans = KMeans(n_clusters=2, n_init = 15, random_state=2)


# In[51]:


kmeans.fit(PCA_components)


# In[52]:


centroids=kmeans.cluster_centers_


# In[53]:


centroid_df = pd.DataFrame(centroids, columns = list(PCA_components) )
centroid_df


# In[54]:


data_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))

data_labels['labels'] = data_labels['labels'].astype('category')


# In[55]:


selection_data_labeled = PCA_components.join(data_labels)


# In[56]:


selection_clusters = selection_data_labeled.groupby(['labels'])
#wine_clusters = wine_data_attr.groupby(['clusters'])
df0=selection_clusters.get_group(0)
df1=selection_clusters.get_group(1)
finaldf = pd.concat([df0,df1])
finaldf.head()


# ## The created labels can be used as target variable for our model.
# 

# In[57]:


final_data = pd.concat([data_labled,finaldf['labels']],axis=1)


# In[58]:


final_data.head()


# ### Targeted Variable Label 0 represent the rejected candidates & Label 1 represent the Shortlisted candidates

# In[59]:


final_data["labels"]=final_data["labels"].replace(0, "Rejected")
final_data["labels"]=final_data["labels"].replace(1, "shortlisted for interview call letter")


# In[60]:


final_data.head()


# In[61]:


final_data['labels'].value_counts()


# In[62]:


final_data.loc[final_data['labels'] == 'Rejected']


# In[63]:


final_data.loc[final_data['labels'] == 'shortlisted for interview call letter']


# ## Comparing pairplot and boxplot we can see label 0 should be rejected and label 1 should be shortlisted for interview call letter.

# ## Since we are doing bivariate analysis predicting yes or no we can use classification techniques to predict the target variable.

# In[64]:


from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier


# In[65]:


x = final_data.drop(['Application_ID','labels'],axis=1)
y = final_data['labels']


# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[68]:


array = [LogisticRegressionCV(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=50,n_jobs=5),
        AdaBoostClassifier(),
        GradientBoostingClassifier()]


# In[69]:


for i in range (0,len(array)):
    array[i].fit(x_train,y_train)


# In[70]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report


# In[71]:


l=[]
for i in range (0,len(array)):
    y_pred=array[i].predict(x_test)
    l.append(accuracy_score(y_pred,y_test))
print(l)


# In[72]:


final = pd.DataFrame(l,index=('Logistic regreesion','Decision tree','Randomforest','Ada boost','Gradient boost'),columns=['Accuracy score'])
final


# In[73]:


rf = LogisticRegressionCV()


# In[74]:


rf_model = rf.fit(x_train,y_train)
rf_model


# In[75]:


y_pred = rf.predict(x_test)
print(y_pred)


# In[76]:


print(classification_report(y_pred,y_test))


# In[77]:


# create client to access our WML service
from watson_machine_learning_client import WatsonMachineLearningAPIClient

wml_credentials = {
  "apikey": "Lu1ARM3brLWwLjE2uJINFbM_7SoaLG2kafbhRTWey9ky",
  "iam_apikey_description": "Auto-generated for key 44e5f924-08b3-4473-b3ed-4cf8c4dc0ec7",
  "iam_apikey_name": "Service credentials-1",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/56a65e0027854707a4c79fcdd8e1bf3d::serviceid:ServiceId-9674b1ae-c2d0-42b8-950b-d35097f3777f",
  "instance_id": "d9acd9e5-f25d-4608-ada0-527a368779cd",
  "password": "Tanmayi@123",
  "url": "https://us-south.ml.cloud.ibm.com",
  "username": "tanmayi211m@gmail.com",
}

client = WatsonMachineLearningAPIClient(wml_credentials)
print(client.version)


# In[78]:


client.repository.list_models()
client.deployments.list()


# In[79]:


meta_props={client.repository.ModelMetaNames.NAME: "LogisticRegression model to predict selected"}
published_model = client.repository.store_model(model=rf, meta_props={client.repository.ModelMetaNames.NAME: "LogisticRegression model to predict selected"})


# In[80]:


client.repository.list_models()

# get UID of our just stored model
model_uid = client.repository.get_model_uid(published_model)
print("Model id: {}".format(model_uid))


# In[81]:


# create deployment
created_deployment = client.deployments.create(model_uid, name="student_selection_model_rf")

# new list of deployments
client.deployments.list()

# get UID of our new deployment
deployment_uid = client.deployments.get_uid(created_deployment)
print("Deployment id: {}".format(deployment_uid))
print(created_deployment)


# In[82]:


scoring_endpoint = client.deployments.get_scoring_url(created_deployment)
print(scoring_endpoint)


# In[83]:


# use our WML client to score our model

# add some test data
scoring_payload = {'fields': ['Current City','Python (out of 3)','R Programming (out of 3)','Deep Learning (out of 3)','PHP (out of 3)','MySQL (out of 3)','HTML (out of 3)','CSS (out of 3)',
'JavaScript (out of 3)','AJAX (out of 3)','Bootstrap (out of 3)','MongoDB (out of 3)','Node.js (out of 3)','ReactJS (out of 3)',
'Other skills','Degree','Stream','Current Year Of Graduation','Performance_PG','Performance_UG','Performance_12','Performance_10'], 
                    'values': [[4,2,0,0,2,2,2,2,2,0,0,0,0,0,193,12,22,11,63,167,130,107]]}


# In[84]:


predictions = client.deployments.score(scoring_endpoint, scoring_payload)
print('prediction',json.dumps(predictions, indent=2))


# In[ ]:




