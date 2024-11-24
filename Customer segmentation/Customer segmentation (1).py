#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[22]:



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[11]:


df=pd.read_csv("C:/Users/Dell i5/OneDrive - Cape Peninsula University of Technology/Desktop/Portfolio projects/Customer segmentation/Mall_Customers.csv")


# # Univariate analysis

# In[12]:


df.head()


# In[13]:


df.describe()


# In[23]:


sns.distplot(df["Annual Income (k$)"])


# In[20]:


df.columns


# In[24]:


columns=['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[26]:


sns.kdeplot(df['Annual Income (k$)'],shade=True,hue=df['Gender'])


# In[27]:



columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(df[i],shade=True,hue=df['Gender'])


# In[29]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df,x='Gender',y=df[i])


# In[32]:


df['Gender'].value_counts(normalize=True)


# # Bivariate analysis

# In[36]:


sns.scatterplot(data=df,x='Annual Income (k$)',y='Spending Score (1-100)')


# In[37]:


sns.pairplot(df,hue='Gender')


# In[38]:



df.groupby(['Gender'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[39]:


df.corr()


# In[42]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# # Clustering

# In[43]:


clustering1=KMeans(n_clusters=3)


# In[44]:


clustering1.fit(df[['Annual Income (k$)']])


# In[45]:


clustering1.labels_


# In[46]:


df['Income Cluster']=clustering1.labels_
df.head()


# In[47]:


df['Income Cluster'].value_counts()


# In[48]:


clustering1.inertia_


# In[50]:


intertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    intertia_scores.append(kmeans.inertia_)


# In[51]:


intertia_scores


# In[52]:


plt.plot(range(1,11),intertia_scores)


# In[53]:



df.columns


# In[54]:



df.groupby('Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[55]:



clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster'] =clustering2.labels_
df.head()


# In[56]:



intertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    intertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),intertia_scores2)


# In[57]:



centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[58]:



plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x ='Annual Income (k$)',y='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
plt.savefig('clustering_bivaraiate.png')


# In[59]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[60]:



df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[61]:



#mulivariate clustering 
from sklearn.preprocessing import StandardScaler


# In[62]:



scale = StandardScaler()


# In[63]:



df.head()


# In[64]:



dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[65]:



dff.columns


# In[66]:



dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]
dff.head()


# In[67]:


dff = scale.fit_transform(dff)


# In[68]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[69]:



intertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    intertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),intertia_scores3)


# In[70]:



df


# In[ ]:



df.to_csv('Clustering.csv')

