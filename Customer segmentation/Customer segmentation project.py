#!/usr/bin/env python
# coding: utf-8

# # Mall CUSTOMER SEGMENTATION ANALYSIS WITH PYTHON

# # INTRODUCTION

# ## Customer Segmentation

# Customer segmentation is the practice of dividing a company’s customers into groups that reflect similarity among customers in each group. The goal of segmenting customers is to decide how to relate to customers in each segment in order to maximize the value of each customer to the business.

# ## Customer Segmentation Analysis

# Customer segmentation analysis is the process performed when looking to discover insights that define specific segments of customers. Marketers and brands leverage this process to determine what campaigns, offers, or products to leverage when communicating with specific segments.
# Customer segmentation analysis is the process performed when looking to discover insights that define specific segments of customers. Marketers and brands leverage this process to determine what campaigns, offers, or products to leverage when communicating with specific segments.
# 
# 

# # METHDOLOGY

# This project uses common cluster analysis method known as k-means cluster analysis, sometimes referred to as scientific segmentation. The clusters that result assist in better customer modeling and predictive analytics, and are also are used to target customers with offers and incentives personalized to their wants, needs and preferences.the process is not based on any predetermined thresholds or rules. Rather, the data itself reveals the customer prototypes that inherently exist within the population of customers
# 
# K-Means is the most popular clustering algorithm. It uses an iterative technique to group unlabeled data into K clusters based on cluster centers (centroids). The data in each cluster are chosen such that their average distance to their respective centroid is minimized.
# 
# 1. Randomly place K centroids for the initial clusters.
# 2. Assign each data point to their nearest centroid.
# 3. Update centroid locations based on the locations of the data points.
# Repeat Steps 2 and 3 until points don’t move between clusters and centroids stabilize.

# # RESULTS 

# Import necessary libraries  and load data

# In[1]:



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("C:/Users/Dell i5/OneDrive - Cape Peninsula University of Technology/Desktop/Portfolio projects/Customer segmentation/Mall_Customers.csv")


# # Univariate analysis

# In[3]:


df.head()


# In[4]:


df.describe()


# 
# The annual income of customers ranges from a low of 15 thousand dollar to a high of 137 thousand dollars with an average of 60 thousand dollars.The median anual income suggest that half of customers earn less that 61.5 thousand dollars while the other half earns more than the amount.

# In[20]:


df.columns


# In[24]:


#plot the distribution of variables (Age, Annual Income (k$), Spending Score (1-100))
columns=['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# The distribution of age and annual income are slightly right skewed. it suggests that there are more younger customers(less than 50) compared to older custmers and that a single peak of the range 60-70 thousand dollars is most common. Moreover the distribution of spending score apears to be roughly normally distributed with the most common score rughly at 50.

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


# females customers have the lower average annual income campared to males and they spend more than males, age has a negative correlation with annual income and spending score while annual income have a positive correlation with the spending score of customers

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


# cluster 0 represents low income and low spending customers
# cluster 1 represents high income and high spendng customers
# cluster 2 represents midium income and medium spending customers
# cluster 3 represents low income and low spending customers 
# cluster 4 represents high income and low spending customers
# 
# tailored marketing strategies for each group can be made. The business may want to have focus promotions on cluster 1 to increase retention, offer incentives for cluster 4 to increase spending, and facilitate a brother investigation to understand why cluster 3 is spending low and explore ways to which these custpmers can be engaged. The results further shows that in the spending and income cluster via gender crosstable,in cluster 3 percentage females is highest at 61% compared to males at 39%, these have an average age of 45 years. 
# 
# 
# 

# In[59]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[60]:



df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# # Mulivariate clustering 

# In[61]:


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

