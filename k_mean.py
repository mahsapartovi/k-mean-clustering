import numpy as nm
import matplotlib.pyplot as mtp 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from google.colab import files
import numpy as nm
import matplotlib.pyplot as mtp    



uploaded = files.upload()

data= pd.read_csv('Mall_Customers.csv') 

data.head(20)

x= data.iloc[:,[3,4]]

# visualising the data to see the distribution.
plt.scatter(x.iloc[:,0],x.iloc[:,1] , c= 'green', s= 50)
plt.title('Income vs Capability')
plt.xlabel(' Annual Income')
plt.ylabel('Spending Capability')
plt.show()

from sklearn.cluster import KMeans  
wcss_list= []  #Initializing the list for the values of WCSS  
  
#Using for loop for iterations from 1 to 10.  
wcss=[]
for i in range(1,11):

    km= KMeans(n_clusters= i, init= 'k-means++', n_init= 10, max_iter=300, random_state= 0)
    km.fit(x)
    wcss.append(km.inertia_)  

plt.plot(range(1,11), wcss, c='blue')
plt.xlabel('No of Cluster')
plt.ylabel('WCSS')
plt.title('wcss vs no of cluster by using ElBOW method')
plt.show()

#fitting with k= 5
X= data.iloc[:,[3,4]].values
km= KMeans(n_clusters= 5, init= 'k-means++', n_init= 10, max_iter=300, random_state= 0)
y_kmeans= km.fit_predict(X)
y_kmeans

#cluster centers
km.cluster_centers_

#Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
#taking x as rows where cluster no is 0 and columns as annual salary and y with row with cluster 0 and columns with spending cap.
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'purple', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'yellow', label = 'Cluster 5')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

a= np.append(X,y_kmeans.reshape(200,1), axis=1)
d= pd.DataFrame(data=a, columns= ['age', 'capability', 'cluster_no'])
d.head()

from sklearn.cluster import DBSCAN
X= data.iloc[:,[3,4]].values

db= DBSCAN(eps=3, min_samples=4, metric='euclidean')
model= db.fit(X)
label=model.labels_
a=  {i:i for i in label if i != -1}
n_cluster= len(a.keys())
plt.figure(figsize=[10,5])
for i in range(0,n_cluster):
   plt.scatter(X[label==i,0], X[label==i,1], s=10, label= 'cluster: {}'.format(i) )

plt.xlabel('Income')
plt.ylabel('Capability')
plt.title('Clusters')
plt.legend(loc= 'upper right')
plt.show()

model.labels_

