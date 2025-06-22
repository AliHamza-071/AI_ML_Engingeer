
# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px

# Set the style for plots
#plt.style.use('seaborn')
#sns.set_palette('Set2')
#plt.rcParams['figure.figsize'] = (10, 6)

# plt.style.use('seaborn-whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (10, 6)
# Load the dataset
df = pd.read_csv('Mall_Customers.csv')
# Display basic information about the dataset
print('Dataset shape:', df.shape)
print('\nBasic information about the dataset:')
print(df.info())
print('\nFirst 5 rows of the dataset:')
print(df.head())
print('\nSummary statistics:')
print(df.describe())
# Check for missing values
print('\nMissing values in each column:')
print(df.isnull().sum())

# Gender distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
# Annual Income distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Annual Income (k$)', bins=20, kde=True)
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.show()
# Spending Score distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Spending Score (1-100)', bins=20, kde=True)
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.show()
# Create a pairplot to visualize relationships between numerical features
plt.figure(figsize=(12, 8))
sns.pairplot(df, hue='Gender')
plt.suptitle('Pairplot of Mall Customer Features', y=1.02)
plt.show()
# Correlation matrix
plt.figure(figsize=(10, 8))
numerical_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
# Feature selection for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    # Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()
# Silhouette Score Analysis
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')
plt.title('Silhouette Score Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
# Based on the Elbow Method and Silhouette Score, let's choose the optimal k
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=300, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)


# Visualize the clusters
plt.figure(figsize=(12, 8))
plt.scatter(X[df['KMeans_Cluster'] == 0]['Annual Income (k$)'], X[df['KMeans_Cluster'] == 0]['Spending Score (1-100)'], s=100, c='red', label='Cluster 1')
plt.scatter(X[df['KMeans_Cluster'] == 1]['Annual Income (k$)'], X[df['KMeans_Cluster'] == 1]['Spending Score (1-100)'], s=100, c='blue', label='Cluster 2')
plt.scatter(X[df['KMeans_Cluster'] == 2]['Annual Income (k$)'], X[df['KMeans_Cluster'] == 2]['Spending Score (1-100)'], s=100, c='green', label='Cluster 3')
plt.scatter(X[df['KMeans_Cluster'] == 3]['Annual Income (k$)'], X[df['KMeans_Cluster'] == 3]['Spending Score (1-100)'], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[df['KMeans_Cluster'] == 4]['Annual Income (k$)'], X[df['KMeans_Cluster'] == 4]['Spending Score (1-100)'], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', marker='*', label='Centroids')
plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
# Include Age as a feature for clustering
X_with_age = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
X_with_age_scaled = scaler.fit_transform(X_with_age)

# Determine optimal clusters for 3D data
wcss_3d = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(X_with_age_scaled)
    wcss_3d.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss_3d, marker='o', linestyle='-')
plt.title('Elbow Method for 3D Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
# Apply K-Means with the optimal K for 3D clustering
optimal_k_3d = 5
kmeans_3d = KMeans(n_clusters=optimal_k_3d, init='k-means++', n_init=10, max_iter=300, random_state=42)
df['KMeans_3D_Cluster'] = kmeans_3d.fit_predict(X_with_age_scaled)
# Create 3D scatter plot
fig = px.scatter_3d(df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                   color='KMeans_3D_Cluster', symbol='Gender',
                   title='3D Customer Segmentation',
                   labels={'KMeans_3D_Cluster': 'Cluster'})
fig.update_layout(scene=dict(xaxis_title='Age', yaxis_title='Annual Income (k$)', zaxis_title='Spending Score (1-100)'))
fig.show()

# Create a linkage matrix for hierarchical clustering
Z = linkage(X_scaled, method='ward')
# Plot the dendrogram
plt.figure(figsize=(12, 8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
dendrogram(Z, truncate_mode='lastp', p=30, leaf_font_size=10., leaf_rotation=90.)
plt.axhline(y=6, color='r', linestyle='--')  # Drawing a horizontal line to suggest the number of clusters
plt.show()


# Apply Agglomerative Clustering
hierarchical_cluster = AgglomerativeClustering(n_clusters=5, linkage='ward') 
df['Hierarchical_Cluster'] = hierarchical_cluster.fit_predict(X_scaled)
# Visualize the hierarchical clusters
plt.figure(figsize=(12, 8))
plt.scatter(X[df['Hierarchical_Cluster'] == 0]['Annual Income (k$)'], X[df['Hierarchical_Cluster'] == 0]['Spending Score (1-100)'], s=100, c='red', label='Cluster 1')
plt.scatter(X[df['Hierarchical_Cluster'] == 1]['Annual Income (k$)'], X[df['Hierarchical_Cluster'] == 1]['Spending Score (1-100)'], s=100, c='blue', label='Cluster 2')
plt.scatter(X[df['Hierarchical_Cluster'] == 2]['Annual Income (k$)'], X[df['Hierarchical_Cluster'] == 2]['Spending Score (1-100)'], s=100, c='green', label='Cluster 3')
plt.scatter(X[df['Hierarchical_Cluster'] == 3]['Annual Income (k$)'], X[df['Hierarchical_Cluster'] == 3]['Spending Score (1-100)'], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[df['Hierarchical_Cluster'] == 4]['Annual Income (k$)'], X[df['Hierarchical_Cluster'] == 4]['Spending Score (1-100)'], s=100, c='magenta', label='Cluster 5')
plt.title('Customer Segments based on Annual Income and Spending Score (Hierarchical Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
# Analyze cluster characteristics
cluster_analysis = df.groupby('KMeans_Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'})

print("\nCluster Analysis (K-Means):")
print(cluster_analysis)

# Visualize cluster characteristics
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='KMeans_Cluster', y='Age', data=df)
plt.title('Age Distribution by Cluster')

plt.subplot(2, 2, 2)
sns.boxplot(x='KMeans_Cluster', y='Annual Income (k$)', data=df)
plt.title('Income Distribution by Cluster')

plt.subplot(2, 2, 3)
sns.boxplot(x='KMeans_Cluster', y='Spending Score (1-100)', data=df)
plt.title('Spending Score Distribution by Cluster')

plt.subplot(2, 2, 4)
cluster_sizes = df['KMeans_Cluster'].value_counts().sort_index()
plt.pie(cluster_sizes, labels=[f'Cluster {i+1}' for i in range(len(cluster_sizes))], autopct='%1.1f%%')
plt.title('Cluster Size Distribution')

plt.tight_layout()
plt.show()

# Gender distribution across clusters
plt.figure(figsize=(12, 6))
crosstab = pd.crosstab(df['KMeans_Cluster'], df['Gender'])
crosstab.plot(kind='bar', stacked=True)
plt.title('Gender Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.grid(axis='y')
plt.show()
# Customer segmentation summary
print("\nCustomer Segmentation Summary:")
for cluster in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    print(f"\nCluster {cluster+1}:")
    print(f"Number of customers: {len(cluster_data)}")
    print(f"Average Age: {cluster_data['Age'].mean():.1f} years")
    print(f"Average Annual Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
    print(f"Average Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f}/100")
    print(f"Gender ratio: {100*sum(cluster_data['Gender']=='Female')/len(cluster_data):.1f}% Female, {100*sum(cluster_data['Gender']=='Male')/len(cluster_data):.1f}% Male")
