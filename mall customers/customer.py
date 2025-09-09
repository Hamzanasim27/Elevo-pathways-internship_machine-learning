import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Display basic information
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data Preprocessing
# Select relevant features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visual Exploration
plt.figure(figsize=(15, 5))

# Original data distribution
plt.subplot(1, 3, 1)
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], alpha=0.7)
plt.title('Original Data Distribution')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)

# Determine optimal number of clusters using Elbow Method
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Elbow Method plot
plt.subplot(1, 3, 2)
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)

# Silhouette Score plot
plt.subplot(1, 3, 3)
plt.plot(k_range, silhouette_scores, marker='o', color='red', linestyle='--')
plt.title('Silhouette Scores')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# Based on elbow method and silhouette score, choose optimal k
optimal_k = 5  # From visual inspection of the plots

# Apply K-Means with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# Visualize K-Means clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                     c=df['Cluster_KMeans'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('K-Means Clustering (5 clusters)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)

# Add cluster centers (original scale)
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers_original[:, 0], centers_original[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.legend()

# Try DBSCAN clustering (Bonus)
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)

plt.subplot(1, 2, 2)
scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                     c=df['Cluster_DBSCAN'], cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('DBSCAN Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Cluster Analysis
print("\n" + "="*50)
print("CLUSTER ANALYSIS - K-MEANS")
print("="*50)

# Analyze each cluster
cluster_analysis = df.groupby('Cluster_KMeans').agg({
    'Annual Income (k$)': ['mean', 'std', 'count'],
    'Spending Score (1-100)': ['mean', 'std'],
    'Age': 'mean',
    'Genre': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
}).round(2)

cluster_analysis.columns = ['Income_Mean', 'Income_Std', 'Count', 
                           'Spending_Mean', 'Spending_Std', 
                           'Age_Mean', 'Most_Common_Gender']

print(cluster_analysis)

# Interpret clusters
print("\n" + "="*50)
print("CLUSTER INTERPRETATION")
print("="*50)

cluster_interpretation = {
    0: "Low Income, Low Spenders - Budget-conscious customers",
    1: "High Income, Low Spenders - Conservative spenders",
    2: "Medium Income, Medium Spenders - Average customers",
    3: "Low Income, High Spenders - Carefree spenders",
    4: "High Income, High Spenders - Premium customers"
}

for cluster_id, interpretation in cluster_interpretation.items():
    if cluster_id in cluster_analysis.index:
        count = cluster_analysis.loc[cluster_id, 'Count']
        print(f"Cluster {cluster_id}: {interpretation} ({count} customers)")

# Compare clustering algorithms
print("\n" + "="*50)
print("CLUSTERING ALGORITHM COMPARISON")
print("="*50)

print(f"K-Means Silhouette Score: {silhouette_score(X_scaled, df['Cluster_KMeans']):.3f}")
print(f"DBSCAN Silhouette Score: {silhouette_score(X_scaled, df['Cluster_DBSCAN']):.3f}")
print(f"Number of clusters (K-Means): {df['Cluster_KMeans'].nunique()}")
print(f"Number of clusters (DBSCAN): {df['Cluster_DBSCAN'].nunique() - 1}")  # -1 for noise
print(f"Noise points in DBSCAN: {(df['Cluster_DBSCAN'] == -1).sum()}")

# Additional visualizations
plt.figure(figsize=(15, 4))

# Age distribution by cluster
plt.subplot(1, 3, 1)
sns.boxplot(x='Cluster_KMeans', y='Age', data=df)
plt.title('Age Distribution by Cluster')
plt.xlabel('Cluster')

# Gender distribution by cluster
plt.subplot(1, 3, 2)
gender_cluster = pd.crosstab(df['Cluster_KMeans'], df['Genre'], normalize='index')
gender_cluster.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Gender Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Proportion')
plt.legend(title='Gender')

# Spending vs Income with cluster colors
plt.subplot(1, 3, 3)
scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                     c=df['Cluster_KMeans'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS BY CLUSTER")
print("="*50)

summary_stats = df.groupby('Cluster_KMeans').agg({
    'Annual Income (k$)': ['mean', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'min', 'max'],
    'Age': 'mean'
}).round(2)

print(summary_stats)