#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

# Load your CSV file
df = pd.read_csv("filtered_data_jobs.csv")

# Define data-related keywords
keywords = [
    "data analyst", "data analysis", "data scientist", "data science",
    "data engineer", "data mining", "machine learning", "data architect",
    "business intelligence", "analytics", "data visualization"
]

# Convert Job Title to lowercase for consistent matching
df['job_title_lower'] = df['Job Title'].astype(str).str.lower()

# Filter rows where the job title contains any of the keywords
mask = df['job_title_lower'].apply(lambda x: any(kw in x for kw in keywords))
filtered_df = df[mask].copy()

# Drop the helper column
filtered_df.drop(columns=['job_title_lower'], inplace=True)

# Drop unwanted columns
columns_to_drop = [
    'latitude', 'longitude', 'Job Posting Date', 'Contact Person',
    'Contact', 'Job Portal', 'Job Description', 'Benefits',
    'skills', 'Responsibilities'
]
filtered_df.drop(columns=[col for col in columns_to_drop if col in filtered_df.columns], inplace=True)

# Cap to 5000 rows (or fewer if fewer exist)
filtered_df = filtered_df.head(5000)

# Save the filtered DataFrame
filtered_df.to_csv("clean_data_jobs.csv", index=False)

print(f"Filtered down to {len(filtered_df)} data-related job postings based on Job Title.")



# In[6]:


import pandas as pd
import json

# Load your CSV file
df = pd.read_csv("clean_data_jobs.csv")

# Define data-related keywords
keywords = [
    "data analyst", "data analysis", "data scientist", "data science",
    "data engineer", "data mining", "machine learning", "data architect",
    "business intelligence", "analytics", "data visualization"
]

# Convert Job Title to lowercase for consistent matching
df['job_title_lower'] = df['Job Title'].astype(str).str.lower()

# Filter rows where the job title contains any of the keywords
mask = df['job_title_lower'].apply(lambda x: any(kw in x for kw in keywords))
filtered_df = df[mask].copy()

# Drop the helper column
filtered_df.drop(columns=['job_title_lower'], inplace=True)

# Drop unwanted columns
columns_to_drop = [
    'latitude', 'longitude', 'Job Posting Date', 'Contact Person',
    'Contact', 'Job Portal', 'Job Description', 'Benefits',
    'skills', 'Responsibilities'
]
filtered_df.drop(columns=[col for col in columns_to_drop if col in filtered_df.columns], inplace=True)

# Extract only the 'Sector' from the Company Profile JSON string
def extract_sector(profile_str):
    try:
        profile_dict = json.loads(profile_str.replace("'", '"'))  # Fix single quotes if needed
        return profile_dict.get('Sector', None)
    except Exception:
        return None

if 'Company Profile' in filtered_df.columns:
    filtered_df['Company Profile'] = filtered_df['Company Profile'].apply(extract_sector)

# Cap to 5000 rows
filtered_df = filtered_df.head(5000)

# Save the filtered DataFrame
filtered_df.to_csv("filtered_data_jobs_w_sector.csv", index=False)

print(f"Filtered down to {len(filtered_df)} data-related job postings based on Job Title.")


# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE

# 1. Data Preprocessing

# Load data
df = pd.read_csv('filtered_data_jobs_w_sector.csv')  # Replace with your file path

# Handle missing values (impute or drop)
df = df.dropna()  # or df.fillna('Unknown') if appropriate for categorical columns

# Encode categorical variables
label_encoder = LabelEncoder()
df['Qualifications_encoded'] = label_encoder.fit_transform(df['Qualifications'])
df['Job Title_encoded'] = label_encoder.fit_transform(df['Job Title'])
df['Work Type_encoded'] = label_encoder.fit_transform(df['Work Type'])
df['Company Size_encoded'] = label_encoder.fit_transform(df['Company Size'])
df['Preference_encoded'] = label_encoder.fit_transform(df['Preference'])

# Normalize 'Experience' (this can be tricky since it's a range, so we need to process it)
def process_experience(experience):
    # Remove non-numeric characters (e.g., 'Years', 'to') and handle the range format
    experience = experience.replace('Years', '').strip()
    if 'to' in experience:
        experience_range = experience.split(' to ')
        try:
            return (int(experience_range[0]) + int(experience_range[1])) / 2  # Average of the range
        except ValueError:
            return np.nan  # In case of any parsing errors, return NaN
    else:
        try:
            return int(experience)  # For cases without 'to', convert directly to int
        except ValueError:
            return np.nan  # In case the experience format is unexpected, return NaN

# Apply the updated function
df['Experience_normalized'] = df['Experience'].apply(process_experience)

# Scaling numeric features
scaler = StandardScaler()
df['Experience_normalized_scaled'] = scaler.fit_transform(df[['Experience_normalized']])

# 2. Feature Selection for Clustering
X = df[['Experience_normalized_scaled', 'Qualifications_encoded', 'Job Title_encoded', 'Work Type_encoded', 'Company Size_encoded']]

# 3. Choosing Similarity Metric (Euclidean distance for K-Means)

# K-Means Clustering - Use Euclidean distance
kmeans = KMeans(init="k-means++", n_init=10, random_state=42)

# 4. Determining k (number of clusters) - Using Elbow Method and Silhouette Score

# Elbow Method to determine optimal k
inertia = []
silhouette_scores = []
k_range = range(2, 11)  # Trying different values of k (2 to 10)

for k in k_range:
    kmeans.set_params(n_clusters=k)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot Elbow Method (Inertia)
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Plot Silhouette Score
plt.figure(figsize=(8, 6))
plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Score For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Based on the plots, choose the optimal k (let's assume it's 4 for example)
k_optimal = 4
kmeans.set_params(n_clusters=k_optimal)
y_kmeans = kmeans.fit_predict(X)

# 5. Interpreting Clusters

# Add cluster labels to the dataframe
df['Cluster'] = y_kmeans

# Analyze the average salary per cluster
df['Salary Range'] = df['Salary Range'].apply(lambda x: x.split('-'))  # Split salary range (if needed)
df['Salary Low'] = df['Salary Range'].apply(lambda x: int(x[0].replace('$', '').replace('K', '').replace(',', '').strip()) * 1000 if x[0] else np.nan)
df['Salary High'] = df['Salary Range'].apply(lambda x: int(x[1].replace('$', '').replace('K', '').replace(',', '').strip()) * 1000 if x[1] else np.nan)

# Calculate the average salary for each cluster
cluster_salary_avg = df.groupby('Cluster')[['Salary Low', 'Salary High']].mean()

# Show the average salary in each cluster
print(cluster_salary_avg)

# Examining the features that dominate each cluster
cluster_summary = df.groupby('Cluster').mean()

# Show the cluster summary (for numerical features)
print(cluster_summary)

# Show examples from each cluster
examples_per_cluster = df.groupby('Cluster').apply(lambda x: x.head(1))  # Display one example per cluster
print(examples_per_cluster)

# 6. Visualize Clusters
# Create a combined metric (example: using a weighted sum of the normalized features)
df['combined_metric'] = (
    df['Experience_normalized_scaled'] * 0.4 + 
    df['Qualifications_encoded'] * 0.2 + 
    df['Job Title_encoded'] * 0.2 + 
    df['Work Type_encoded'] * 0.1 + 
    df['Company Size_encoded'] * 0.1
)

# Now, randomly sample 50 rows from the dataframe
sampled_df = df.sample(n=50, random_state=42)

# Create a scatter plot of the combined_metric vs Salary Low for the sampled points
plt.figure(figsize=(8, 6))
plt.scatter(sampled_df['combined_metric'], sampled_df['Salary Low'], c=sampled_df['Cluster'], cmap='viridis', marker='o', s=100, edgecolor='k')

# Label the axes and add a color legend for clusters
plt.title('Combined Metric vs. Salary Low (50 Random Points)', fontsize=16)
plt.xlabel('Combined Metric (Experience, Qualifications, Job Title, Company Size)', fontsize=12)
plt.ylabel('Salary Low', fontsize=12)

# Show color legend for clusters
plt.colorbar(label='Cluster')

# Show plot
plt.show()





# Standardize all numeric features (including those that were encoded)
X_scaled = df[['Experience_normalized_scaled', 'Qualifications_encoded', 'Job Title_encoded', 'Work Type_encoded', 'Company Size_encoded']]
X_scaled = StandardScaler().fit_transform(X_scaled)

# Apply PCA (reduce to 2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA components to the DataFrame for visualization
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Verify the PCA columns are added by inspecting the dataframe
print(df[['PCA1', 'PCA2']].head())

# Now, you can plot the PCA components
sampled_df = df.sample(n=300, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(sampled_df['PCA1'], sampled_df['PCA2'], c=sampled_df['Cluster'], cmap='viridis', marker='o', s=100, edgecolor='k')
plt.title('PCA Components vs. Salary Low (50 Random Points)', fontsize=16)
plt.xlabel('PCA1', fontsize=12)
plt.ylabel('PCA2', fontsize=12)
plt.colorbar(label='Cluster')
plt.show()



# Use your scaled features (e.g., X or X_scaled)
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Cluster'], cmap='viridis', s=60, edgecolor='k')
plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(label='Cluster')
plt.show()




# In[ ]:




