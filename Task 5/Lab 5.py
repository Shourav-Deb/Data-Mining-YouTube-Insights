# =====================================================
# Task 5: Clustering YouTube Comments
# =====================================================


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = os.path.join(ROOT_DIR, "cleaned_comments.csv")

print("CSV path:", INPUT_FILE)
print("File exists:", os.path.exists(INPUT_FILE))

df = pd.read_csv(INPUT_FILE)

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print(f"\nMy dataset contains {df.shape[0]} comments and {df.shape[1]} columns.")

print("\nCritical Prompt #1:")
if df.shape[0] < 50:
    print("My dataset is quite small, so the clustering may work technically but the topics may not be very meaningful. With fewer comments, clusters can overlap and become less reliable.")
elif df.shape[0] < 200:
    print("My dataset is moderate in size, so I may find some useful themes, but the quality of the clusters will still depend on how repetitive or varied the comments are.")
else:
    print("My dataset is fairly large, so I expect the clustering to show more meaningful themes. A larger number of comments usually gives the algorithms more repeated patterns to group together.")

if 'cleaned_tokens' in df.columns:
    corpus = df['cleaned_tokens'].apply(
        lambda x: ' '.join(ast.literal_eval(x)) if isinstance(x, str) else ' '.join(x)
    ).tolist()
elif 'cleaned_text' in df.columns:
    corpus = df['cleaned_text'].astype(str).tolist()
else:
    raise Exception("No usable text column found.")

vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(corpus)

print("\nTF-IDF matrix shape:")
print(X.shape)

inertia = []
K = range(2, 8)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid(True)

elbow_plot_path = os.path.join(OUTPUT_DIR, "elbow_plot.png")
plt.savefig(elbow_plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Elbow plot saved as: {elbow_plot_path}")

print("\nInertia values:")
for k, value in zip(K, inertia):
    print(f"k = {k}, inertia = {value:.4f}")

X_pca = PCA(n_components=2).fit_transform(X.toarray())

X_scaled = StandardScaler(with_mean=False).fit_transform(X)

results = []

for eps in [0.3, 0.5, 0.7]:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    results.append({
        'eps': eps,
        'min_samples': 5,
        'clusters': n_clusters,
        'noise_points': n_noise
    })

    print(f"eps={eps}: clusters={n_clusters}, noise_points={n_noise}")

results_df = pd.DataFrame(results)

print("\nDBSCAN tuning results:")
print(results_df)

if results_df['clusters'].sum() == 0:
    print("\nThe first three eps values did not form any clusters, so I tested a few larger values as well.\n")

    extra_results = []
    for eps in [5, 10, 20, 30, 40]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        extra_results.append({
            'eps': eps,
            'min_samples': 5,
            'clusters': n_clusters,
            'noise_points': n_noise
        })

        print(f"extra eps={eps}: clusters={n_clusters}, noise_points={n_noise}")

    extra_results_df = pd.DataFrame(extra_results)

    print("\nAdditional DBSCAN results:")
    print(extra_results_df)

print("\n1st part code complete.\n\n")

optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['kmeans_label'] = kmeans.fit_predict(X)

terms = vectorizer.get_feature_names_out()

print(f"K-Means results with k = {optimal_k}\n")
for i in range(optimal_k):
    cluster_center = kmeans.cluster_centers_[i]
    top_indices = cluster_center.argsort()[-10:][::-1]
    top_terms = [terms[idx] for idx in top_indices]
    print(f"Cluster {i}: {', '.join(top_terms)}")

print("\nK-Means cluster counts:")
print(df['kmeans_label'].value_counts().sort_index())

print("\nSample comments from each K-Means cluster:")
for i in range(optimal_k):
    print(f"\nCluster {i} sample comments:")
    sample_comments = df[df['kmeans_label'] == i]['comment_text'].head(5).tolist()
    for j, comment in enumerate(sample_comments, 1):
        print(f"{j}. {comment}")

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['kmeans_label'], cmap='tab10', alpha=0.6)
plt.title('K-Means Cluster Visualization (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.grid(True)

kmeans_plot_path = os.path.join(OUTPUT_DIR, "kmeans_pca_plot.png")
plt.savefig(kmeans_plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nK-Means PCA plot saved as: {kmeans_plot_path}")

print("\nCritical Prompt #2:")
print("I chose k = 4 because the elbow plot showed a gradual decrease in inertia without a very sharp break, so 4 seemed like a reasonable middle choice. It allowed me to separate the comments into multiple themes without creating too many very small clusters. Some of the clusters were understandable. For example, Cluster 2 was a good cluster because its top words such as infinite, universe, finite, space, amount, number, and infinity clearly match the main discussion about whether the universe is infinite and what infinity means. Cluster 0 was more confusing because it was much larger than the others and included broader words such as reply, like, video, know, one, brain, and time, which suggests that it mixed several general reactions and side discussions instead of one clear topic.\n\n")

best_eps = 40

dbscan_final = DBSCAN(eps=best_eps, min_samples=5)
final_labels = dbscan_final.fit_predict(X_scaled)

df['dbscan_label'] = final_labels

n_clusters_final = len(set(final_labels)) - (1 if -1 in final_labels else 0)
n_noise_final = list(final_labels).count(-1)

print(f"\nFinal DBSCAN results with eps = {best_eps}")
print(f"Number of clusters: {n_clusters_final}")
print(f"Number of noise points: {n_noise_final}")

print("\nDBSCAN label counts:")
print(pd.Series(final_labels).value_counts().sort_index())

print("\nSample comments from each DBSCAN label:")
for label in sorted(set(final_labels)):
    print(f"\nDBSCAN label {label} sample comments:")
    sample_comments = df[df['dbscan_label'] == label]['comment_text'].head(5).tolist()
    for j, comment in enumerate(sample_comments, 1):
        print(f"{j}. {comment}")

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='viridis', alpha=0.6)
plt.title(f'DBSCAN Visualization (eps={best_eps}, -1 is Noise)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.grid(True)

dbscan_plot_path = os.path.join(OUTPUT_DIR, "dbscan_plot.png")
plt.savefig(dbscan_plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nDBSCAN plot saved as: {dbscan_plot_path}")

print("\nCritical Prompt #3:")
print("K-Means gave me a more useful understanding of the comments because it assigned every comment to a cluster and made it easier to identify broad discussion themes in the video. In contrast, DBSCAN marked many comments as noise, which suggests that the comments were too spread out in TF-IDF space to form dense groups. Based on these results, K-Means was more useful for my dataset. DBSCAN would be superior in a situation where the comments include many outliers, spam, or small irregular groups that do not fit well into a fixed number of clusters.")

print("\nCritical Prompt #4:")
print("My dataset has 739 comments, which is large enough for clustering and gives the algorithms enough repeated patterns to analyze. However, the topic and language style of YouTube comments still affected the results because the comments vary in length, wording, and focus. That made K-Means more effective for identifying broad themes, while DBSCAN struggled because many comments did not form dense neighborhoods. If an airline or marketing company used these techniques on customer feedback, the most important lesson from this lab is that clustering can reveal useful patterns, but the results must always be interpreted carefully and checked against the actual meaning of the text.")

print("\nFinal Summary Paragraph:")
print("Overall, K-Means and DBSCAN were not equally effective on my YouTube comment dataset. K-Means was more useful because it grouped all comments into clusters and helped reveal broad themes in the discussion, even though some clusters may still have been mixed or overlapping. DBSCAN was less effective for this dataset because it treated many comments as noise and produced very limited clustering structure. This shows that the usefulness of a clustering algorithm depends heavily on the structure and style of the text data being analyzed.")

output_csv_path = os.path.join(OUTPUT_DIR, "lab5_output_with_clusters.csv")
df.to_csv(output_csv_path, index=False)

print(f"\nClustered CSV saved as: {output_csv_path}")
print("\n2nd part code complete.")