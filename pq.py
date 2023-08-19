import numpy as np
from sklearn.cluster import KMeans

# parse the vectors
with open('vectors2.txt', 'r') as f:
    lines = f.readlines()
vectors = [list(map(float, line.strip().split(','))) for line in lines]

# constants
M = 4
K = 256
num_vectors = len(vectors)
vector_dimension = len(vectors[0]) if num_vectors > 0 else 0
sub_vector_length = vector_dimension // M

# Split vectors into sub-vectors
sub_vectors = [np.array([v[i:i+sub_vector_length] for v in vectors]) for i in range(0, vector_dimension, sub_vector_length)]

# compute the centroids using K-means clustering
centroids = []
for i, sub_vector_set in enumerate(sub_vectors):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(sub_vector_set)
    sorted_centroids = sorted(kmeans.cluster_centers_, key=lambda x: tuple(x))
    centroids.append(sorted_centroids)

    # Print the first 3 components of each centroid in the current codebook
    print(f"Codebook {i + 1}:")
    for centroid_idx, centroid in enumerate(sorted_centroids):
        print(f"Centroid {centroid_idx + 1}: {centroid[:3]}")
        if centroid_idx >= 1:
            break
    print("\n")

# Convert centroids to numpy arrays for easier manipulation
centroids = [np.array(centroid) for centroid in centroids]

# Quantize the vectors using the codebooks
quantized_vectors = []
for v in vectors:
    quantized_vector = []
    sub_v = [v[i:i+sub_vector_length] for i in range(0, vector_dimension, sub_vector_length)]
    for i, s in enumerate(sub_v):
        centroid_distances = np.linalg.norm(centroids[i] - s, axis=1)
        nearest_centroid_idx = np.argmin(centroid_distances)
        quantized_vector.append(nearest_centroid_idx)
    quantized_vectors.append(quantized_vector)

# print it!
quantized_vectors = np.array(quantized_vectors)
print(quantized_vectors)
