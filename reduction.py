import numpy as np

# PARAMETERS
INPUT_FILE = 'tdidf.npy'       # Your original tf-idf file
OUTPUT_FILE = 'pcaNormalizedTfidf.npy'   # Output after PCA + Normalization
N_COMPONENTS = 100             # How many dimensions you want after PCA

# STEP 1: Load tf-idf matrix
print("Loading TF-IDF matrix...")
X = np.load(INPUT_FILE)
print(f"Original shape: {X.shape}")

# STEP 2: Center the data (important for PCA)
print("Centering the data...")
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# STEP 3: Calculate Covariance Matrix
print("Calculating covariance matrix...")
cov_matrix = np.cov(X_centered, rowvar=False)

# STEP 4: Compute eigenvalues and eigenvectors
print("Calculating eigenvectors and eigenvalues...")
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# STEP 5: Sort eigenvectors by descending eigenvalues
print("Sorting eigenvectors...")
sorted_indices = np.argsort(eigenvalues)[::-1]
top_eigenvectors = eigenvectors[:, sorted_indices[:N_COMPONENTS]]

# STEP 6: Project the data
print(f"Projecting data to {N_COMPONENTS} dimensions...")
X_reduced = np.dot(X_centered, top_eigenvectors)

# STEP 7: Normalize the data (Standardization: zero mean, unit variance)
print("Normalizing the reduced data...")

# Manually normalize by subtracting the mean and dividing by the standard deviation
X_mean_reduced = np.mean(X_reduced, axis=0)
X_std_reduced = np.std(X_reduced, axis=0)

X_normalized = (X_reduced - X_mean_reduced) / X_std_reduced

# STEP 8: Save the normalized data
print(f"Saving normalized PCA matrix to '{OUTPUT_FILE}'...")
np.save(OUTPUT_FILE, X_normalized)

print("PCA + Normalization completed successfully!")
