import numpy as np
import csv


def initialize_membership_matrix(n_samples, n_clusters):
    print("Initializing membership matrix...")
    membership = np.random.rand(n_samples, n_clusters)
    membership = membership / np.sum(membership, axis=1, keepdims=True)
    return membership


def calculate_cluster_centers(membership, data, m):
    print("Calculating cluster centers...")
    num = (membership ** m).T @ data
    den = np.sum(membership ** m, axis=0).reshape(-1, 1)
    centers = num / den
    return centers


def update_membership_matrix(membership, centers, data, m):
    print("Updating membership matrix...")
    n_samples = data.shape[0]
    n_clusters = centers.shape[0]
    new_membership = np.zeros((n_samples, n_clusters))

    for i in range(n_samples):
        for j in range(n_clusters):
            numerator = np.linalg.norm(data[i] - centers[j]) + 1e-10
            denom_sum = 0
            for k in range(n_clusters):
                denominator = np.linalg.norm(data[i] - centers[k]) + 1e-10
                denom_sum += (numerator / denominator) ** (2 / (m - 1))
            new_membership[i, j] = 1.0 / denom_sum

    return new_membership


def fuzzy_c_means(data, n_clusters=10, m=2.0, max_iter=100, error=1e-5):
    n_samples = data.shape[0]

    membership = initialize_membership_matrix(n_samples, n_clusters)

    for iteration in range(max_iter):
        print(f"\nIteration {iteration + 1}")
        centers = calculate_cluster_centers(membership, data, m)
        new_membership = update_membership_matrix(membership, centers, data, m)

        # Check convergence
        diff = np.linalg.norm(new_membership - membership)
        print(f"Difference: {diff:.6f}")

        if diff < error:
            print(f"Converged after {iteration + 1} iterations.\n")
            break

        membership = new_membership

    return membership


def save_membership_matrix_to_csv(membership_matrix, filename='membership_matrix.csv'):
    print(f"Saving membership matrix to {filename}...")
    n_clusters = membership_matrix.shape[1]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        header = [f'Cluster_{i}' for i in range(n_clusters)]
        writer.writerow(header)

        # Write rows
        for row in membership_matrix:
            writer.writerow(row)


# ----------------------------------------------
# Main script

print("Loading TD-IDF matrix...")
tdidf = np.load('pcaNormalizedTfidf.npy')  # Make sure the file is named 'tdidf.npy'

n_clusters = 10

# Run Fuzzy C-Means
membership_matrix = fuzzy_c_means(tdidf, n_clusters=n_clusters)

# Final print
print("\nFinal relationship of each book with each cluster:")
for i, memberships in enumerate(membership_matrix):
    memberships_str = ", ".join([f"Cluster {j}: {membership:.4f}" for j, membership in enumerate(memberships)])
    print(f"Book {i}: {memberships_str}")

# Save membership matrix as .npy
np.save('membership_matrix.npy', membership_matrix)
print("\nMembership matrix saved as 'membership_matrix.npy'.")

# Save membership matrix as .csv
save_membership_matrix_to_csv(membership_matrix, filename='membership_matrix.csv')
print("Membership matrix also saved as 'membership_matrix.csv'.")

# Check: Sum of each book's memberships should be 1
sum_check = np.sum(membership_matrix, axis=1)
print("\nSum of memberships for each book (should be all 1s):")
print(sum_check)
