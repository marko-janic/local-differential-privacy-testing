import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Load the dataset
features_path = 'dataset/features.csv'
data = pd.read_csv(features_path, index_col=0)

# List of possible categories for GenHlth
categories = [1, 2, 3, 4, 5]


# Function to unary encode the "GenHlth" value
def unary_encode(value, categories):
    """Unary encode a value given the list of possible categories."""
    vector = [0] * len(categories)
    index = categories.index(value)
    vector[index] = 1
    return vector


# Function to apply randomized response to a unary encoded vector
def unary_randomized_response(encoded_vector, epsilon):
    """Apply randomized response to each bit in the unary encoded vector."""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    noised_vector = []
    flip_count = 0
    for bit in encoded_vector:
        if np.random.rand() < p:
            noised_vector.append(bit)
        else:
            noised_vector.append(1 - bit)
            flip_count += 1
    return noised_vector, flip_count


# Function to denoise the noised unary encoded vectors
def denoise_unary_response(noised_vectors, epsilon, categories):
    """Denoise the noised unary encoded vectors to estimate true counts."""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    total_counts = np.zeros(len(categories))
    for vector in noised_vectors:
        for i, bit in enumerate(vector):
            total_counts[i] += bit
    denoised_counts = (total_counts - len(noised_vectors) * (1 - p)) / (2 * p - 1)
    return denoised_counts


# Analyze the data for different epsilon values
epsilons = [0.1, 0.5, 1.0, 1.5, 2.0, 5, 10]
relative_errors = []
runtimes = []
average_flips = []

for epsilon in epsilons:
    total_flips = 0
    start_time = time.time()

    noised_vectors = []
    for value in data['GenHlth']:
        encoded_vector = unary_encode(value, categories)
        noised_vector, flip_count = unary_randomized_response(encoded_vector, epsilon)
        noised_vectors.append(noised_vector)
        total_flips += flip_count

    denoised_counts = denoise_unary_response(noised_vectors, epsilon, categories)

    # Calculate the original counts
    original_counts = data['GenHlth'].value_counts().sort_index().values

    # Calculate the relative error
    relative_error = np.abs(denoised_counts - original_counts) / original_counts
    mean_relative_error = np.mean(relative_error)
    relative_errors.append(mean_relative_error)

    # Record the runtime
    runtime = time.time() - start_time
    runtimes.append(runtime)

    # Calculate the average number of flips
    avg_flips = total_flips / len(data)
    average_flips.append(avg_flips)

    print(f"Epsilon: {epsilon}")
    print(f"Original counts: {original_counts}")
    print(f"Denoised counts: {denoised_counts}")
    print(f"Mean relative error: {mean_relative_error}")
    print(f"Runtime: {runtime}")
    print(f"Average flips: {avg_flips}\n")

# Plot the results
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(epsilons, relative_errors, marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Mean Relative Error')
plt.title('Mean Relative Error vs. Epsilon')

plt.subplot(1, 3, 2)
plt.plot(epsilons, runtimes, marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime vs. Epsilon')

plt.subplot(1, 3, 3)
plt.plot(epsilons, average_flips, marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Average Flips')
plt.title('Average Flips vs. Epsilon')

plt.tight_layout()
plt.show()
