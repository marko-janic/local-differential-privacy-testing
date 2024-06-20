import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Function to apply randomized response
def randomized_response(value, epsilon):
    """
    Apply randomized response to a binary value.
    Args:
    value (int): The original binary value (0 or 1).
    epsilon (float): Privacy budget parameter.

    Returns:
    int: The noised value (0 or 1).
    """
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    if np.random.rand() < p:
        return value
    else:
        return 1 - value

# Function to calculate original, noised, and denoised counts
def analyze_data(data, epsilon):
    # Calculate the count of entries with HvyAlcoholConsump == 1 in the original dataset
    original_count = data['HvyAlcoholConsump'].sum()

    # Apply randomized response to the HvyAlcoholConsump column
    data['HvyAlcoholConsump_noised'] = data['HvyAlcoholConsump'].apply(lambda x: randomized_response(x, epsilon))

    # Calculate the count of entries with HvyAlcoholConsump_noised == 1 in the noised dataset
    noised_count = data['HvyAlcoholConsump_noised'].sum()

    # Denoise the count
    total_count = len(data)
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    denoised_proportion = (noised_count / total_count - (1 - p)) / (2 * p - 1)
    denoised_count = denoised_proportion * total_count

    return original_count, noised_count, denoised_count

# Load the features dataset
features_path = 'dataset/features.csv'
data = pd.read_csv(features_path, index_col=0)

# Define different epsilon values to test
epsilons = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]

# Test for different epsilon values
results = []

for epsilon in epsilons:
    original_counts = []
    noised_counts = []
    denoised_counts = []
    runtimes = []

    for _ in range(10):  # Run 10 times for each epsilon
        subset_data = data.copy()  # Make a copy to avoid SettingWithCopyWarning

        # Measure runtime
        start_time = time.time()
        original_count, noised_count, denoised_count = analyze_data(subset_data, epsilon=epsilon)
        end_time = time.time()

        runtime = end_time - start_time
        original_counts.append(original_count)
        noised_counts.append(noised_count)
        denoised_counts.append(denoised_count)
        runtimes.append(runtime)

    # Calculate means
    mean_original_count = np.mean(original_counts)
    mean_noised_count = np.mean(noised_counts)
    mean_denoised_count = np.mean(denoised_counts)
    mean_runtime = np.mean(runtimes)
    relative_error = abs(mean_original_count - mean_denoised_count) / mean_original_count if mean_original_count != 0 else np.nan

    results.append((epsilon, mean_original_count, mean_noised_count, mean_denoised_count, relative_error, mean_runtime))

# Print the results and calculate relative error
for epsilon, mean_original_count, mean_noised_count, mean_denoised_count, relative_error, mean_runtime in results:
    print(f"Epsilon: {epsilon}")
    print(f"Mean original count of HvyAlcoholConsump == 1: {mean_original_count}")
    print(f"Mean noised count of HvyAlcoholConsump == 1: {mean_noised_count}")
    print(f"Mean denoised count of HvyAlcoholConsump == 1: {mean_denoised_count}")
    print(f"Relative error: {relative_error}")
    print(f"Mean runtime: {mean_runtime}")
    print("\n")

# Extract epsilons, relative errors, and runtimes for plotting
epsilons = [result[0] for result in results]
relative_errors = [result[4] for result in results]
runtimes = [result[5] for result in results]

# Plot epsilon against relative error
plt.figure(figsize=(10, 6))
plt.plot(epsilons, relative_errors, marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Relative Error')
plt.title('Relative Error vs Epsilon')
# plt.xscale('log')
plt.grid(True)
plt.show()

# Plot epsilon against runtime
plt.figure(figsize=(10, 6))
plt.plot(epsilons, runtimes, marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime vs Epsilon')
# plt.xscale('log')
plt.grid(True)
plt.show()
