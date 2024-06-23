import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


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


# We first retrieve the original counts, then we perturbate the bits for a noised count,
# and lastly we denoise the result to get our statistic for a comparing it with the original counts
def analyze_data(data, epsilon):
    original_count = data['HvyAlcoholConsump'].sum()
    data['HvyAlcoholConsump_noised'] = data['HvyAlcoholConsump'].apply(lambda x: randomized_response(x, epsilon))

    noised_count = data['HvyAlcoholConsump_noised'].sum()

    total_count = len(data)
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    denoised_proportion = (noised_count / total_count - (1 - p)) / (2 * p - 1)
    denoised_count = denoised_proportion * total_count

    return original_count, noised_count, denoised_count


# Load the features dataset
features_path = '../dataset/features.csv'
data = pd.read_csv(features_path, index_col=0)

# Different hardcoded epsilon values
epsilons = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
results = []

for epsilon in epsilons:
    original_counts = []
    noised_counts = []
    denoised_counts = []
    runtimes = []
    flip_counts = []

    for _ in range(10):  # Run 10 times for each epsilon for aggregation
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

        # Calculate number of bit flips
        flip_count = subset_data['HvyAlcoholConsump'].ne(subset_data['HvyAlcoholConsump_noised']).sum()
        flip_counts.append(flip_count)

    # Calculate means (noised_counts is not really necessary)
    mean_original_count = np.mean(original_counts)
    mean_noised_count = np.mean(noised_counts)
    mean_denoised_count = np.mean(denoised_counts)
    mean_runtime = np.mean(runtimes)
    mean_flip_count = np.mean(flip_counts)

    # Calculate relative error based on input size
    relative_error = abs(mean_denoised_count - mean_original_count) / len(data)

    results.append((epsilon, mean_original_count, mean_noised_count, mean_denoised_count, relative_error, mean_runtime,
                    mean_flip_count))

# Print the results and calculate relative error
for epsilon, mean_original_count, mean_noised_count, mean_denoised_count, relative_error, mean_runtime, mean_flip_count in results:
    print(f"Epsilon: {epsilon}")
    print(f"Mean original count of HvyAlcoholConsump == 1: {mean_original_count}")
    print(f"Mean noised count of HvyAlcoholConsump == 1: {mean_noised_count}")
    print(f"Mean denoised count of HvyAlcoholConsump == 1: {mean_denoised_count}")
    print(f"Relative error: {relative_error}")
    print(f"Mean runtime: {mean_runtime}")
    print(f"Mean number of bit flips: {mean_flip_count}")
    print("\n")

# Extract epsilons, relative errors, runtimes, and flip counts for plots
epsilons = [result[0] for result in results]
relative_errors = [result[4] for result in results]
runtimes = [result[5] for result in results]
flip_counts = [result[6] for result in results]

bit_flip_probs = [1 - (np.exp(epsilon) / (1 + np.exp(epsilon))) for epsilon in epsilons]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(epsilons, relative_errors, marker='o', label='Relative Error')
ax1.set_xlabel('Epsilon')
ax1.set_ylabel('Relative Error')
ax1.set_title('Relative Error and Bit Flips vs Epsilon')
ax1.grid(True)

ax2.plot(epsilons, runtimes, marker='o', color='r', label='Runtime (seconds)')
ax2.set_xlabel('Epsilon')
ax2.set_ylabel('Runtime (seconds)')
ax2.set_title('Runtime vs Epsilon')
ax2.grid(True)

# Add number of bit flips to first plot
ax1b = ax1.twinx()
ax1b.plot(epsilons, flip_counts, marker='o', color='g', label='Number of Bit Flips')
ax1b.set_ylabel('Number of Bit Flips')

ax1.legend(loc='upper right')
ax2.legend(loc='upper left')
ax1b.legend(loc='upper center')

plt.tight_layout()
plt.show()

# Output the table of epsilon values and probabilities of bit flips
flip_prob_df = pd.DataFrame({
    'Epsilon': epsilons,
    'Probability of Bit Flip': bit_flip_probs
})
print(flip_prob_df)
