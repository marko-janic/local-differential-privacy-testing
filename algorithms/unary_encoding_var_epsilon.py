import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

features_path = '../dataset/features.csv'
data = pd.read_csv(features_path, index_col=0)

# Categories of the two columns
genhlth_categories = [1, 2, 3, 4, 5]
age_categories = list(range(1, 14))


def unary_encode(value, categories):
    """Unary encode a value given the list of possible categories."""
    vector = [0] * len(categories)
    index = categories.index(value)
    vector[index] = 1
    return vector


# apply randomized response to a unary encoded vector
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


# denoise the noised unary encoded vectors
def denoise_unary_response(noised_vectors, epsilon, categories):
    """Denoise the noised unary encoded vectors to estimate true counts."""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    total_counts = np.zeros(len(categories))
    for vector in noised_vectors:
        for i, bit in enumerate(vector):
            total_counts[i] += bit
    denoised_counts = (total_counts - len(noised_vectors) * (1 - p)) / (2 * p - 1)
    return denoised_counts


# Harcdoded epsilons
epsilons = [0.1, 0.5, 1.0, 1.5, 2.0, 5, 10]

relative_errors_genhlth = []
relative_errors_age = []
runtimes_genhlth = []
runtimes_age = []
average_flips_genhlth = []
average_flips_age = []

denoised_counts_epsilon_1_genhlth = []
denoised_counts_epsilon_1_age = []

for epsilon in epsilons:
    total_flips_genhlth = 0
    total_flips_age = 0
    start_time_genhlth = time.time()
    start_time_age = time.time()

    noised_vectors_genhlth = []
    noised_vectors_age = []
    for genhlth_value, age_value in zip(data['GenHlth'], data['Age']):
        encoded_vector_genhlth = unary_encode(genhlth_value, genhlth_categories)
        encoded_vector_age = unary_encode(age_value, age_categories)

        noised_vector_genhlth, flip_count_genhlth = unary_randomized_response(encoded_vector_genhlth, epsilon)
        noised_vector_age, flip_count_age = unary_randomized_response(encoded_vector_age, epsilon)

        noised_vectors_genhlth.append(noised_vector_genhlth)
        noised_vectors_age.append(noised_vector_age)

        total_flips_genhlth += flip_count_genhlth
        total_flips_age += flip_count_age

    denoised_counts_genhlth = denoise_unary_response(noised_vectors_genhlth, epsilon, genhlth_categories)
    denoised_counts_age = denoise_unary_response(noised_vectors_age, epsilon, age_categories)

    # For histograms with given epsilon
    if epsilon == 1:
        denoised_counts_epsilon_1_genhlth = denoised_counts_genhlth
        denoised_counts_epsilon_1_age = denoised_counts_age

    # original counts
    original_counts_genhlth = data['GenHlth'].value_counts().sort_index().values
    original_counts_age = data['Age'].value_counts().sort_index().values

    # Calculate relative error based on input size
    relative_error_genhlth = np.abs(denoised_counts_genhlth - original_counts_genhlth) / len(data)
    mean_relative_error_genhlth = np.mean(relative_error_genhlth)
    relative_errors_genhlth.append(mean_relative_error_genhlth)

    relative_error_age = np.abs(denoised_counts_age - original_counts_age) / len(data)
    mean_relative_error_age = np.mean(relative_error_age)
    relative_errors_age.append(mean_relative_error_age)

    # CAlculate the runtime
    runtime_genhlth = time.time() - start_time_genhlth
    runtimes_genhlth.append(runtime_genhlth)

    runtime_age = time.time() - start_time_age
    runtimes_age.append(runtime_age)

    # Calculate the average number of bit flips in vector
    avg_flips_genhlth = total_flips_genhlth / len(data)
    average_flips_genhlth.append(avg_flips_genhlth)

    avg_flips_age = total_flips_age / len(data)
    average_flips_age.append(avg_flips_age)

    print(f"Epsilon: {epsilon}")
    print(f"Original counts (GenHlth): {original_counts_genhlth}")
    print(f"Denoised counts (GenHlth): {denoised_counts_genhlth}")
    print(f"Mean relative error (GenHlth): {mean_relative_error_genhlth}")
    print(f"Original counts (Age): {original_counts_age}")
    print(f"Denoised counts (Age): {denoised_counts_age}")
    print(f"Mean relative error (Age): {mean_relative_error_age}")
    print(f"Runtime (GenHlth): {runtime_genhlth}")
    print(f"Runtime (Age): {runtime_age}")
    print(f"Average flips (GenHlth): {avg_flips_genhlth}")
    print(f"Average flips (Age): {avg_flips_age}\n")

# Plot
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(epsilons, relative_errors_genhlth, marker='o', label='GenHlth')
plt.plot(epsilons, relative_errors_age, marker='o', label='Age')
plt.xlabel('Epsilon')
plt.ylabel('Mean Relative Error')
plt.title('Mean Relative Error vs. Epsilon')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epsilons, average_flips_genhlth, marker='o', label='GenHlth')
plt.plot(epsilons, average_flips_age, marker='o', label='Age')
plt.xlabel('Epsilon')
plt.ylabel('Average Flips')
plt.title('Average Flips vs. Epsilon')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epsilons, runtimes_genhlth, marker='o', label='GenHlth')
plt.plot(epsilons, runtimes_age, marker='o', label='Age')
plt.xlabel('Epsilon')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime vs. Epsilon')
plt.legend()

plt.tight_layout()
plt.show()

# Plot actual distribution(original) vs denoised distribution for epsilon = 1 (only GenHlth)
plt.figure(figsize=(10, 6))
bar_width = 0.35

x = np.arange(len(genhlth_categories))

plt.bar(x, original_counts_genhlth, bar_width, label='Original', alpha=0.6, color='b')
plt.bar(x + bar_width, denoised_counts_epsilon_1_genhlth, bar_width, label='Denoised (ε=1)', alpha=0.6, color='r')

plt.xlabel('GenHlth Categories')
plt.ylabel('Counts')
plt.title('Original vs Denoised Counts (GenHlth, ε=1)')
plt.xticks(x + bar_width / 2, genhlth_categories)
plt.legend()
plt.tight_layout()
plt.show()
