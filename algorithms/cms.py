import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import murmurhash
import seaborn as sns


def test_cms_histograms(data_column):
    """
    This function was mostly for debugging. It plots two histograms one with the real counts of our data
    and the other with the estimated counts of our data.

    Args:
         data_column: the data to do the histogram on
    """
    data_column_counts = data_column.value_counts().sort_index(ascending=True)
    data_values = np.sort(data_column.unique())

    hash_val = 10
    row = 50
    epsilon = 0.1
    n = len(data_column)

    print("Computing Sketch Matrix")
    sketch = apply_count_mean_sketch(data_column, hash_val, row, epsilon)

    print("Estimating Counts")
    estimated_counts = []
    for value in data_values:
        estimated_counts.append(sketch.estimate_data_element(value))

    bar_width = 0.35
    plt.bar(range(len(data_column_counts)), data_column_counts, bar_width, label="Real Counts")
    plt.bar([p + bar_width for p in range(len(estimated_counts))], estimated_counts, bar_width, label="Estimated Counts")
    plt.legend()
    plt.xlabel("General Health")
    plt.xticks(rotation=0)
    plt.ylabel("Number of People")
    plt.title("Histogram CSM Estimated vs Real Counts")
    plt.text(.01, .99, "Epsilon = " + str(epsilon) + "\nk = " + str(hash_val) + "\nm = " + str(row),
             ha='left', va='top', transform=plt.gca().transAxes)
    plt.savefig("figures/cms_histogram_hash="+str(hash_val)+"_rows="+str(row)+"_epsilon="+str(epsilon)+".png")
    plt.show()

    normalized_estimated_counts = [element/n for element in estimated_counts]
    rmse = np.sqrt(((data_column_counts/n - normalized_estimated_counts)**2).mean())
    print("RMSE: ", rmse)


def test_loaded_matrix():
    visualization_array = np.load("figures/cms_heatmap_epsilon=0.5_rows10to210_hashes1to21.npy")
    hashes_min = 1
    hashes_max = 21
    rows_min = 10
    rows_max = 210
    epsilon = 0.5
    extent = [rows_min, rows_max, hashes_min, hashes_max]

    print(visualization_array)
    plt.imshow(visualization_array, cmap="hot", interpolation="nearest", extent=extent, aspect="auto")
    plt.colorbar(label="Normalized RMSE between Real values and Estimations")
    plt.title("Count Mean Sketch Heatmap, Epsilon = " + str(epsilon))
    plt.xlabel("Number of Rows")
    plt.ylabel("Number of Hashes")
    plt.show()


def test_count_mean_sketch(data_column, epsilon):
    n = len(data_column)
    hashes_min = 1
    hashes_max = 1000
    hashes_step = 99
    rows_min = 5
    rows_max = 1000
    rows_step = 99
    extent = [rows_min, rows_max, hashes_min, hashes_max]
    hashes = np.arange(hashes_min, hashes_max, hashes_step)
    rows = np.arange(rows_min, rows_max, rows_step)
    visualization_array = np.zeros((len(hashes), len(rows)))
    visualization_array_2 = np.zeros((len(hashes), len(rows)))
    print(visualization_array.shape)

    data_column_counts = data_column.value_counts().sort_index(ascending=True).to_frame()
    float_dcc = data_column_counts.astype(float)
    float_dcc = float_dcc.iloc[:, 0]
    normalized_float_dcc = float_dcc / n

    data_values = np.sort(data_column.unique())
    i = 0
    j = 0
    for hash_val in hashes:
        for row in rows:

            sketch = apply_count_mean_sketch(data_column, hash_val, row, epsilon)
            estimated_counts = []
            for value in data_values:
                estimated_counts.append(sketch.estimate_data_element(value))
            estimated_counts = np.array(estimated_counts)
            normalized_estimated_counts = [element/n for element in estimated_counts]

            rmse = np.sqrt(((float_dcc - estimated_counts) ** 2).mean())
            normalized_rmse = np.sqrt(((normalized_float_dcc - normalized_estimated_counts)**2).mean())

            print(i, j)
            visualization_array[i, j] = normalized_rmse
            visualization_array_2[i, j] = rmse

            j += 1
        j = 0
        i += 1

    np.save("figures/cms_heatmap_epsilon=" + str(epsilon) + "_rows" + str(rows_min) + "to"+str(rows_max) + "_hashes"+str(hashes_min) +
            "to" + str(hashes_max), visualization_array)
    np.save(
        "figures/cms_heatmap_epsilon=" + str(epsilon) + "_rows" + str(rows_min) + "to" + str(rows_max) + "_hashes" + str(hashes_min) +
        "to" + str(hashes_max) + "_not_normalized", visualization_array_2)
    plt.imshow(visualization_array, cmap="hot", interpolation="nearest", extent=extent, aspect="auto")
    plt.colorbar(label="Normalized RMSE between Real values and Estimations")
    plt.title("Count Mean Sketch Heatmap, Epsilon = " + str(epsilon))
    plt.xlabel("Number of Rows")
    plt.ylabel("Number of Hashes")
    plt.savefig("figures/cms_heatmap_epsilon=" + str(epsilon) + "_rows" + str(rows_min) + "to"+str(rows_max) + "_hashes"+str(hashes_min)
                + "to" + str(hashes_max) + ".png")
    plt.show()


def apply_count_mean_sketch(data_column, n_hashes, n_rows, epsilon):
    sketch = CountMeanSketch(k=n_hashes, m=n_rows, epsilon=epsilon)

    v_tilde_list = []
    j_list = []
    for data in data_column:
        v_tilde, j = sketch.client(data)
        v_tilde_list.append(v_tilde)
        j_list.append(j)
    sketch.update_sketch_matrix(v_tilde_list, j_list)

    return sketch


class CountMeanSketch:
    """
    Class implementing the Count Mean Sketch (CMS) algorithm for differentially private estimation.

    Attributes:
        k (int): Number of hash functions.
        m (int): Size of the sketch matrix.
        epsilon (float): Privacy parameter.
        H (list): List of k hash functions.
        M (list): Sketch matrix of size k x m.

    Methods:
        update(self, data): Updates the sketch matrix with a new data point.
        estimate(self, data): Estimates the count for a specific data point.
    """

    def __init__(self, k, m, epsilon):
        """
        Initializes the CMS object.

        Args:
            k (int): Number of hash functions.
            m (int): Size of the sketch matrix.
            epsilon (float): Privacy parameter.
        """

        self.k = k
        self.m = m
        self.epsilon = epsilon
        self.n = 0
        self.c_epsilon = (np.exp(self.epsilon/2) + 1)/(np.exp(self.epsilon/2) - 1)

        self.H = self.select_k_hash_functions()  # Initialize k hash functions
        #self.M = [[0] * m for _ in range(k)]  # Initialize sketch matrix
        self.M = np.zeros((k, m))

    def select_k_hash_functions(self):
        """
        Selects k universal hash functions from the murmurhash library.

        Returns:
            A dictionary of k hash functions, where the key is the seed for the hash function.
        """
        # Initialize an empty dictionary to store hash functions
        hash_functions = {}

        # Create k universal hash functions using the generated seeds
        for j in range(self.k):
            hash_functions[j] = lambda x: murmurhash.hash(bytes(x), j) % self.m

        return hash_functions

    def client(self, d):
        j = random.randrange(self.k)

        # Encoding vector where its -1 everywhere except at index h_j(d) where its 1
        v = np.ones(self.m) * -1
        v[self.H[j](d)] = 1

        v_tilde = v
        flip_probability = 1/(np.exp(self.epsilon/2) + 1)
        # Now flip each bit with probability 1 / (e^(epsilon/2) + 1)
        v_tilde = np.where(np.random.rand(len(v_tilde)) < flip_probability, -v_tilde, v_tilde)

        # Increase the size of our hypothetical dataset on the server
        self.n += 1
        return v_tilde, j

    def update_sketch_matrix(self, v_tilde_list, j_list):
        """
        Update sketch matrix given list of data (v_tilde, j)^n from the client

        :return:
        """
        for i in range(len(v_tilde_list)):
            v_tilde = v_tilde_list[i]
            j = j_list[i]

            x_tilde = self.k * ((self.c_epsilon / 2) * v_tilde + 0.5)

            self.M[j] = self.M[j] + x_tilde

    def estimate_data_element(self, d):
        inner_sum = 0
        for l_index in range(self.k):
            inner_sum += self.M[l_index][self.H[l_index](d)]

        return (self.m/(self.m - 1))*((1/self.k) * inner_sum - (self.n/self.m))
