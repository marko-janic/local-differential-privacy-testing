import pandas as pd

# Local imports
from algorithms.cms import test_count_mean_sketch
from algorithms.laplace import test_laplace
from algorithms.cms import test_loaded_matrix
from algorithms.cms import test_cms_histograms

def main():
    features = pd.read_csv("dataset/features.csv")
    #print(features.head(0))

    # Test Laplace
    #test_laplace(features["Age"], "Age")

    # Test Count min Sketch
    #test_loaded_matrix()
    test_count_mean_sketch(features["GenHlth"], 3)
    #test_cms_histograms(features["GenHlth"])


if __name__ == "__main__":
    main()
