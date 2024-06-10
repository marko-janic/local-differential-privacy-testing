import pandas as pd
import matplotlib.pyplot as plt


def main():
    features = pd.read_csv("dataset/features.csv")

    #print(features)

    plt.hist(features["Age"], bins=10)
    plt.show()


if __name__ == "__main__":
    main()
