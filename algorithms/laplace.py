import numpy as np
import time
import matplotlib.pyplot as plt


def test_laplace(test_column, column_name):
    # the different epsilon values:
    epsilons = np.arange(0.1, 3.1, 0.1)
    times = []
    errors = []

    for epsilon in epsilons:
        start = time.time()
        laplace_column = apply_laplace_mechanism(test_column, epsilon)
        end = time.time()
        times.append(end-start)

        # calculate errors between the columns
        errors.append(((test_column - laplace_column)**2).mean())

    plt.title("Laplace Method: Epsilon vs Computation Time for Column: " + column_name)
    plt.plot(epsilons, times)
    plt.xlabel("Epsilon Values")
    plt.ylabel("Time in seconds")
    plt.savefig("figures/laplace_eps_vs_comp_" + column_name + ".png")
    plt.show()

    plt.title("Laplace Mechanism: MSE vs Epsilon for Column: " + column_name)
    plt.plot(epsilons, errors)
    plt.xlabel("Epsilon Values")
    plt.ylabel("Mean Squared Error")
    plt.savefig("figures/laplace_eps_vs_mse_" + column_name + ".png")
    plt.show()

    alpha = 0.5
    epsilon = 0.8
    laplace_column = apply_laplace_mechanism(test_column, epsilon)
    plt.title("Laplace Mechanism: Histogram Comparison for Column: " + column_name)
    plt.hist(test_column, bins=30, color="b", alpha=alpha, label="True Values")
    plt.hist(laplace_column, bins=30, color="r", alpha=alpha, label="Laplace Values")
    plt.legend()
    plt.text(.01, .99, "Epsilon = " + str(epsilon), ha='left', va='top', transform=plt.gca().transAxes)
    plt.savefig("figures/laplace_hist_" + column_name + ".png")
    plt.show()


def apply_laplace_mechanism(data_column, epsilon):
    laplace_data_column = data_column.copy()
    noise = np.random.laplace(loc=0, scale=1/epsilon, size=laplace_data_column.shape)
    laplace_data_column = laplace_data_column + noise
    return laplace_data_column
