from ucimlrepo import fetch_ucirepo


def import_dataset():
    # fetch dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

    # data (as pandas dataframes)
    features = cdc_diabetes_health_indicators.data.features
    targets = cdc_diabetes_health_indicators.data.targets

    features.to_csv("dataset/features.csv")
    targets.to_csv("dataset/targets.csv")


if __name__ == "__main__":
    import_dataset()
