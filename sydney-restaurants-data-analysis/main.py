from predictive_modelling_regression import regression_model
from predictive_modelling_classification import classifier_model

import pandas as pd

import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


def main():

    # print the output heading model results centered
    print("\n")
    print("-----------------------------------------------------------------------")
    print(f"{'Model Results':^70}")
    print("-----------------------------------------------------------------------")
    print("\n")
    regression_model()
    print("\n")
    classifier_model()
    print("\n")
    print("-----------------------------------------------------------------------")
    print("\n")


if __name__ == "__main__":
    main()
