from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

def load_split_data(test_size=0.2):
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(X, y, test_size=test_size, random_state=42)