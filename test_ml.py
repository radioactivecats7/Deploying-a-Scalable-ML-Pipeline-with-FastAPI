import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
# TODO: implement the first test. Change the function name and input as needed

@pytest.fixture
def sample_data_test():
    """
    # Sample Data for the tests
    """
    return pd.Dataframe({
        "age":[25, 45, 38, 29],
        "workclass": ["Private", "Self-emp-not-inc", "Private", "State-gov"],
        "fnlgt": [226802, 89814, 336951, 123011],
        "education": ["Bachelors", "Masters", "HS-grad", "Assoc-acdm"],
        "education-num": [13, 14, 9, 12],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Never-married"],
        "occupation": ["Tech-support", "Exec-managerial", "Sales", "Adm-clerical"],
        "relationship": ["Not-in-family", "Husband", "Unmarried", "Own-child"],
        "race": ["White", "White", "Black", "Asian-Pac-Islander"],
        "capital-gain": [0, 5000, 0, 0],
        "capital-loss": [0, 0, 0, 0],
        "hours-per-week": [40, 60, 35, 40],
        "native-country": ["United-States", "United-States", "United-States", "India"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"]
    })


def test_train_model_returns_random_forest(sample_data_test):
    """
    # This test verifies that train_model returns a fitted RandomForestClassifier
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    X, y, _, _ = process_data(
        sample_data_test,
        categorial_ceatures=cat_features,
        label="salary",
        training=True
    )

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)

    # Fitted RandomForest models have this attribute
    assert hasattr(model, "estimators_")
    assert len(model.estimators_) > 0

# TODO: implement the second test. Change the function name and input as needed
def test_process_data_returns_expected_shape(sample_Data_test):
    """
    # This test verifies that process_data returns correctly shaped feature and target arrays
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    X, y, encoder, lb = process_data(
        sample_data_test,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    assert isinstance(X, np.ndarray)

    assert X.shape[0] == sample_data_test.shape[0]

    assert len(y) == sample_data_test.shape[0]

    assert set(np.unique(y)).issubset({0, 1})

    assert encoder is not None
    assert lb is not None

# TODO: implement the third test. Change the function name and input as needed
def test_all_artifacts_returned(sample_data_test):
    """
    # This test verifies that process_data returns all the required artifacts with the correct types
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    X, y, encoder, lb = process_data(
        sample_data_test,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert isinstance(encoder, OneHotEncoder), "encoder should be OneHotEncoder"
    assert isinstance(lb, LabelBinarizer), "lb should be LabelBinarizer"

    #Here is where the we verify the artifacts have been fitted
    assert hasattr(encoder, "categories_")
    assert hasattr(lb, "classes_")
