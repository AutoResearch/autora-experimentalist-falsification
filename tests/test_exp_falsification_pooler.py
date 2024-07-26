import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression

from autora.experimentalist.falsification import falsification_pool
from autora.variable import DV, IV, ValueType, VariableCollection


@pytest.fixture
def seed():
    """
    Ensures that the results are the same each time the tests are run.
    """
    torch.manual_seed(180)
    return


def get_xor_data(n: int = 10):
    X = ([[1, 0]] * n) + ([[0, 1]] * n) + ([[0, 0]] * n) + ([[1, 1]])
    y = ([1] * n) + ([1] * n) + ([0] * n) + ([0])
    return X, y


def get_sin_data(n: int = 100):
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x)
    return x, y


@pytest.fixture
def synthetic_logr_model():
    """
    Creates logistic regression classifier for 3 classes based on synthetic data_closed_loop.
    """
    X, y = get_xor_data()
    model = LogisticRegression()
    model.fit(X, y)
    return model


@pytest.fixture
def synthetic_linr_model():
    """
    Creates linear regression based on synthetic data_closed_loop.
    """
    x, y = get_sin_data()
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model


def test_falsification_pool_classification(synthetic_logr_model, seed):

    # Import model and data_closed_loop
    conditions, observations = get_xor_data()
    model = synthetic_logr_model

    # Specify independent variables
    iv1 = IV(
        name="x",
        value_range=(0, 1),
        units="intensity",
        variable_label="stimulus 1",
    )

    # specify dependent variables
    dv1 = DV(
        name="y",
        value_range=(0, 1),
        units="class",
        variable_label="class",
        type=ValueType.CLASS,
    )

    # Variable collection with ivs and dvs
    metadata = VariableCollection(
        independent_variables=[iv1, iv1],
        dependent_variables=[dv1],
    )

    # Run falsification pooler
    new_conditions = falsification_pool(
        model=model,
        reference_conditions=conditions,
        reference_observations=observations,
        metadata=metadata,
        num_samples=2,
        training_epochs=1000,
        optimization_epochs=1000,
        training_lr=1e-3,
        optimization_lr=1e-3,
        limit_offset=10**-10,
        limit_repulsion=0,
        plot=False,
    )

    # convert Iterable to numpy array
    new_conditions = np.array(new_conditions)

    # Check that at least one of the resulting samples is the one that is
    # underrepresented in the data_closed_loop used for model training
    assert (new_conditions[0, 0] > 0.99 and new_conditions[0, 1] > 0.99) or (
        new_conditions[1, 0] > 0.99 and new_conditions[1, 1] > 0.99
    )


def test_falsification_pool_regression(synthetic_linr_model, seed):

    # Import model and data_closed_loop
    conditions, observations = get_sin_data()
    model = synthetic_linr_model

    # Specify independent variables
    iv = IV(
        name="x",
        value_range=(0, 2 * np.pi),
        units="intensity",
        variable_label="stimulus",
    )

    # specify dependent variables
    dv = DV(
        name="y",
        value_range=(-1, 1),
        units="real",
        variable_label="response",
        type=ValueType.REAL,
    )

    # Variable collection with ivs and dvs
    metadata = VariableCollection(
        independent_variables=[iv],
        dependent_variables=[dv],
    )

    new_conditions = falsification_pool(
        model=model,
        reference_conditions=conditions,
        reference_observations=observations,
        metadata=metadata,
        num_samples=5,
        training_epochs=1000,
        optimization_epochs=5000,
        training_lr=1e-3,
        optimization_lr=5e-3,
        limit_offset=0,
        limit_repulsion=0.01,
        plot=False,
    )

    # convert Iterable to numpy array
    new_conditions = np.array(new_conditions)

    for condition in new_conditions:
        assert (
            condition < 0.1
            or condition > 6.1
            or (condition < 2.5 and condition > 1.5)
            or (condition < 5 and condition > 4)
        )


def test_falsification_pandas(synthetic_logr_model, seed):

    # Import model and data_closed_loop
    conditions, observations = get_xor_data()
    model = synthetic_logr_model

    conditions = pd.DataFrame(conditions, columns=["x1", "x2"])
    observations = pd.DataFrame(observations, columns=["y"])

    # Specify independent variables
    iv1 = IV(
        name="x1",
        value_range=(0, 1),
        units="intensity",
        variable_label="stimulus 1",
    )

    iv2 = IV(
        name="x2",
        value_range=(0, 1),
        units="intensity",
        variable_label="stimulus 2",
    )

    # specify dependent variables
    dv1 = DV(
        name="y",
        value_range=(0, 1),
        units="class",
        variable_label="class",
        type=ValueType.CLASS,
    )

    # Variable collection with ivs and dvs
    metadata = VariableCollection(
        independent_variables=[iv1, iv2],
        dependent_variables=[dv1],
    )

    # Run falsification pooler
    new_conditions = falsification_pool(
        model=model,
        reference_conditions=conditions,
        reference_observations=observations,
        metadata=metadata,
        num_samples=2,
        training_epochs=1000,
        optimization_epochs=1000,
        training_lr=1e-3,
        optimization_lr=1e-3,
        limit_offset=10**-10,
        limit_repulsion=0,
        plot=False,
    )

    # convert Iterable to numpy array
    new_conditions = np.array(new_conditions)

    # Check that at least one of the resulting samples is the one that is
    # underrepresented in the data_closed_loop used for model training
    assert (new_conditions[0, 0] > 0.99 and new_conditions[0, 1] > 0.99) or (
        new_conditions[1, 0] > 0.99 and new_conditions[1, 1] > 0.99
    )


def test_doc_example():
    # Specify X and Y
    X = np.linspace(0, 2 * np.pi, 100)
    Y = np.sin(X)

    # We need to provide the pooler with some metadata specifying the
    # independent and dependent variables
    # Specify independent variable
    iv = IV(
        name="x",
        value_range=(0, 2 * np.pi),
    )

    # specify dependent variable
    dv = DV(
        name="y",
        type=ValueType.REAL,
    )

    # Variable collection with ivs and dvs
    metadata = VariableCollection(
        independent_variables=[iv],
        dependent_variables=[dv],
    )

    # Fit a linear regression to the data
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)

    # Sample four novel conditions
    X_sampled = falsification_pool(
        model=model,
        reference_conditions=X,
        reference_observations=Y,
        metadata=metadata,
        num_samples=4,
        limit_repulsion=0.01,
    )

    # convert Iterable to numpy array
    X_sampled = np.array(X_sampled)

    print(X_sampled)
