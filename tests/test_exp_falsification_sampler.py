import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression

from autora.experimentalist.falsification import (
    falsification_sample,
    falsification_score_sample,
    falsification_score_sample_from_predictions,
)
from autora.experimentalist.grid import pool
from autora.experimentalist.pipeline import Pipeline
from autora.variable import DV, IV, ValueType, VariableCollection
from tests.test_exp_falsification_pooler import get_sin_data, get_xor_data

x_min_regression = 0
x_max_regression = 6


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


@pytest.fixture
def classification_data_to_test():
    data = np.array(
        [
            [1, 0],
            [0, 1],
            [0, 0],
            [1, 1],
        ]
    )
    return data


@pytest.fixture
def seed():
    """
    Ensures that the results are the same each time the tests are run.
    """
    torch.manual_seed(180)
    np.random.seed(180)
    return


@pytest.fixture
def get_square_data():
    X = np.linspace(x_min_regression, x_max_regression, 100)
    Y = np.square(X)
    return X, Y


@pytest.fixture
def regression_data_to_test():
    data = np.linspace(x_min_regression, x_max_regression, 11)
    return data


def test_falsification_classification(
    synthetic_logr_model, classification_data_to_test, seed
):
    # Import model and data_closed_loop
    X_train, Y_train = get_xor_data()
    X = classification_data_to_test
    model = synthetic_logr_model

    # Specify independent variables
    iv1 = IV(
        name="x",
        value_range=(0, 5),
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

    # Run falsification sampler
    falsification_pipeline = Pipeline(
        [("sampler", falsification_sample)],
        params={
            "sampler": dict(
                conditions=X,
                model=model,
                reference_conditions=X_train,
                reference_observations=Y_train,
                metadata=metadata,
                num_samples=2,
                training_epochs=1000,
                training_lr=1e-3,
            ),
        },
    )

    samples = np.array(falsification_pipeline.run())

    # Check that at least one of the resulting samples is the one that is
    # underrepresented in the data_closed_loop used for model training

    assert (samples[0, :] == [1, 1]).all or (samples[1, :] == [1, 1]).all


def test_falsification_regression(synthetic_linr_model, regression_data_to_test, seed):

    # Import model and data_closed_loop
    X_train, Y_train = get_sin_data()
    X = regression_data_to_test
    model = synthetic_linr_model

    # specify meta data_closed_loop

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

    falsification_pipeline = Pipeline(
        [("sampler", falsification_sample)],
        params={
            "sampler": dict(
                conditions=X,
                model=model,
                reference_conditions=X_train,
                reference_observations=Y_train,
                metadata=metadata,
                num_samples=5,
                training_epochs=1000,
                training_lr=1e-3,
                plot=False,
            ),
        },
    )

    sample = np.array(falsification_pipeline.run())

    # the first value should be close to one of the local maxima of the
    # sine function
    assert sample[0] == 0 or sample[0] == 6
    if sample[0] == 0:
        assert (
            sample[1] == 6
            or np.round(sample[1], 2) == 1.8
            or np.round(sample[1], 2 == 4.2)
        )

    assert (
        np.round(sample[2], 2) == 1.8
        or np.round(sample[2], 2) == 4.2
        or np.round(sample[2], 2) == 6
    )
    if np.round(sample[2], 2) == 1.8:
        assert np.round(sample[3], 2) == 4.2


def test_falsification_regression_without_model(
    synthetic_linr_model, get_square_data, regression_data_to_test, seed
):
    # obtain the data for training the model
    X_train, Y_train = get_square_data

    # obtain candidate conditions to be evaluated
    X = regression_data_to_test

    # reshape data
    X_train = X_train.reshape(-1, 1)
    Y_train = Y_train.reshape(-1, 1)
    X = X.reshape(-1, 1)

    # fit a linear model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # compute model predictions for trained conditions
    Y_predicted = model.predict(X_train)

    # get scores from falsification sampler
    X_selected, scores = falsification_score_sample_from_predictions(
        conditions=X,
        predicted_observations=Y_predicted,
        reference_conditions=X_train,
        reference_observations=Y_train,
        training_epochs=1000,
        training_lr=1e-3,
        plot=False,
    )

    # check if the scores are normalized
    assert np.round(np.mean(scores), 4) == 0
    assert np.round(np.std(scores), 4) == 1

    # check if the scores are properly ordered
    assert scores[0] > scores[1] > scores[2]

    # check if the right data points were selected
    assert X_selected[0, 0] == 0 or X_selected[0, 0] == 6
    assert X_selected[1, 0] == 0 or X_selected[1, 0] == 6
    # assert X_selected[2, 0] == 3


def test_falsification_reconstruction_without_model(
    synthetic_linr_model, get_square_data, regression_data_to_test, seed
):

    # obtain candidate conditions to be evaluated
    X = regression_data_to_test

    # generate sampled conditions
    X_train = np.linspace(x_min_regression, x_max_regression, 100)

    # generate reconstructed data (this data may be produced by an autoencoder)
    X_reconstructed = X_train + np.sin(X_train)

    # get scores from falsification sampler
    X_selected, scores = falsification_score_sample_from_predictions(
        conditions=X,
        predicted_observations=X_reconstructed,
        reference_conditions=X_train,
        reference_observations=X_train,
        training_epochs=1000,
        training_lr=1e-3,
        plot=False,
    )

    # check if the scores are normalized
    assert np.round(np.mean(scores), 4) == 0
    assert np.round(np.std(scores), 4) == 1

    # check if the scores are properly ordered
    assert scores[0] > scores[1]

    # check if the data points with the highest predicted error were selected
    assert np.round(X_selected[0, 0], 4) == 1.8 or np.round(X_selected[0, 0], 4) == 4.8
    assert np.round(X_selected[1, 0], 4) == 1.8 or np.round(X_selected[1, 0], 4) == 4.8


def test_iterator_input(synthetic_linr_model):
    # Import model and data_closed_loop
    X_train, Y_train = get_sin_data()
    model = synthetic_linr_model

    # specify meta data_closed_loop

    # Specify independent variables
    iv = IV(
        name="x",
        value_range=(0, 2 * np.pi),
        allowed_values=(np.linspace(0, 2 * np.pi, 100)),
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

    X = pool(metadata)

    new_conditions = np.array(falsification_sample(
        conditions=X,
        model=model,
        reference_conditions=X_train,
        reference_observations=Y_train,
        metadata=metadata,
        num_samples=5,
        training_epochs=1000,
        training_lr=1e-3,
        plot=False,
    ))

    assert new_conditions.shape[0] == 5


def test_falsification_pandas(synthetic_logr_model, classification_data_to_test, seed):
    # Import model and data_closed_loop
    X_train, Y_train = get_xor_data()
    X = classification_data_to_test
    model = synthetic_logr_model

    X = pd.DataFrame(X, columns=["x1", "x2"])
    # X_train = pd.DataFrame(X_train, columns=["x1", "x2"])
    # Y_train = pd.DataFrame(Y_train, columns=["y"])

    # Specify independent variables
    iv1 = IV(
        name="x1",
        value_range=(0, 5),
        units="intensity",
        variable_label="stimulus 1",
    )

    iv2 = IV(
        name="x2",
        value_range=(0, 5),
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

    # Run falsification sampler
    falsification_pipeline = Pipeline(
        [("sampler", falsification_sample)],
        params={
            "sampler": dict(
                conditions=X,
                model=model,
                reference_conditions=X_train,
                reference_observations=Y_train,
                metadata=metadata,
                num_samples=2,
                training_epochs=1000,
                training_lr=1e-3,
            ),
        },
    )

    samples = falsification_pipeline.run()

    assert isinstance(samples, pd.DataFrame)
    assert samples.columns.tolist() == ["x1", "x2"]

    # Check that at least one of the resulting samples is the one that is
    # underrepresented in the data_closed_loop used for model training

    assert (np.array(samples.iloc[0]) == [1, 1]).all or (
        np.array(samples.iloc[1]) == [1, 1]
    ).all


def test_pandas_score():
    # Specify X and Y
    X = np.linspace(0, 2 * np.pi, 100)
    Y = np.sin(X)
    X_prime = np.linspace(0, 6.5, 14)

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

    X = pd.DataFrame(X, columns=["x"])
    Y = pd.DataFrame(Y, columns=["y"])
    X_prime = pd.DataFrame(X_prime, columns=["x"])

    # Sample four novel conditions
    X_selected = falsification_sample(
        conditions=X_prime,
        model=model,
        reference_conditions=X,
        reference_observations=Y,
        metadata=metadata,
        num_samples=4,
    )

    assert isinstance(X_selected, pd.DataFrame)
    assert X_selected.columns.tolist() == ["x"]

    # We may also obtain samples along with their z-scored novelty scores
    X_selected = falsification_score_sample(
        conditions=X_prime,
        model=model,
        reference_conditions=X,
        reference_observations=Y,
        metadata=metadata,
        num_samples=4,
    )


def test_doc_example():
    # Specify X and Y
    X = np.linspace(0, 2 * np.pi, 100)
    Y = np.sin(X)
    X_prime = np.linspace(0, 6.5, 14)

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
    X_selected = falsification_sample(
        conditions=X_prime,
        model=model,
        reference_conditions=X,
        reference_observations=Y,
        metadata=metadata,
        num_samples=4,
    )

    # convert Iterable to numpy array
    X_selected = np.array(list(X_selected))

    # We may also obtain samples along with their z-scored novelty scores
    X_selected = falsification_score_sample(
        conditions=X_prime,
        model=model,
        reference_conditions=X,
        reference_observations=Y,
        metadata=metadata,
        num_samples=4,
    )

    print(X_selected)
