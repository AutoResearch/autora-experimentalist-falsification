from typing import List, Optional, Tuple, cast

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Variable

from autora.variable import ValueType, VariableCollection

from autora.utils.deprecation import deprecated_alias


def falsification_pool(
    model,
    reference_conditions: np.ndarray,
    reference_observations: np.ndarray,
    metadata: VariableCollection,
    num_samples: int = 100,
    training_epochs: int = 1000,
    optimization_epochs: int = 1000,
    training_lr: float = 1e-3,
    optimization_lr: float = 1e-3,
    limit_offset: float = 0,  # 10**-10,
    limit_repulsion: float = 0,
    plot: bool = False,
):
    """
    A pooler that generates samples for independent variables with the objective of maximizing the
    (approximated) loss of the model. The samples are generated by first training a neural network
    to approximate the loss of a model for all patterns in the training data.
    Once trained, the network is then inverted to generate samples that maximize the approximated
    loss of the model.

    Note: If the pooler returns samples that are close to the boundaries of the variable space,
    then it is advisable to increase the limit_repulsion parameter (e.g., to 0.000001).

    Args:
        model: Scikit-learn model, could be either a classification or regression model
        reference_conditions: data that the model was trained on
        reference_observations: labels that the model was trained on
        metadata: Meta-data about the dependent and independent variables
        num_samples: number of samples to return
        training_epochs: number of epochs to train the popper network for approximating the
        error fo the model
        optimization_epochs: number of epochs to optimize the samples based on the trained
        popper network
        training_lr: learning rate for training the popper network
        optimization_lr: learning rate for optimizing the samples
        limit_offset: a limited offset to prevent the samples from being too close to the value
        boundaries
        limit_repulsion: a limited repulsion to prevent the samples from being too close to the
        allowed value boundaries
        plot: print out the prediction of the popper network as well as its training loss

    Returns: Sampled pool

    """

    # format input

    reference_conditions = np.array(reference_conditions)
    if len(reference_conditions.shape) == 1:
        reference_conditions = reference_conditions.reshape(-1, 1)

    x = np.empty([num_samples, reference_conditions.shape[1]])

    reference_observations = np.array(reference_observations)
    if len(reference_observations.shape) == 1:
        reference_observations = reference_observations.reshape(-1, 1)

    if metadata.dependent_variables[0].type == ValueType.CLASS:
        # find all unique values in reference_observations
        num_classes = len(np.unique(reference_observations))
        reference_observations = class_to_onehot(reference_observations, n_classes=num_classes)

    reference_conditions_tensor = torch.from_numpy(reference_conditions).float()

    iv_limit_list = get_iv_limits(reference_conditions, metadata)

    popper_net, model_loss = train_popper_net_with_model(model,
                                              reference_conditions,
                                              reference_observations,
                                              metadata,
                                              iv_limit_list,
                                              training_epochs,
                                              training_lr,
                                              plot)

    # now that the popper network is trained we can sample new data points
    # to sample data points we need to provide the popper network with an initial
    # condition we will sample those initial conditions proportional to the loss of the current
    # model

    # feed average model losses through softmax
    # model_loss_avg= torch.from_numpy(np.mean(model_loss.detach().numpy(), axis=1)).float()
    softmax_func = torch.nn.Softmax(dim=0)
    probabilities = softmax_func(model_loss)
    # sample data point in proportion to model loss
    transform_category = torch.distributions.categorical.Categorical(probabilities)

    popper_net.freeze_weights()

    for condition in range(num_samples):

        index = transform_category.sample()
        input_sample = torch.flatten(reference_conditions_tensor[index, :])
        popper_input = Variable(input_sample, requires_grad=True)

        # invert the popper network to determine optimal experiment conditions
        for optimization_epoch in range(optimization_epochs):
            # feedforward pass on popper network
            popper_prediction = popper_net(popper_input)
            # compute gradient that maximizes output of popper network
            # (i.e. predicted loss of original model)
            popper_loss_optim = -popper_prediction
            popper_loss_optim.backward()

            with torch.no_grad():

                # first add repulsion from variable limits
                for idx in range(len(input_sample)):
                    iv_value = popper_input[idx]
                    iv_limits = iv_limit_list[idx]
                    dist_to_min = np.abs(iv_value - np.min(iv_limits))
                    dist_to_max = np.abs(iv_value - np.max(iv_limits))
                    # deal with boundary case where distance is 0 or very small
                    dist_to_min = np.max([dist_to_min, 0.00000001])
                    dist_to_max = np.max([dist_to_max, 0.00000001])
                    repulsion_from_min = limit_repulsion / (dist_to_min**2)
                    repulsion_from_max = limit_repulsion / (dist_to_max**2)
                    iv_value_repulsed = (
                        iv_value + repulsion_from_min - repulsion_from_max
                    )
                    popper_input[idx] = iv_value_repulsed

                # now add gradient for theory loss maximization
                delta = -optimization_lr * popper_input.grad
                popper_input += delta

                # finally, clip input variable from its limits
                for idx in range(len(input_sample)):
                    iv_raw_value = input_sample[idx]
                    iv_limits = iv_limit_list[idx]
                    iv_clipped_value = np.min(
                        [iv_raw_value, np.max(iv_limits) - limit_offset]
                    )
                    iv_clipped_value = np.max(
                        [
                            iv_clipped_value,
                            np.min(iv_limits) + limit_offset,
                        ]
                    )
                    popper_input[idx] = iv_clipped_value
                popper_input.grad.zero_()

        # add condition to new experiment sequence
        for idx in range(len(input_sample)):
            iv_limits = iv_limit_list[idx]

            # first clip value
            iv_clipped_value = np.min([iv_raw_value, np.max(iv_limits) - limit_offset])
            iv_clipped_value = np.max(
                [iv_clipped_value, np.min(iv_limits) + limit_offset]
            )
            # make sure to convert variable to original scale
            iv_clipped_scaled_value = iv_clipped_value

            x[condition, idx] = iv_clipped_scaled_value

    return iter(x)

def get_iv_limits(
        reference_conditions: np.ndarray,
        metadata: VariableCollection,
                  ):
    """
    Get the limits of the independent variables

    Args:
        reference_conditions: data that the model was trained on
        metadata: Meta-data about the dependent and independent variables

    Returns: List of limits for each independent variable
    """

    # create list of IV limits
    iv_limit_list = list()
    if metadata is not None:
        ivs = metadata.independent_variables
        for iv in ivs:
            if hasattr(iv, "value_range"):
                value_range = cast(Tuple, iv.value_range)
                lower_bound = value_range[0]
                upper_bound = value_range[1]
                iv_limit_list.append(([lower_bound, upper_bound]))
    else:
        for col in range(reference_conditions.shape[1]):
            min = np.min(reference_conditions[:, col])
            max = np.max(reference_conditions[:, col])
            iv_limit_list.append(([min, max]))

    return iv_limit_list


def train_popper_net_with_model(
    model,
    reference_conditions: np.ndarray,
    reference_observations: np.ndarray,
    metadata: VariableCollection,
    iv_limit_list: List,
    training_epochs: int = 1000,
    training_lr: float = 1e-3,
    plot: bool = False,
):
    """
    Trains a neural network to approximate the loss of a model for all patterns in the training data
    Once trained, the network is then inverted to generate samples that maximize the approximated
    loss of the model.

    Note: If the pooler returns samples that are close to the boundaries of the variable space,
    then it is advisable to increase the limit_repulsion parameter (e.g., to 0.000001).

    Args:
        model: Scikit-learn model, could be either a classification or regression model
        reference_conditions: data that the model was trained on
        reference_observations: labels that the model was trained on
        metadata: Meta-data about the dependent and independent variables
        training_epochs: number of epochs to train the popper network for approximating the
        error fo the model
        training_lr: learning rate for training the popper network
        plot: print out the prediction of the popper network as well as its training loss

    Returns: Trained popper net.

    """

    model_predict = getattr(model, "predict_proba", None)
    if callable(model_predict) is False:
        model_predict = getattr(model, "predict", None)

    if callable(model_predict) is False or model_predict is None:
        raise Exception("Model must have `predict` or `predict_proba` method.")

    model_prediction = model_predict(reference_conditions)

    return train_popper_net(model_prediction,
                         reference_conditions,
                         reference_observations,
                         metadata,
                         iv_limit_list,
                         training_epochs,
                         training_lr,
                         plot)



def train_popper_net(
    model_prediction,
    reference_conditions: np.ndarray,
    reference_observations: np.ndarray,
    metadata: VariableCollection,
    iv_limit_list: List,
    training_epochs: int = 1000,
    training_lr: float = 1e-3,
    plot: bool = False,
):
    """
    Trains a neural network to approximate the loss of a model for all patterns in the training data
    Once trained, the network is then inverted to generate samples that maximize the approximated
    loss of the model.

    Note: If the pooler returns samples that are close to the boundaries of the variable space,
    then it is advisable to increase the limit_repulsion parameter (e.g., to 0.000001).

    Args:
        model: Scikit-learn model, could be either a classification or regression model
        reference_conditions: data that the model was trained on
        reference_observations: labels that the model was trained on
        metadata: Meta-data about the dependent and independent variables
        training_epochs: number of epochs to train the popper network for approximating the
        error fo the model
        training_lr: learning rate for training the popper network
        plot: print out the prediction of the popper network as well as its training loss

    Returns: Trained popper net.

    """

    # get dimensions of input and output
    n_input = reference_conditions.shape[1]
    n_output = 1  # only predicting one MSE

    # get input pattern for popper net
    popper_input = Variable(torch.from_numpy(reference_conditions), requires_grad=False).float()

    # get target pattern for popper net
    if isinstance(model_prediction, np.ndarray) is False:
        try:
            model_prediction = np.array(model_prediction)
        except Exception:
            raise Exception("Model prediction must be convertable to numpy array.")
    if model_prediction.ndim == 1:
        model_prediction = model_prediction.reshape(-1, 1)

    criterion = nn.MSELoss()
    model_loss = (model_prediction - reference_observations) ** 2
    model_loss = np.mean(model_loss, axis=1)

    # standardize the loss
    scaler = StandardScaler()
    model_loss = scaler.fit_transform(model_loss.reshape(-1, 1)).flatten()

    model_loss = torch.from_numpy(model_loss).float()
    popper_target = Variable(model_loss, requires_grad=False)

    # create the network
    popper_net = PopperNet(n_input, n_output)

    # reformat input in case it is 1D
    if len(popper_input.shape) == 1:
        popper_input = popper_input.flatten()
        popper_input = popper_input.reshape(-1, 1)

    # define the optimizer
    popper_optimizer = torch.optim.Adam(popper_net.parameters(), lr=training_lr)

    # train the network
    losses = []
    for epoch in range(training_epochs):
        popper_prediction = popper_net(popper_input)
        loss = criterion(popper_prediction, popper_target.reshape(-1, 1))
        popper_optimizer.zero_grad()
        loss.backward()
        popper_optimizer.step()
        losses.append(loss.item())

    if plot:
        if len(iv_limit_list) > 1:
            Warning("Plotting currently not supported for more than two independent variables.")
        else:
            popper_input_full = np.linspace(
                iv_limit_list[0][0], iv_limit_list[0][1], 1000
            ).reshape(-1, 1)
            popper_input_full = Variable(
                torch.from_numpy(popper_input_full), requires_grad=False
            ).float()
            popper_prediction = popper_net(popper_input_full)
            plot_falsification_diagnostics(
                losses,
                popper_input,
                popper_input_full,
                popper_prediction,
                popper_target,
                model_prediction,
                reference_observations,
            )

    return popper_net, model_loss





def plot_falsification_diagnostics(
    losses,
    popper_input,
    popper_input_full,
    popper_prediction,
    popper_target,
    model_prediction,
    target,
):
    import matplotlib.pyplot as plt

    if popper_input.shape[1] > 1:
        plot_input = popper_input[:, 0]
    else:
        plot_input = popper_input

    if model_prediction.ndim > 1:
        if model_prediction.shape[1] > 1:
            model_prediction = model_prediction[:, 0]
            target = target[:, 0]

    # PREDICTED MODEL ERROR PLOT
    plot_input_order = np.argsort(np.array(plot_input).flatten())
    plot_input = plot_input[plot_input_order]
    popper_target = popper_target[plot_input_order]
    # popper_prediction = popper_prediction[plot_input_order]
    plt.plot(popper_input_full, popper_prediction.detach().numpy(), label="Predicted MSE of the Model")
    plt.scatter(
        plot_input, popper_target.detach().numpy(), s=20, c="red", label="True MSE of the Model"
    )
    plt.xlabel("Experimental Condition X")
    plt.ylabel("MSE of Model")
    plt.title("Prediction of Falsification Network")
    plt.legend()
    plt.show()

    # CONVERGENCE PLOT
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss for the Falsification Network")
    plt.show()

    # MODEL PREDICTION PLOT
    model_prediction = model_prediction[plot_input_order]
    target = target[plot_input_order]
    plt.plot(plot_input, model_prediction, label="Model Prediction")
    plt.scatter(plot_input, target, s=20, c="red", label="Data")
    plt.xlabel("Experimental Condition X")
    plt.ylabel("Observation Y")
    plt.title("Model Prediction Vs. Data")
    plt.legend()
    plt.show()


# define the network
class PopperNet(nn.Module):
    def __init__(self, n_input: torch.Tensor, n_output: torch.Tensor):
        # Perform initialization of the pytorch superclass
        super(PopperNet, self).__init__()

        # Define network layer dimensions
        D_in, H1, H2, H3, D_out = [n_input, 64, 64, 64, n_output]

        # Define layer types
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)

    def forward(self, x: torch.Tensor):
        """
        This method defines the network layering and activation functions
        """
        x = self.linear1(x)  # hidden layer
        x = torch.tanh(x)  # activation function

        x = self.linear2(x)  # hidden layer
        x = torch.tanh(x)  # activation function

        x = self.linear3(x)  # hidden layer
        x = torch.tanh(x)  # activation function

        x = self.linear4(x)  # output layer

        return x

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False


def class_to_onehot(y: np.array, n_classes: Optional[int] = None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        n_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not n_classes:
        n_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, n_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (n_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

falsification_pooler = deprecated_alias(falsification_pool, "falsification_pooler")
