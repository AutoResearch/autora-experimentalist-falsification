# Falsification Pooler

The falsification pooler identifies novel experimental conditions $X'$ under 
which the loss $\hat{\mathcal{L}}(M,X,Y,X')$ of the best 
candidate model is predicted to be the highest. This loss is 
approximated with a multi-layer perceptron, which is trained to 
predict the loss of a candidate model, $M$, given experiment 
conditions $X$  and dependent measures $Y$ that have already been probed:

$$
\underset{X'}{argmax}~\hat{\mathcal{L}}(M,X,Y,X').
$$


## Example

To illustrate the falsification strategy, consider a dataset representing the sine function:

$$
f(X) = \sin(X).
$$

The dataset consists of 100 data points ranging from $X=0$ to $X=2\pi$.

In addition, let's consider a linear regression as a model ($M$) of the data. 

The following figure illustrates the prediction of the fitted linear regression
(shown in blue) for the pre-collected sine dataset (conditions $X$ and observations $Y$; shown in red):

![Linear Regression vs. Sinus Data](pooler-model-vs-data.png)

One can observe that the linear regression is a poor fit for the sine data, in particular for regions around the 
extrema of the sine function, as well as the lower and upper bounds of the domain.

The figure below shows the mean-squared error (MSE) of the linear regression 
as a function of the input $X$ (red dots):

![MSE of Linear Regression](pooler-mse.png)

The falsification sampler attempts to predict the MSE of the linear regression using a neural network (shown in blue).

Once the falsiifcaiton sampler has been trained, it can be used to identify novel experimental conditions $X'$ 
that are predicted to maximize the predicted MSE, such as at the boundaries of the domain, 
as well as around the extrema of the sine function. An example output of the falsification sampler is:

````
[0.        ]
[4.17222738]
[4.17222738]
[6.28318531]]
````

To prevent the falsification sampler from sampling at the limits of the domain ($0$ and $2/pi$),
it can be provided with optional parameter ``limit_repulsion`` that bias samples for new
experimental conditions away from the boundaries of $X$, as shown in the example below.

### Example Code
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from autora.variable import DV, IV, ValueType, VariableCollection
from autora.experimentalist.pooler.falsification import falsification_pooler

# Specify X and Y
X = np.linspace(0, 2 * np.pi, 100)
Y = np.sin(X)

# We need to provide the pooler with some metadata specifying the independent and dependent variables
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
X_sampled = falsification_pooler(
    model=model,
    reference_conditions=X,
    reference_observations=Y,
    metadata=metadata,
    num_samples=4,
    limit_repulsion=0.01,
)

# convert Iterable to numpy array
X_sampled = np.array(list(X_sampled))

print(X_sampled)
```

Output:
````
[[6.28318531]
 [2.16611028]
 [2.16512322]
 [2.17908978]]
````

