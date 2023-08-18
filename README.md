# AutoRA Falsification Experimentalist

The falsification pooler and sampler identify novel experimental conditions $X'$ under 
which the loss $\hat{\mathcal{L}}(M,X,Y,X')$ of the best 
candidate model is predicted to be the highest. This loss is 
approximated with a multi-layer perceptron, which is trained to 
predict the loss of a candidate model, $M$, given experiment 
conditions $X$  and dependent measures $Y$ that have already been probed:

$$
\underset{X'}{argmax}~\hat{\mathcal{L}}(M,X,Y,X').
$$

## Quickstart Guide

You will need:

- `python` 3.8 or greater: [https://www.python.org/downloads/](https://www.python.org/downloads/)

*Falsification Experimentalist* is a part of the `autora` package:

```shell
pip install -U autora["experimentalist-falsification"]
```


Check your installation by running:
```shell
python -c "from autora.experimentalist.falsification import falsification_pool"
```
