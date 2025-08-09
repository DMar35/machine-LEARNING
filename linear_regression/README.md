# Theory

## What is Linear Regression:

- Linear regression is a form of supervised learning that maps a given
  $x$ to an output $y$ where $y$ can be any value. The hypothesis for
  linear regression is defined as follows:
  $$h_{\theta}(x) = \theta_0 + {\theta_1}{x_1} + {\theta_2}{x_2} + \dots            {\theta_n}x_n$$

- The goal is find the parameters $\theta$ such that they minimize the
  cost function $J(\theta)$. In the case of linear regression this is
  **mean squared error** which is defined as follows:
  $$J(\theta) = \frac{1}{2m} \sum_{i=1}^m \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)^2$$

## Variable definitions:

| Symbol             | Description                                                     |
|:-------------------|:----------------------------------------------------------------|
| $\theta$           | Parameters (model weights)                                      |
| $h_{\theta}(x)$    | Hypothesis function parameterized by $\theta$, mapping $x$ to predicted $y$ |
| $m$                | Number of training examples (row count)                        |
| $n$                | Number of features (column count)                              |
| $x$                | Input feature vector                                            |
| $y$                | Output/target variable                                          |
| $(x^{(i)}, y^{(i)})$| $i$-th training example                                        |


## Gradient Descent:

- Gradient descent is an optimization algorithm used to iteratively
  update the model parameters/weights($\theta$) of the model in order to
  minimize a given loss function. By moving the parameters in the
  direction of the steepest descent of the loss, the algorithm improves
  the model's performance over time. You can do this for example by
  starting with some $\theta$(i.e. a vector full of 0s).

- Formal Definition:
  $$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

### Stochastic Gradient Descent:

- SGD uses one training example per pass in contrast to regular batch
  gradient descent which uses all training examples. This makes it more
  efficient and therefore better for large datasets, albeit more
  unpredictable.

### Normal Equation:

- An equation for linear regression ONLY that finds the optimal value(s)
  of $\theta$ in one step. It uses the following formula:
  $$\theta = (X^{T}X)^{-1}X^{T}y$$

## How Linear Regression Works:

### Running gradient descent

As stated before, the goal of linear regression is to find the
parameters $\theta$ such that they minimize the mean squared error. To
do this, we perform the following steps:

1.  Compute our prediction:
    $h_\theta(x) = \theta_0 + \theta_1x + \theta_2x + \dots \theta_nx$

2.  Compute the error(loss) of our prediction: $h_\theta(x) - y$

3.  Compute the gradients:

    - Gradient for bias term:
      $$\frac{\partial J(\theta)}{\partial \theta_0} = \frac{1}{m} \sum \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)$$

    - Gradient for all other features:
      $$\frac{\partial J(\theta)}{\partial \theta_i} = \frac{1}{m} \sum \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr) x^{(i)}$$

4.  Update params:

    - For bias term:
      $\displaystyle \theta_0 := \theta_0 - \alpha \cdot \text{bias_gradient}$

    - For features:
      $\displaystyle \theta_i := \theta_i - \alpha \cdot \text{weight_gradient}$

5.  Repeat steps 1-4 until local optima is found(only global minimum for
    linear regression)

### An example loop

The following is an example for one loop using a singular training
example with one feature: ($x$ = 2, $y$ = 4) and a learning
rate($\alpha$) of 0.1

1.  Compute prediction:
    $h_\theta(x) = \theta_0 + \theta_1x = 0 + 0 \cdot 2 = 0$

2.  Computer error: $h_\theta(x) - y = 0 - 4 = -4$

3.  Compute gradients:

    - Bias gradient = $h_\theta(x^{(i)}) - y^{(i)} = -4 + 0 = -4$

    - Weight gradient =
      $h_\theta(x^{(i)}) - y^{(i)} = (-4 + 0) \cdot 2 = -8$

4.  Update params:

    - For bias term:
      $\displaystyle \theta_0 := \theta_0 - \alpha \cdot \text{bias_gradient} = 0 - 0.1 \cdot -4 = 0.4$

    - For features:
      $\displaystyle \theta_i := \theta_i - \alpha \cdot \text{weight_gradient} = 0 - 0.1 \cdot -8 = 0.8$




  <br></br>

# Coding: What I learned

## General

- How to use Sklearn to load datasets and split them into training/test
  sets.

- General architecture for how to build a model class(although this
  differed for NumPy and PyTorch).

- Exposure to different shorthand techniques such as using the @ symbol
  to multiply between matrices.

- Learned how to tune learning rates and adjust epoch count to improve
  model accuracy.

## NumPy version

- How to build a linear regression model from scratch without using any
  built-in linear regression utility functions.

- Deepened my understanding of the core steps of linear regression(and
  the math behind it), including hypothesis formulation, loss
  calculation, gradient computation, and parameter updates during
  training.

- Learned about some basic but useful NumPy functions such as
  np.random.randn for initializing the weights vector and bias term as
  well as np.mean.

## PyTorch version

- Learned how to take advantage of PyTorch's built in modules to
  automate most of the core functionality of the model structure
  including:

  - Creating a linear layer using **nn.Linear** to define the hypothesis
    for linear regression

  - The importance of passing in **nn.Module** so that the model we
    create can access the necessary features provided by PyTorch.

  - Calling **super().\_\_init\_\_()** as the first line of our
    **\_\_init\_\_** method.

  - Converting NumPy arrays to tensors via the **torch.from_numpy**
    method

  - Using the **view** method to convert the output vectors into 2d
    vectors so that they match the shape of the model's output.

  - Using PyTorch's autograd capabilities via **loss.backward()** to\
    automatically compute the gradients rather than doing it manually.

  - Using **optimizer.step()** to update the parameters and
    **optimizer.zero_grad()** to reset them before the next loop.

  - How passing in **model.parameters()** into our optimizer function
    (SGD in this case) automatically registers all learnable parameters
    (weights/biases) inside the **\_\_init\_\_** method.

- The **\_\_init\_\_** method is used to define the model's layers.

- The **forward()** method is where you define how input data flows
  through the layers to produce an output and how it is mandatory to
  name this method forward.

- How calling the **model** class created automatically invokes the
  **forward()** method

- In PyTorch, it is typically best practice to separate the training
  loop from the model class.

- Using **model.eval()** and **torch.no_grad()** to optimize for
  predictions and model evaluation once training is finished.


