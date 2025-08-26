## Feed-Forward Neural Networks

---

### Preface

Some mathematical notations to know:

- $\mathbb{R}$ means a real-valued scalar like $5$, $2.5$, $\pi$, etc. Other scalars are $\mathbb{N}$ for the natural mumbers, $\mathbb{Z}$ for integers, and $\mathbb{Q}$ for rational numbers. We will be commonly using $\mathbb{R}$, $\mathbb{N}$, and $mathbb{Z}$ to represent scalars.
- $\mathbb{R}^D$ means a $D$-dimensional column vector where the entries are real-valued so if $x \in \mathbb{R}^D$, then $x = (x_1, ..., x_D)$ where each $x_i \in \mathbb{R}$.
- $M_{r \times c}(\mathbb{R})$ means a matrix with $r$ rows and $c$ columns where the entries are real-valued.

---

### Linear regression

A common model for regression are linear regression models with the form:

$$\hat{y} = w^T x + b$$

$$y = \hat{y} + \epsilon = w^T x + b + \epsilon$$

where $x \in \mathbb{R}^D$ represents a single observation with $D$ parameters,
$w \in \mathbb{R}^D$ represents the linear coefficients,
$b \in \mathbb{R}$ represents the bias,
$\epsilon \sim N(0,\sigma^2)$ is Gaussian error,
$y \in \mathbb{R}$ represents the output,
and $\hat{y} \in \mathbb{R}$ is the predicted output of the model.
How can this model be extended to fit data that are non-linear and/or follow complex patterns?

#### Applying non-linearity

A simple way to make the linear model non-linear in respect to its input is to apply a non-linear transformation to the input $x$. Let $\phi: \mathbb{R}^D \rightarrow \mathbb{R}^D$ be a non-linear mapping. In mathematical notation:

$$\hat{y} = w^T \phi(x) + b$$

Some common non-linear mappings are polynomial $\phi(x) = x^k$, exponential $\phi(x) = a^x$,
and trigonometric $\phi(x) = \sin(x)$.
How would we be able to known which pattern the data follows?

#### Stacking Layers

Generally, data can follow very complex patterns in relation to its input. Rather than testing various forms of pattern like polynomial or exponential, we can treat $\hat{y} = w^T \phi(x) + b$ as a layer to learn some feature / pattern(s) of the data. Then, we can recursively stack $L \in \mathbb{N}$ of these layers so the model can learn more complex features based on previous features. In mathematical notation:

$$f_{\ell} = \phi(w_{\ell}^T f_{\ell-1}  + b_{\ell})$$

---

### Feed-Forward Neural Networks

Simply, Feed-Forward Neural Networks (FFNNs) are just recursively stacking non-linear layers.
In full mathematical notation, a FFNN model:

$$f(x; \theta) = f_L(x; \theta_L)$$

where

- Pre-activation: <span style="color: blue;"> $z_{\ell} = W_{\ell} f_{\ell-1}(x; \theta_{\ell-1}) + b_{\ell} 1_{D_{\ell}}$ </span> where $1_{D_{\ell}}$ is a $D_{\ell}$-dimensional vector of 1s.
- Neuron: <span style="color: blue;"> $f_{\ell}(x; \theta_{\ell}) = \phi_{\ell}(z_{\ell}; V_{\ell})$ </span>

such that $f_{0}(x; \theta_{0}) = x$ is the initial input,
$\theta_{\ell} = (b_{\ell}, W_{\ell})$ are the parameters in the $\ell$-th layer,
and $\theta = \cup_{\ell=1}^L \theta_{\ell}$ are all the parameters in the model.

- Activation Function: $\phi_{\ell}: \mathbb{R}^{D_{\ell}} \rightarrow \mathbb{R}^{D_{\ell}}$ where $V_{\ell}$ are specified parameters of the activation function for layer $\ell$
- Weights: $W_{\ell} \in M_{D_{\ell} \times D_{\ell-1}}(\mathbb{R})$
- Bias: $b_{\ell} \in \mathbb{R}$
- Hidden Units: $D_{\ell} \in \mathbb{N}$ is the dimension of layer $\ell$

Now, knowing the form of a FFNN, we still need to cover how a FFNN "learns" to fit the data.
Some topics covered are:

- Which activation functions $\phi_{\ell}$ are commonly used and why?
- Loss and Optimization: How to determine if a FFNN is "improving" and what is the process to improve FFNNs?
- Weight-initialization Methods: What values should the parameters $\theta$ be initially set to?
- Regularization Techniques: How to make control how much a FFNN improves so it does not overfit the data?

---

### Bonus: Calculating Total Parameters and Computational Complexity

Recall the total parameters of a FFNN is $\theta = \cup_{\ell=1}^L \theta_{\ell}$
where $\theta_{\ell} = (b_{\ell}, W_{\ell})$ are the parameters in the $\ell$-th layer.
Since $W_{\ell} \in M_{D_{\ell} \times D_{\ell-1}}(\mathbb{R})$ and $b_{\ell} \in \mathbb{R}$,
then the total number of parameters are:

$$\sum_{\ell=1}^L (D_{\ell} D_{\ell-1} + 1)$$

A common metric to measure the computational complexity is determing the total number of floating points operations, FLOPS,
such as addition, subtraction, etc. In the $\ell$-th layer, for each row of $W_{\ell}$, each entry of the $D_{\ell-1}$ entries are multiplied to entries in $f_{\ell-1}(x; \theta_{\ell-1})$ and then, those $D_{\ell-1}$ are added together along with the bias $b_{\ell}$. Since there are $D_{\ell}$ rows, then there is a total of $D_{\ell} (D_{\ell-1} + (D_{\ell-1}-1) + 1) = 2 D_{\ell} D_{\ell-1}$ FLOPs in the $\ell$-th layer. Thus, the total number of FLOPs:

$$\sum_{\ell=1}^L 2 D_{\ell} D_{\ell-1}$$




