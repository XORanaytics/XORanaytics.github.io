## Gradient Descent

---

### Preface

Some mathematical notations to know:

- $\mathbb{R}$ means a real-valued scalar like $5$, $2.5$, $\pi$, etc. Other scalars are $\mathbb{N}$ for the natural mumbers, $\mathbb{Z}$ for integers, and $\mathbb{Q}$ for rational numbers. We will be commonly using $\mathbb{R}$, $\mathbb{N}$, and $mathbb{Z}$ to represent scalars.
- $\mathbb{R}^D$ means a $D$-dimensional column vector where the entries are real-valued so if $x \in \mathbb{R}^D$, then $x = (x_1, ..., x_D)$ where each $x_i \in \mathbb{R}$.
- $M_{r \times c}(\mathbb{R})$ means a matrix with $r$ rows and $c$ columns where the entries are real-valued.

---

### Introduction

The process of training neural networks is called optimization and the specific optimization methods are called optimizers.
There are many optimization methods, but today, I will be covering the most basic and popular, gradient descent.

So, how does neural networks perform optimization?
In mathematical terms, all neural networks are essentially trying to learn some mapping $f_{\theta}: x \rightarrow y$
that minimizes a loss $L(\theta)$ where $x$ is the input, $y$ is the output, and
$\theta$ are the parameters of the neural networks that needs to be learned or optimized.

For an example, take a simple model, the linear regression model.
For $x \in \mathbb{R}^D$, the linear regression model is:

$$\hat{y}
= f(x)
= w^T x + b$$

For a linear regression model, the mapping is $f_{\theta} = w^T x + b$ where the parameters that need to be learned
is $\theta = \{ w, b \}$ where $w \in \mathbb{R}^D$ and $w \in \mathbb{R}$.
The most common loss that linear regression minimizes is mean squared error (MSE).
Suppose our data is $(x_1,y_1), ..., (x_n,y_n)$, then MSE loss is defined:

$$L(\theta)
= \frac{1}{n} \sum_{i=1}^n (y_i - f(x_i))^2
= \frac{1}{n} \sum_{i=1}^n (y_i - [w^T x_i + b])^2$$

where the values for $\theta = \{ w, b\}$ needs to be determined that minimizes $L(\theta)$.
This may seem difficult, but in fact, the equations that determine the values for $\theta$ are known and quite simple actually.
But for a neural network, the model $f_{\theta}$ is NOT simple so how do we minimize $L(\theta)$ in general?

In fact, optimization for neural networks is still complicated.
Before we learn how neural networks fully perform optimization, let's start with a simple example for motivation.
Suppose we want to minimize $f(x) = x^2$.

- If I start at $x=2$, then $f(2) = 4$. To make $f(x)$ smaller, $x$ should decrease say to $x=1$ so $f(1)=1$.
- If I start at $x=-2$, then $f(-2) = 4$. To make $f(x)$ smaller, $x$ should increase say to $x=-1$ so $f(-1)=1$.

It should be obvious that $f(x) = x^2$ minimizes when $x = 0$ with $f(0) = 0$.
How can we generalize this behavior?

USE THE DERIVATIVE!!!
You may have heard before that the derivative is the direction of steepest ascent.
The reason is because the derivative always points in the direction where $f$ increases the most
(in fact this is true even for equations with multiple variables like $f(x,y) = x^2 + y^2$ where
the derivative is instead called the gradient).
But wait, we want $f$ to decrease so instead of moving with the derivative,
we need to move in the opposite direction of the derivative!

The derivative of $f(x) = x^2$ is $f'(x) = 2x$ so we want to move instead $-f'(x) = -2x$.
This means if $x > 0$ like $x = 2$, we need to move $-f'(2) = -4$ from $x = 2$.
Ignore the 4 for now, the point is that at $x=2$, the direction we need to move is negative (i.e. left).
A similar logic applies when $x < 0$ like $x = -2$ where the direction we need to movve is positive (i.e. right).
Finally, when $x = 0$, the derivative $f'(0) = 0$ so we don't have to move anywhere because we REACHED our minimum.

---

### Gradient Descent

As shown in the previous example, the minimum can be reached by going in the opposite direction
of the derivative (for one variable) or gradient (for multiple variables).
This is called gradient descent (GD). The formula for gradient descent to minimize $f(x)$:

$$x_{t+1} = x_t - \eta_t [\frac{\partial f}{\partial x}]_{x = x_t}$$

where

- $x_t \in \mathbb{R}^D$ is the current value
- $x_{t+1} \in \mathbb{R}^D$ is the next value
- $[\frac{\partial f}{\partial x}]_{x = x_t} \in \mathbb{R}^D$ is the gradient of $f(x)$ in respect to $x$ evaluated at $x = x_t$
- $\eta_t$ is called the learning rate

As shown in the previous example, the derivative tells us where to move, NOT how much to move
so the learning rate $\eta_t$ is a multiplier to control how much to move.
If the learning rate is too big, $x_t$ might never converge to the minimum,
but if learning rate is too small, $x_t$ will converge very slowly.

Before we continue any further, it should be noted that optimization methods MAY NOT lead to the smallest
possible value for $f(x)$. There exist equations $f(x)$ where there are multiple points $x$ that reach a small value
for $f$ near $x$, which are called local minimums. There is not a clear way in general to determine the smallest among
these local minimums so keep in mind that optimization methods are only finding local minimums.

---

### Gradient Descent Example

Let's try a simple example with gradient descent: $f(x,y) = x^2 + y^2$.

Let start by choosing a random value for $(x,y)$ say $x_0 = (5,-3)$ so $f(5,-3) = 34$.
For our current knowledge, let's keep the learning rate simple at a constant $\eta_t = 0.1$.
Let's perform gradient descent!

The gradient of $f(x,y)$ is:

$$\nabla f = (\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}) = (2x, 2y)$$

- At iteration $t = 1$, our previous $x_0 = (5,-3)$ so $\nabla f = (10, -6)$.
Thus, $x_1 = (5,-3) - 0.1 (10, -6) = (4, -2.4)$. Note $f(4, -2.4) = 21.76$.

- At iteration $t = 2$, our previous $x_1 = (4, -2.4)$ so $\nabla f = (8, -4.8)$.
Thus, $x_2 = (4, -2.4) - 0.1 (8, -4.8) = (3.2, -1.92)$. Note $f(3.2, -1.92) \approx 13.93$.

- At iteration $t = 3$, our previous $x_2 = (3.2, -1.92)$ so $\nabla f = (6.4, -3.84)$.
Thus, $x_3 = (3.2, -1.92) - 0.1 (6.4, -3.84) = (2.56, -1.536)$. Note $f(2.56, -1.536) \approx 8.91$.

Performing more and more iterations, gradient descent should converge
closer and closer to $(0,0)$ with $f(0,0) = 0$ as the minimum.

---

### Gradient Descent with Neural Networks

So far, all the examples covered with gradient descent has been with simple functions $f(x)$.
How do we perform gradient descent with neural networks?
Well, exactly the same way! ... although a bit more complicated.

Suppose our neural network is defined:

$$f(x; \theta) = f_L(x; \theta_L)$$

where

- Pre-activation: $z_{\ell} = W_{\ell} f_{\ell-1}(x; \theta_{\ell-1}) + b_{\ell}$
- Neuron: $f_{\ell}(x; \theta_{\ell}) = \phi_{\ell}(z_{\ell}; V_{\ell})$

And suppose we want to minimize MSE loss $L(\theta) = \frac{1}{n} \sum_{i=1}^n (y_i - f_L(x_i))^2$
where our data is $(x_1, y_1), ..., (x_n, y_n)$ and the parameters we want to learn
are $\theta_{\ell} = \{ W_{\ell}, b_{\ell} \}$ for $\ell = 1, ..., L$.
The process of inputting an observation $(x_i, y_i)$ through the model is called forward propagation.
After forward propagation is done, we need to minimize the loss by updating each parameter using
an optimization method. Using gradient descent for EACH parameters $W_{\ell}$ and $b_{\ell}$ where:

$$(W_{\ell})_{t+1} = (W_{\ell})_t - \eta_t [\frac{\partial L}{\partial W_{\ell}}]_{W_{\ell} = (W_{\ell})_t}$$

$$(b_{\ell})_{t+1} = (b_{\ell})_t - \eta_t [\frac{\partial L}{\partial b_{\ell}}]_{b_{\ell} = (b_{\ell})_t}$$

$\frac{\partial L}{\partial W_{\ell}}$ may seem simple, but in fact, $L(\theta)$ is not directly connected to
$W_{\ell}$ so you need to apply the derivative chain rule to get from $L$ to $f_L$ to $z_L$ to $f_{L-1}$
to etc until you reach $z_{\ell}$, which is finally directly connected to $W_{\ell}$:

$$\frac{\partial L}{\partial W_{\ell}}
= \frac{\partial L}{\partial f_L} \frac{\partial f_L}{\partial z_L} \frac{\partial z_L}{\partial f_{L-1}}
\frac{\partial f_{L-1}}{\partial z_{L-1}} \frac{\partial z_{L-1}}{\partial f_{L-2}}
... \frac{\partial z_{\ell}}{\partial W_{\ell}}$$

This is called backpropagation since you need to compute the gradients starting at the end of the neural network
and then, compute the gradient one after another as you move back one layer after another. Finally,
when you reach the layer where the parameter is that is being updated, all the gradients are multiplied
and the parameter is updated by that value.
Backpropagation also needs to be applied for $\frac{\partial L}{\partial b_{\ell}}$.
This may seem complicated, which it is for humans, but this is all computed quite easily by computers.
This process of repeating forward and backpropagation over and over trains the model by gradually minimizing the loss.

Well, that's basically it for how feed-forward neural networks are optimized, just gradient descent and backpropagation.
Note gradient descent is only the most basic optimizer.
Other optimizers have been developed that have shown to optimize neural networks more efficiently.

---

### Bonus: Stochastic Gradient Descent

A variant of gradient descent is stochastic gradient descent (SGD).
Optimization methods are minimizing loss $L(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(y_i, f_L(x_i))$
where our data is $(x_1, y_1), ..., (x_n, y_n)$ and $\ell(y_i, f_L(x_i))$ is some loss function
like MSE $\ell(y_i, f_L(x_i) = (y_i - f_L(x_i))^2$.
Rather than using the entire dataset, which can be computationally expensive since
the gradient needs to be computed for each observation $(x_i, y_i)$,
SGD instead only chooses one observation to use. So, SGD is optimizing:

$$L(\theta) = \ell(y_i, f_L(x_i))$$

for some chosen $(x_i,y_i)$.
Obviously this is a terrible metric since the parameters are being trained to fit one observation
so if two observations are vastly different, then parameters will at best take forever to converge
or at worst, never converge.
Thus, SGD is generally implemented using a small batch of $M$ randomly selected observations from
$(x_1, y_1), ..., (x_n, y_n)$ where $M \ll n$.
This version of SGD is called miniBatch SGD, which is very commonly used especially for
large neural networks like Large Language Models (LLMs).
If the minibatch is $B$, then miniBatch SGD minimizes:

$$L(\theta) = \frac{1}{|B|} \sum_{i \in B} \ell(y_i, f_L(x_i))$$

Simply, minibatch SGD means to only train a model with a small batch of randomly selected data
out of the entire dataset to make training the model much less computationally expensive.

