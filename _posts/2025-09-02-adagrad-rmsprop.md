## AdaGrad and RMSProp

---

### Preface

Some mathematical notations to know:

- $\mathbb{R}$ means a real-valued scalar like $5$, $2.5$, $\pi$, etc. Other scalars are $\mathbb{N}$ for the natural mumbers, $\mathbb{Z}$ for integers, and $\mathbb{Q}$ for rational numbers. We will be commonly using $\mathbb{R}$, $\mathbb{N}$, and $mathbb{Z}$ to represent scalars.
- $\mathbb{R}^D$ means a $D$-dimensional column vector where the entries are real-valued so if $x \in \mathbb{R}^D$, then $x = (x_1, ..., x_D)$ where each $x_i \in \mathbb{R}$.
- $M_{r \times c}(\mathbb{R})$ means a matrix with $r$ rows and $c$ columns where the entries are real-valued.

---

### AdaGrad

One issue with optimization methods like
<a href="/2025/09/02/gradient-descent.html"> gradient descent, SGD </a>,
and <a href="/2025/09/02/momentum.html"> momentum </a>
is the learning rate $\eta_t$ needs to be manually chosen.
As stated in the post for gradient descent, if the learning rate is too big, the parameters might never converge
and if the learning rate is too small, the parameters are slow to converge.

One solution to manually specifying the learning rate is the adaptive gradient (AdaGrad) optimization method.
The idea is simple: As the number of iterations increase, the learning rate should decrease since the parameters
are generally do not have to change as much as previous iterations due to the parameters getting closer to
their minimizers.
Additionally, AdaGrad also updates its parameters elementwise.
The equation for AdaGrad is:

$$x_{t+1,j} = x_{t,j} - \frac{\eta_t}{\sqrt{\sum_{\tau=0}^t g_{\tau,j}^2 + \epsilon}} g_{t,j}$$

where $x \in \mathbb{R}^D$ so $j \in \{1, ..., D\}$ and:

- $g_{t,j} = [\frac{\partial f}{\partial x_j}]_{x_j = x_{t,j}}$
- $\epsilon$ is a small value to avoid division by 0 like $10^{-10}$

In vector form:

$$x_{t+1}
= \begin{bmatrix}
  x_{t+1,1} \\
  \vdots \\
  x_{t+1,D}
\end{bmatrix}
= x_t - \eta_t
\begin{bmatrix}
  \frac{1}{\sqrt{\sum_{\tau=0}^t g_{\tau,1}^2 + \epsilon}} & ... & 0 \\
  \vdots & \ddots & \vdots \\
  0 & ... & \frac{1}{\sqrt{\sum_{\tau=0}^t g_{\tau,D}^2 + \epsilon}}
\end{bmatrix}
g_t$$

where

$$g_t
= \begin{bmatrix}
  g_{t,1} \\
  \vdots \\
  g_{t,D}
\end{bmatrix}_{x = x_t}
= \begin{bmatrix}
  \frac{\partial f}{\partial x_1} \\
  \vdots \\
  \frac{\partial f}{\partial x_D}
\end{bmatrix}_{x = x_t}$$

One issue with AdaGrad is $\frac{1}{\sqrt{\sum_{\tau=0}^t g_{\tau,j}^2 + \epsilon}}$
is monotonically decreasing at the number of iterations $t$ increases.
This means that convergence becomes very slow over time.

---

### RMSProp

Optimization method RMSProp addresses the monotone increasing issue of AdaGrad
by using an exponentially weighted average of the squares of the past gradients
instead of summing the squares of the past gradients in AdaGrad.

$$s_{t+1,j} = \beta s_{t,j} + (1 - \beta) g_{t,j}^2$$

$$x_{t+1,j} = x_{t,j} - \frac{\eta_t}{\sqrt{s_{t+1,j} + \epsilon}} g_{t,j}$$

Note

$$s_{t+1,j}
= \beta s_{t,j} + (1 - \beta) g_{t,j}^2
= \beta [\beta s_{t-1,j} + (1 - \beta) g_{t-1,j}^2] + (1 - \beta) g_{t,j}^2
= ...
= (1-\beta) \sum_{\tau=0}^t \beta^{\tau} g_{t-\tau,j}^2$$

In vector form:

$$x_{t+1}
= \begin{bmatrix}
  x_{t+1,1} \\
  \vdots \\
  x_{t+1,D}
\end{bmatrix}
= x_t - \eta_t
\begin{bmatrix}
  \frac{1}{\sqrt{(1-\beta) \sum_{\tau=0}^t \beta^{\tau} g_{t-\tau,1}^2 + \epsilon}} & ... & 0 \\
  \vdots & \ddots & \vdots \\
  0 & ... & \frac{1}{\sqrt{(1-\beta) \sum_{\tau=0}^t \beta^{\tau} g_{t-\tau,D}^2 + \epsilon}}
\end{bmatrix}
g_t$$

Since past gradients are exponentially weighted,
then $\frac{1}{\sqrt{(1-\beta) \sum_{\tau=0}^t \beta^{\tau} g_{t-\tau,j}^2 + \epsilon}}$
is not monotonically decreasing so convergence over time does not necessarily decrease.

Note both AdaGrad and RMSProp still include a learning rate $\eta_t$ so the learning rate can still be user controlled.
Essentially, AdaGrad and RMSProp are just scaling the learning rate down based on the size of the gradients.

---

### AdaGrad or RMSProp with Neural Networks

Suppose our neural network is defined:

$$f(x; \theta) = f_L(x; \theta_L)$$

where

- Pre-activation: $z_{\ell} = W_{\ell} f_{\ell-1}(x; \theta_{\ell-1}) + b_{\ell}$
- Neuron: $f_{\ell}(x; \theta_{\ell}) = \phi_{\ell}(z_{\ell}; V_{\ell})$

And suppose we want to minimize MSE loss $L(\theta) = \frac{1}{n} \sum_{i=1}^n (y_i - f_L(x_i))^2$
where our data is $(x_1, y_1), ..., (x_n, y_n)$ and the parameters we want to learn
are $\theta_{\ell} = \{ W_{\ell}, b_{\ell} \}$ for $\ell = 1, ..., L$.

Recall that gradient descent for feed-forward neural networks for EACH parameters $W_{\ell}$ and $b_{\ell}$ are:

$$(W_{\ell})_{t+1} = (W_{\ell})_t - \eta_t [\frac{\partial L}{\partial W_{\ell}}]_{W_{\ell} = (W_{\ell})_t}$$

$$(b_{\ell})_{t+1} = (b_{\ell})_t - \eta_t [\frac{\partial L}{\partial b_{\ell}}]_{b_{\ell} = (b_{\ell})_t}$$

To change gradient descent to instead AdaGrad or RMSProp is simple:

$$(W_{\ell})_{t+1,i,j}
= (W_{\ell,i,j})_t - \eta_t v_{W_{\ell,i,j}} [\frac{\partial L}{\partial W_{\ell,i,j}}]_{W_{\ell,i,j} = (W_{\ell,i,j})_t}$$

$$(b_{\ell})_{t+1,i}
= (b_{\ell,i})_t - \eta_t v_{b_{\ell,i}} [\frac{\partial L}{\partial b_{\ell,i}}]_{b_{\ell,i} = (b_{\ell,i})_t}$$

where $v_{W_{\ell,i,j}}$ and $v_{b_{\ell,i}}$ are the corresponding values as defined above for
AdaGrad and RMSProp's scaled learning rate.

