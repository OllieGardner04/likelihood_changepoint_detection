# likelihood_changepoint_detection
An algorithm for detecting a changepoint in data where the underlying structure is known, but the changepoint is not.

## The general idea

Assume we have data $x$ and $y$, and we know that, at some value of $x$, the data generating process of $y$ changes. In this model, we know the underlying data structure, meaning we know the relationship between $x$ and $y$. What we do not know however, is the point at which the underlying structure changes. Let the underlying data generating process be given by

$$g(x) = \begin{cases}
	g_{1}(x), & \min(x) \leq z \leq \theta, \\
	g_{2}(x), & \theta < z \leq \max(x).
\end{cases}$$
