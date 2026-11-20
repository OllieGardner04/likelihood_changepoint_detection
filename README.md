# likelihood_changepoint_detection
An algorithm for detecting a changepoint in data where the underlying structure is known, but the changepoint is not.

## The general idea

\[g(z) = \begin{cases}
	g_{1}(z), & \min(z) \leq z \leq \theta_{1}, \\
	g_{2}(z), & \theta_{1} < z \leq \theta_{2}, \\
	\vdots \\
	g_{n}(z), & \theta_{n-1} < z \leq \max(z).
\end{cases}\]
