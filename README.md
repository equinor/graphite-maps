# graphite-maps
Graph informed triangular ensemble-to-posterior maps

## Row Solver: Iterative vs. Closed-Form

To estimate the precision matrix, we need to find the non-zero coefficients of the lower triangular matrix $C$. This is done row by row. For a given row, the objective function to minimize is:

$$
f(c)
= \frac{1}{2}\lVert U_{\mathrm{reduced}} c \rVert_2^2
	- n\log(c_d)
	+ \frac{\lambda}{2}\lVert c_{\mathrm{off}}\rVert_2^2,
$$

where

- $c = [c_{\mathrm{off}}, c_d]$ is the vector of non-zero coefficients for the row,
- $c_d > 0$ is the diagonal coefficient,
- $c_{\mathrm{off}}$ are the off-diagonal coefficients,
- $U_{\mathrm{reduced}}$ is the data matrix restricted to the non-zero columns of the row,
- $\lambda$ is the L2 regularization weight,
- $n$ is the number of samples.

### The Iterative Solution

The iterative solver (`solver="iterative"`) minimizes this objective function directly using a general-purpose numerical optimization algorithm, such as L-BFGS-B (via `scipy.optimize.minimize`).

In this approach, the solver starts with an initial guess for $c$ and iteratively updates it by evaluating the objective function and its gradient at each step. While this method is robust, it requires many iterations and repeated matrix-vector multiplications to converge to the optimal coefficients, which can be computationally expensive for large problems.

### The Closed-Form Solution

The closed-form solver (`solver="closed_form"`) provides a faster alternative. This approach is fundamentally possible because the problem is decoupled row by row, leaving only a single logarithmic term ($-n\log(c_d)$) in the objective. This allows us to factor out $c_d$ and reparameterize the off-diagonal elements.

The key idea is to rewrite the objective so that, for a fixed diagonal coefficient, the off-diagonal coefficients appear in a quadratic form. That lets us apply the standard ridge-regression normal equations directly, then solve a remaining 1D problem for the diagonal term.

Here is the derivation in that order:

First split the matrix-vector product by columns:

$$
U_{\mathrm{reduced}}c
= U_{\mathrm{reduced}}[:, :-1]c_{\mathrm{off}} + U_{\mathrm{reduced}}[:, -1]c_d.
$$

Now define

$$
X = U_{\mathrm{reduced}}[:, :-1], \qquad z = U_{\mathrm{reduced}}[:, -1],
$$

so that

$$
U_{\mathrm{reduced}}c = X c_{\mathrm{off}} + z c_d.
$$

To expose the quadratic ridge-regression structure, reparameterize the off-diagonal coefficients by factoring out $c_d$:

$$
\beta = -\frac{c_{\mathrm{off}}}{c_d}.
$$

Then $X c_{\mathrm{off}} + z c_d = c_d(z - X\beta)$, and the objective becomes

$$
f(\beta, c_d)
= \frac{1}{2}c_d^2\lVert z - X\beta\rVert_2^2
	- n\log(c_d)
	+ \frac{\lambda}{2}c_d^2\lVert\beta\rVert_2^2.
$$

For any fixed $c_d > 0$, we can factor out $\frac{1}{2}c_d^2$ from the terms involving $\beta$. Because scaling by a positive constant and adding a constant ($-n\log(c_d)$) does not change the location of the minimum, minimizing over $\beta$ is equivalent to minimizing the inner quadratic form $\lVert z - X\beta\rVert_2^2 + \lambda \lVert\beta\rVert_2^2$. This gives the standard ridge regression normal equations:

$$
(X^\top X + \lambda I)\tilde\beta = X^\top z.
$$

Denote $\tilde\beta$ as this optimal solution. We can then evaluate the quadratic part of the objective at $\tilde\beta$. Let

$$
\alpha = \lVert z - X\tilde\beta\rVert_2^2 + \lambda\lVert\tilde\beta\rVert_2^2.
$$

By expanding $\lVert z - X\tilde\beta\rVert_2^2$ and using the normal equations $X^\top X \tilde\beta + \lambda \tilde\beta = X^\top z$, we can simplify $\alpha$:

$$
\begin{aligned}
\alpha &= z^\top z - 2 z^\top X \tilde\beta + \tilde\beta^\top X^\top X \tilde\beta + \lambda \tilde\beta^\top \tilde\beta \\
       &= z^\top z - 2 z^\top X \tilde\beta + \tilde\beta^\top (X^\top X + \lambda I) \tilde\beta \\
       &= z^\top z - 2 z^\top X \tilde\beta + \tilde\beta^\top (X^\top z) \\
       &= z^\top z - z^\top X \tilde\beta \\
       &= \lVert z\rVert_2^2 - (X^\top z)^\top\tilde\beta.
\end{aligned}
$$

By factoring out $\frac{1}{2}c_d^2$ from the original objective function, we get:

$$
f(\beta, c_d) = \frac{1}{2}c_d^2 \left( \lVert z - X\beta\rVert_2^2 + \lambda\lVert\beta\rVert_2^2 \right) - n\log(c_d).
$$

Substituting the optimal $\tilde\beta$ back into this equation, the term in the parentheses is exactly our definition of $\alpha$. This reduces the objective to a function of $c_d$ alone:

$$
f(c_d) = \frac{1}{2}\alpha c_d^2 - n\log(c_d).
$$

Taking the derivative with respect to $c_d$ and setting it to zero yields the first-order condition:

$$
f'(c_d) = \alpha c_d - \frac{n}{c_d} = 0,
$$

which gives the optimal diagonal coefficient:

$$
c_d^2 = \frac{n}{\alpha} \implies c_d = \sqrt{\frac{n}{\alpha}}.
$$

Finally, we recover the off-diagonal coefficients using our reparameterization:

$$
c_{\mathrm{off}} = -c_d\tilde\beta.
$$

This matches the implementation steps in `_solve_row_closed_form`.
