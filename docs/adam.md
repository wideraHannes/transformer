## Momentum

> Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations

It does this by adding a fraction $\beta$ of the update vector of the past time step to the current update vector.
with $\beta=0.9$ its the same as averaging over the last 10 days temperatur.
So in this case remember the last 10 parameter updates to accelerate on steep slopes.
Analogy: Ball rolling down a hill. Accelerates.

$$

	\begin{align}
	v_{dW}&=\beta*v_{dW}+(1-\beta)*dW \\
	v_{db}&=\beta*v_{db}+(1-\beta)*db \\
	W &= W - \alpha*v_{dW}, b=b-\alpha*v_{db}
	\end{align}


$$

## RMSprop

(Root Mean Square Prop)

![alt text](<./assets/Bildschirmfoto 2024-12-11 um 15.37.17.png>)

- We wanna slow down the learning in the b direction
- and accelerate in the w direction
  -> so far the same reasoning Momentum.

**Implementation details:**
On iteration t:
Compute dW, db, on the current mini-batch.

$$

	\begin{align}
	S_{dW}&=\beta*S_{dW}+(1-\beta)*dW^2 \\
	S_{db}&=\beta*S_{db}+(1-\beta)*db^2 \\
	W &= W - \alpha \cdot \frac{dW}{\sqrt{S_{dW}}}, b=b-\alpha \cdot \frac{db}{\sqrt{S_{db}}}
	\end{align}


$$

now we calculate $S_{dW}$ and square d*w and d_b
The Updating is also distinct to Momentum with the fraction.
b direction we wanna slow down and w direction we wanna speed up.
so we want: $S*{dW}$ is small and $S_{db}$ ist big

#### (hint) Bias Correction in exponentially weighted average

With Bias Correction we can make exponentially Weighted average more accuratly.

with this formula
$v_{t}=\beta*v_{t-1}+(1-\beta)*\sigma_t$
we have a bad starting point of the final Temperature plot because we dont have any previous knowledge.

So during this initial phase of 10-20 days we need some improvent.
Devide $v_t$ by the bias correction term:

$\frac{v_{t}}{1-\beta^t}$

This will remove the bias and as the t goes higher also the denominator goes away.

## Adam

> Putting Momentum and RMSPRop together

$V_{dw}=0, S_{dw}=0, V_{db}=0, S_{db}=0$
**Implementation details:**
On iteration t:
Compute dW, db, on the current mini-batch.

$$

	\begin{align}
	Momentum: v_{dW}&=\beta_1*v_{dW}+(1-\beta_{1)*dW},\quad v_{db}=\beta_1*v_{db}+(1-\beta_{1)*db } \\
	RMSprop: S_{dW}&=\beta_2*S_{dW}+(1-\beta_2)*dW^{2},\quad S_{db}=\beta_2*S_{db}+(1-\beta_2)*db^2 \\
	\end{align}


$$

Now We implement Bias Correction as above

$$
\begin{align}
V_{dw}^{corrected}&= \frac{v_{dw}}{1-\beta_1^{t}}, \quad
V_{db}^{corrected}= \frac{v_{db}}{1-\beta_1^{t}} \\
S_{dw}^{corrected} &= \frac{S_{dw}}{1-\beta_2^{t}}, \quad
S_{db}^{corrected}= \frac{S_{db}}{1-\beta_2^{t}} \\
\end{align}
$$

Finally we can perform the Update

$$
W = W - \alpha \frac{V_{dw}^{corrected}}{\sqrt{S_{dw}^{corrected} }+ \epsilon}, \quad b = b - \alpha \frac{V_{db}^{corrected}}{\sqrt{S_{db}^{corrected}}+ \epsilon}
$$

Hyperparameter Choice:

- $\alpha$: Needs to be tuned
- $\beta_1$ : 0.9 (dw)
- $\beta_{2}$: 0.999 (dw^2)
- $\epsilon$: $10^{-8}$

=> Generally we only tune \alpha

- Adam: Adaptive Moment Estimation

## Adam W

In summary, AdamW decouples weight decay from gradient updates, providing better regularization and often superior performance compared to the original Adam optimizer.

Why This Matters:
AdamW’s separation of weight decay avoids interference with the adaptive learning rates, making regularization more stable and effective. This often improves the model's ability to generalize to unseen data.

Key Takeaway:
AdamW is like Adam but better at regularization because it treats weight decay as an independent operation, not part of the gradient calculation. This seemingly small change leads to big improvements in performance and generalization.
