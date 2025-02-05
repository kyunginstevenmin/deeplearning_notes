Notes on SGD: [[Back Propagation]], [[Stochastic Gradient Descent]]



## RMS Prop:
- more oscillating gradient directions have larger total moments.
- scale down learning rates for parameters with large mean squared derivatives.
    - mean square derivatives is the running estimate of average squared derivative.
**Procedure:**
- maintain running estimate of mean squared value of derivative of each parameter.
- scale learning rate by inverse of root mean square derivative.
![[Pasted image 20240530213855.png]]
**More notes on RMS prop:**
- Its an extension of Adagrad, which scaled the learning rate differently for each parameter according the sum of gradient squared. This addressed the problem of equal step size for every parameter causing divergence, since the optimal learning rate is different for each parameter. But this shrinked the learning rate too fast before convergence to a good local minima. So instead of using the sum of squared derivatives, we use the running average.
-
> RMSProp uses an exponentially decaying average to discard history from the extreme past so that it can converge rapidly after finding a convex bowl, as if it were an instance of the AdaGrad algorithm initialized within that bowl. — Page 308, [Deep Learning](https://amzn.to/3qSk3C2), 2016.


## ADAM: RMSprop with momentum

RMS adjust learning rate according to inverse of root mean squared derivative (penalizes total movement of gradient/oscillation).
Momentum: adjusts gradient according to running average of previous gradient.
- can overcome saddle points.
- prevent oscilations.
**ADAM combines both:**
![[Pasted image 20240530221151.png]]
