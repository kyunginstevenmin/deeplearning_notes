---
Class: Intro to Deep Learning
Title: L6 Optimization
Topics:
  - Optimization
  - DeepLearning
  - machine Learning
tags:
  - "#convergence"
---
Training Part IV
- Convergence issues
- Loss Surfaces
- Momentum
# Convergence
## Optimal step size for convex quadratic functions is used to get rough idea of optimal step size.
**Newton's Method:**
	Optimal step size in a convex quadratic function is given by 1/second derivative at that point. This is known as the Newton's method. The second derivative gives us information about how the gradient is expected to change. This allows us to find optimal step size to the minimum. 
		For intuition based on physics, its like the acceleration, or gravity, that gives us information about the behavior of the curve. Like the curve of a projection thrown in the air. The first derivative gives us speed. Second derivative gives us change in speed over time, how gradient changes w.r.t. time. 
		This information is needed to find where the first derivative is going to hit 0. Because the second derivative is the change in first derivative w.r.t. w. (again with physics, second derivative is acceleration. First derivative is speed. To know the maximum of height of project, we need to know location, current speed, and acceleration. We know where speed is going to = 0 if we know acceleration.)
	**Why 1/second derivative makes sense?**
	It makes sense to have step size 1/second derivative. Because larger the second derivative, the faster the gradient is changing, meaning that its a steep and narrow curve. So you would want the step size to be proportionally smaller relative to how fast the gradient changes.

**Work through quadratic approximation and finding optimal step size:**
For any quadratic function, we can find the minimum by finding where the derivative = 0.
For a quadratic function whose equation that we don't know, we can estimate the function as a quadratic function using Taylor series. 
The equation for a Taylor's approximation of a quadratic function is:
![[Pasted image 20240523074937.png]]
Under the assumption of a quadratic loss function, we compute the Taylor's series approximation at point wk. 
We know how to minimize a quadratic function. Find derivative w.r.t. w and set it = to 0, and solve for w.
![[Pasted image 20240523075217.png]]
This means w such that dE/dW = 0 is E''(w)-1 step away from wk. Current wk - E''(w)-1 times the derivative E'(wk) gives us wmin. Therefore the optimal step size from wk to w min is given by E''(w)-1.
![[Pasted image 20240523075451.png]]
**Optimal step size is dependent on the location on the function, since its equal to the second derivative of a function. Thus calculate second derivative after every step to identify optimal step size.
## The behavior of non-optimal step sizes under convex quadratic assumptions:
![[Pasted image 20240523075553.png]]
	a) if n < nopt: monotonic convergence to wmin. Doesn't overshoot minimum. Gradient doesn't change sign.
	b) n = nopt: single step convergence to w min.
	c) nopt< n < 2nopt:  Oscillating convergence of w. The derivatives changes sign every step. But loss should decrease every step.
	d) n > 2nopt: Divergence. We escape the parabola, since this is convex quadratic, the step takes us to a higher loss than our initial loss.

## Multivariate function minimization. Vector Update rule:
For Loss functions with multivariate inputs, we will look at case where w1 and w2 are uncoupled. They don't have interaction effects. In such a case, the Loss w.r.t. each w are independent from each other. 
![[Pasted image 20240523085353.png]]
- Loss is quadratic along both axis. But steepness is different for each axis. This means the optimal step size is also different for each w.
- The axis with narrower contour line has steeper gradient, and thus have smaller optimal step size.
- But step size is uniform for both directions, which causes problems.
![[Pasted image 20240523085051.png]]
**Illustration of problem taking equal step size.**
- The figure on the left is narrower, and has smaller step size.
- The figure on the right is wider and has wider step size.
- Taking optimal step size for right curve means we will overshoot for curve on left.

**Illustration of step size**
![[Pasted image 20240523090012.png]]
a) n = 2.1xn2opt = 0.7. This means we diverge w.r.t. x2(horizontal). but w.r.t. x1, we monotonically converge.
b) n = 2xn2opt = 0.66. We osciliate w.r.t. x2. We converge monotonically w.r.t. x1 because the step size is less than the optimal step size of n1.
c) We osciliate but converge w.r.t. x2 because n is greater than n2opt but less than twice the n2opt. We converge monotonically for x1.
d) We converge at one step for x2. But convergence is slow for x1, since the learning rate is a third of the optimal learning rate of x1.
e) We converge monotonically for both x1 and x2 since learning rate is less than the optimal learning rate for both x1 and x2.

**Optimal learning rate for vector updates: n should be less than twice the smallest optimal learning rate. Or else, we will diverge w.r.t. xi with smallest learning rate. 
The speed of convergence thus depends on the ratio of max(optimal learning step)/min(optimal learning step**
	If = 1, then fast convergence.
	Because we are limited by the smallest optimal learning step, in converging in the direction with greatest optimal learning step.

## Learning rate that leads to divergence is not necessarily bad. Its good for escaping local optimums.
**Decaying learning rate is used with high initial learning rate. We escape local optimums with intial learning rate >2nopt. Iteratively reduce learning rate.**
This works because good global minimum functions are 'usually' deep and wide. Meaning even with learning rate >2, its more difficult to escape than small local optimums. So using large learning rate, you have more chance of escaping local optimum and escaping global optimum. 

**Types of decaying learning rates**
![[Pasted image 20240523091606.png]]
**Q. so step 1 means that we find a potential deep global minimum. Then we optimize within that global minimum?****

## Convergence: Story so far
- Gradient descent can overshoot local minimums where d/dx = 0, but this is good.
- Other convergence issues:
	- Loss function have saddle points. Saddle points are where the derivative = 0, but is not a minimum. 
	- ![[Pasted image 20240523112641.png]]
- Vanilla gradient descent: Taking single and uniform step to steepest down slope.
	- Equal step for all dimensions means step is too small for some directions, too large for other directions.
	- Can we take different size step for each direction?
## Derivative inspired algorithms: Rprop, Quick Prop
**Previously we altered step sizes so that we can converge to global minimum. We can also alter the derivatives.**
### RProp: Resillient propagation
1. We take steps independently for each dimension instead of taking uniform step for all directions.
2. advantage: simple. Makes no assumption of convexness of loss function.
Description:
1. start at intial w. Find derivative and take step.
2. At the new w, calculate the sign of the derivative. 
	1. If the sign is the same as the previous derivative, take a wider step(scaled derivative calulated at step 1). 
	2. repeat with wider step while derivative at new w is still the same sign.
3. If the derivative changes sign, return to previous w. scale Down the step by b.
![[Pasted image 20240523122115.png]]
**Pseudo code**
![[Pasted image 20240523122213.png]]
### Momentum methods:
- This method keeps track of overshoots in updates by calculating running average of previous gradients. Oscillations means the gradient keeps changing sign. Use this information to adjust gradient. 
	- So if gradient was positive, then becomes negative, next step should incorporate this information, by using a scaled average of the two gradients.
![[Pasted image 20240523122447.png]]
## Nestorov's Accelerated Gradient
- This is similar to momentum method but changes order of operation. Instead of calculating gradient at current location, then added scaled previous step, we:
	1.  take the scaled previous step.
	2. calculate the gradient at the new location 
	3. add previously scaled step and gradient which gives us our final step.
![[Pasted image 20240523123242.png]]
- loss is calculated not at w(k-1) but at w(k-1) + B x deltaW(k-1).

