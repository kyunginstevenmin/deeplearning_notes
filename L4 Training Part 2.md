# Intro
1. We want to minimize the empirical error w.r.t. every parameter in the network.
2. Minimization of a function is an optimization problem.
	1. There can be analytic solutions to find where first derivative = 0 and second derivative is +. But often there is not analytic solution
3. Gradient descent algorithm is a way to experimentally find the minimum by iterative steps towards X at which f(X) decreases the most, which is the direction of the gradient.
	- The gradient is the transpose of the derivative.
	- Change in f(X) is the inner product of the gradient and change in X. This means that the change in f(X) is greatest when the angle between the gradient and change in X is 0, or when they are pointing towards the same direction (Given that magnitude is fixed). 
	- Thus gradient is the direction of X at which change in f(X) is the greatest.
4. The algorithm for gradient descent is given by:
# Contents:

## 1. We want to minimize the empirical error w.r.t. every parameter in the network.

## 2. Minimization of a function is an optimization problem.
	1. There can be analytic solutions to find where first derivative = 0 and second derivative is +. But often there is not analytic solution
### Notes on derivatives:
- Derivatives are functions of the variable: this means that derivatives changes w.r.t. x.
![[Pasted image 20240520105302.png]]

### Derivatives of vectors:
- derivatives of vectors are vectors of partial derivatives.
	- Each partial derivative shows how much f(x) changes when xi changes minutely while other xd are constant.
	- Total change in f(X) is given by product sum of derivative and change in x(vector).
	- ![[Pasted image 20240520105558.png]]
- Gradient is the transpose of the derivative. It gives direction of X at which f(X) increases or decreases the most.
	- We can see this through property of inner product.	


## 3. Properties of Gradient Descent
**Thus gradient is the direction of X at which change in f(X) is the greatest.**
- The gradient is the transpose of the derivative.
- Change in f(X) is the inner product of the gradient and change in X. This means that the change in f(X) is greatest when the angle between the gradient and change in X is 0, or when they are pointing towards the same direction (Given that magnitude is fixed). 
- Thus gradient is the direction of X at which change in f(X) is the greatest.
- This direction is also perpendicular to the level curve, which is the contour of x at which f(x) is the same.
![[Pasted image 20240520110031.png]]
	![[Pasted image 20240520110210.png]]
	


## 4. Gradient descent algorithm is a way to experimentally find the minimum by iterative steps towards X at which f(X) decreases the most, which is the direction of the gradient.
![[Pasted image 20240520110533.png]]

The algorithm for gradient descent is given by:
	- we are minimizing loss(wk) w.r.t. w. 
	- initialize variables: w and k
	- Criteria of convergence to minimum: We iterate while the change in loss(wk) from loss(w(k-1)) is greater than epsilon:
		- When the change in loss is less than tolerated amount epsilon, we reached minimum.
	- Direction of update: negative sign
		- when gradient is positive, we take step opposite of gradient, because want to decrease the function.
		- when gradient is negative, we take positive direction towards gradient.
		- hence the negative sign.
	- step size nk:
		- nk is a iteration dependent step size.
		- How is it iteration dependent?
	![[Pasted image 20240520104439.png]]

## 5. Problem set up
**Individual neurons are an activation function applied on a affine function of inputs.**
![[Pasted image 20240520151850.png]]
**Outputs can be scalars and also vectors. There can be vector activations that couples the outcome of the entire layer. Softmax is an example that is particularly useful for multi-class class. It calculates the probability of each outcome in a multiclass classification. The sum of probability of all classes sum to 1.**
- Exponential function is applied to each zi. This makes each zi positive.
- Dividing by sum of all zi for all i in layer normalizes the values, which gives yi, which is the probability of each uotcome. They sum to 1.
- Thus the output is a vector of probabilities
![[Pasted image 20240520152428.png]]

## 6. Divergence functions:
Divergence functions are the loss functions or the error functions that we are trying to minimize w.r.t. the parameters. There are multiple types. For regression, L2 divergence. For classification, KL divergence.
**L2 Divergence**
- Its the sum of squared difference of i'th dimension.
- The derivative is (y-d). if y>d, then the derivative of error is positive. Increasing y increases error.
- If derivative is negative, y<d. Then increasing y decreases error.
![[Pasted image 20240520154513.png]]
**KL divergence**
- for probability, KL divergence is used.
- its the -log(probability). Ranges from 0 to infinity.
- also known as the cross entropy
![[Pasted image 20240520154737.png]]
![[Pasted image 20240520154900.png]]

**KL divergence of multiclass classification**
- The divergence formula reduces to just -log(yc), where yc is the probability output of desired class c. 
	- This is the log probability for the desired class, c=1.
	- ranges from 0 when yc=1 to infinity when yc-> 1.
- The derivative = -1/yc.
	- The smaller the yc, the divergence, since p=1 is the desired outcome. 
	- increasing yc makes it closer to p=1, which reduces error, hence the derivative is negative.
	- even at yc=1, which is minimum divergence, the derivative is not 0.
![[Pasted image 20240520155351.png]]
