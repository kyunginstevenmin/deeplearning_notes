Topics:
- problem of learning
- perpectron rule for learning individual perceptrons
- greedy solutions for classification networks:
- learning through empirical risk minimization

### learning problem
1. Learning a neural network is learning the parameters and the bias of a function.
2. Assumption: the neural network has the correct architecture that is able to represent g(x).
3. We want to minimize the error of the real function and our neural network. But we don't have the real function. 
	1. This is the integral of the difference between the two functions.
![[Pasted image 20240519184953.png]]
4. Instead, we calculate the error on drawn samples. We minimize the error on the input-output pairs.
	1. The input and output pairs are sampled according to the distribution of P(X) by default usually.
	2. We minimize the Empirical error
![[Pasted image 20240519185215.png]]
5. Minimizing the empirical error on specific data points would hopefully mean that our functions would make good predictions at other X's.
6. Generalization: Also we assume generalization of learning from training to estimating true function, which we test be calculating error on unseen test data.
# learning through empirical risk minimization

**The activation function being smooth and continuous instead of being a threshold gives information about whether shifting the threshold towards a direction is good or not.**
- shifting the threshold for threshold activation doesn't give information about which direction to shift threshold.
![[Pasted image 20240519210155.png]]

## Two key requirements for learnability
1. **continuously varying activation: differentiable**
- If the activation function is continuous function of z, it is differentiable w.r.t. z. Z is differentiable w.r.t. w or x. Thus the entire network is differentiable to w or x, at any neurons.
- We can compute how y would change with respect to change in inputs or weights.
- This means we can also calculate how the error would change w.r.t. inputs or weights, given that the error function is continous/differentiable.
![[Pasted image 20240519210507.png]]

### continuously varying error function:
**Because we don't have access to the true g(x), we can't calculated the expected divergence. Instead we calculate the empirical estimate of the expected divergence. This is the average divergence over all samples.**
![[Pasted image 20240519213212.png]]


**Empirical Risk Minimization with respect to W.**
![[Pasted image 20240519213419.png]]
- we find parameter w value that minimizes the loss function, or the empirical risk. 
- empirical risk is a function of W for a given training set, because Xi is fixed for a training set.

### Summary:
- We learn networks by “fitting” them to training instances drawn from a target function.
	- We fit to training instances because we don't have the target function, but only samples.
	- We assume that fitting to samples will generalize in domains where we don't have samples, and to test samples.
- Learning networks of threshold-activation perceptrons requires solving a hard combinatorial-optimization problem - Because we cannot compute the influence of small changes to the parameters on the overall error
	- Threshold activation doesn't give directional information about shifting threshold.
- Instead we use continuous activation functions with non-zero derivatives to enables us to estimate network parameters 
	- This makes the output of the network differentiable w.r.t every parameter in the network
- The logistic activation perceptron actually computes the a posteriori probability of the output given the input 
- We define differentiable divergence between the output of the network and the desired output for the training instances 
	- And a total error, which is the average divergence over all training instances 
- We optimize network parameters to minimize this error 
	- Empirical risk minimization 
	- This is an instance of function minimization
