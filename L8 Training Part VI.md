---
Class: 
Title: 
Topics:
  - Optimization
  - regularization
  - divergence
  - batch normalization
  - dropout
tags:
  - deeplearning
  - "#dropout"
  - "#regularization"
  - "#batchnormalization"
---
## Choice of Divergence affects both learned network and results
### The optimal loss function w.r.t. parameter is bowl shaped.
**steeper far from minimum and flatter close to minimum**
- Flat gradient far from minimum is bad because it takes a long time to converge.
- Steep gradient close to minimum is bad because of overshoot.
- ![image](<Pasted image 20240528084841.png>)

### The shape of the loss function depends on what divergence function we use and what parameter we calculate the differentiate w.r.t.
![image](<Pasted image 20240528084906.png>)
- note: when we use different divergence as loss function, with sigmoid activation function.
- **The layers of activation function makes it difficult to predict shape of loss function w.r.t. parameters.**
	- L2 is better loss function w.r.t. y.
	- KL is better loss fucntion w.r.t. z. 

## The problem of covariate shifts:
When we use minibatches, each minibatch may not have identical distribution. 
The different in distribution is made worse by the layers of non-linear transformation.
Solution is two fold:
1. to move each batch to standard location so that all batches have mean = 0 and unit standard deviation.
2. move entire batch by scaling by gamma and adding beta.

## Batch Normalization
![image](<Pasted image 20240528090034.png>)
Normalization on each instances of batch:
**At what level:** Specific for each neuron.
	- normalizing each scalar feature independently.
	- for a layer with d-dimensional input, we normalize each dimension with mean and variance of minibatch.
	- we scale and shift normalized value of each activation. The parameters beta and gamma of each activation, is specific to the input. So just like the normalization is performed for each feature/dimension, the post normalization and shifting is also on the same scope.
**Timing of BN transformation:** After linear combinations of inputs and prior to activation function. 
**During training: batch normalize each batch separately.**

**During inference/testing: use the mean of batch means and mean of batch variance.**
	**Q. How do you calculate the mean of batch means?**
		A. Running Average: if alpha = 0.5, then arithmetic mean of batch.
		![image](<Pasted image 20240611105923.png>)
	**Q. Why would you use anything other than arithmetic mean of batch mean?**
		A. Because initial values of batch mean are batch mean of less trained network. The gamma and beta parameters are also less learned, including those of the previous layers. The mean of the batches will change as more training steps are taken. Why? because the parameters of previous layers change. This is internal covariate shift, which was the whole point of performing batch normalization. What mean and variance to use during inference must be decided.
		

**Pseudo code/instructions:**
For each instance zi:
1. Calculate the batch mean and batch variance. And normalize.
2. apply neuron specific transformation with scaling by gamma and adding beta.
	1. This is so that the input to the activation function would not be constrained by the normalization step, which shifts mean to 0 and variance to 1.
	2. It restores the representational power of the network.
	3. Does this just undo normalization?
		1. "Indeed, by setting γ(k) =sqrt(Var[x(k)]) and β(k) = E[x(k)], we could recover the original activations, if that were the optimal thing to do."


### Complications for derivatives:
**Usual derivatives are calculated as the empirical average of derivatives for all training samples.** This is because we can assume that each input of the mini-batch are independent from each other. Then the loss function and the derivatives are independent.
![image](<Pasted image 20240528090413.png>)


**Batch normalization means that each batch normalized instances are no longer independent from other inputs since they are normalized using the same transformation: batch mean and batch variance.**
	![image](<Pasted image 20240528090635.png>)

**Using chain rule to calculate gradient of Loss w.r.t. beta and gamma.**
	![image](<Pasted image 20240611110812.png>)
	- dLoss/dB = dLoss/dz: because Beta has power of 1. dz/dB = 1.
	- dLoss/dy = u of that training instance x dLoss/dz of that instance.
	


**When all instances of batch are identical or nearly identical, dLoss/dzi = 0. This means batch norm works well when the samples within batch are diverse.** 
- also doesn't work when batch size= 1. 
![image](<Pasted image 20240528090808.png>)

**Batch Norm for inference**
When we were training, we used batch specific mean and variance to normalize the training instances.
When we make predictions, we use the entire training set without batches. But we need to normalize the input because the parameters for individual neurons have been trained on normalized inputs.
**We use the average values of batch mean and batch variance over all batches.**
![image](<Pasted image 20240528091409.png>)
Note:
- these are neuron specific mean and variances used to normalize input to neurons.
- Ub and sigma B are from final converged network.
	- What does this mean? When the network is still converging, inputs to the neurons will continue to change as weights continue to change. 
	- When the network has converged, we will have constant inputs each layers of neuron.
	- This is when we calculate the mean of the batch and variance of the batch. Then calculate the mean of each of these.
	- We want to use the final converged network, because the we want to apply the same transformation on the test set as the transformation used when we trained the final converged parameters,

**Notes: copied from slide**
	- Batch normalization may only be applied to some layers – Or even only selected neurons in the layer 
	- Improves both convergence rate and neural network performance– Anecdotal evidence that BN eliminates the need for dropout 
	- To get maximum benefit from BN, learning rates must be increased and learning rate decay can be faster 
		- Since the data generally remain in the high-gradient regions of the activations
	- Also needs better randomization of training data order

## Overfitting and Regularization
![image](<Pasted image 20240528102303.png>)
- Large weight values mean that individual neurons are able to fit to steep changes.

### Regularization puts constrains on L2 norm of weights.
![image](<Pasted image 20240528102453.png>)
- lambda is the regularization constraint.
- But we sum over all layers k and lambda is applied on the sum of all Wk.
	- So regularization is not layer specific, but network wide?


**Pseudo-code for Regularization on Mini-batch**
![image](<Pasted image 20240528103549.png>)
Question: are the constrain on magnitude of weights layer specific?

## Smoothness through network structure
- Using deeper networks with same number of parameters makes function smoother.
- This is because further layers uses output of previous layers. This means the gradients often end up getting flatter further up. 
![image](<Pasted image 20240528103800.png>)
## Drop out regularization:
- For each training instance, each neuron(including input) has probability of drop out of (1-a).
	- This means that neuron is turned off in the network.
- This creates 2^N different networks where N is the number of neurons.
- This creates a network that learned by averaging over all possible networks.
- mechanism: They force neurons to have more dense connections. They prevent non-compressive layers just cloning input to output, which happens sometimes.

### Testing with dropout
There are effectively 2^N different networks. We want the ensemble of the network output.
**To choose the final network that has the average of the 2^N networks, we need the expected value of the final network.**
![image](<Pasted image 20240528104912.png>)
- The network is a function of individual neurons in the net.

**But instead we calculate the network of the expected value of each individual node.**
![image](<Pasted image 20240528105050.png>)
**The expected value of the j'th neuron of k'th layer can be computed by:**
![image](<Pasted image 20240528105156.png>)
- expected value of bernoulli = p (or alpha in this case).

**Implementing dropout during test**
Expected value of each yi(k) can be calculated by multiplying each by alpha.
Alternatively, you can multiply all weights by alpha.

## Other Heuristics
**Early stopping:**
Training error and test error does not always go together. Model can overfit to training data.
We calculate running test error periodically on validation set.
Stop early if training error and test error begins to diverge after many epochs.
![image](<Pasted image 20240528105639.png>)

**Gradient clipping: Ceiling on value of gradient prevents overshoots.**
Change to weight is proportional to both step size and gradient. Gradient that are too big may cause overshooting over minimas. Ceiling on the gradient prevents this.
![image](<Pasted image 20240528105816.png>)


**Data Augmentation: we augment existing data to create more samples of data.**
Common augmentations: rotation, stretch, adding noise, other distortions.

**Other tricks**
- Normalizing entire input: standardize with mean = 0, sd = 1
	- equialent to batch norm on entire input.

## Set up of problem
1. Obtain training data
2. Choose network architecture
	1. more neurons need more data: increased complexity means more data for generalization
	2. deep is better but harder to train: smoother curves.
3. choose appropriate divergence function:
	1. L2 norm for regression, KL for classification,
4. Choose regularization
5. Choose heuristics: 
	1. batch norm: if there is covariate shift between batches
	2. drop out: akin to bagging
		1. Doesn't always ensure best performance.
6. Choose optimization algorithm:
	1. Gradient descent methods.
		1. Batch size
		2. Learning rate
		3. gradient choosing: ei momentum methods.
		4. ADAM: combination of altering learning rate and gradeint.
7. Grid search for hyperparameters on heldout
	1. learning rate, regularization parameter lambda,
8. Train:
	1. periodically eEvaluate on validation data: for early stopping if required.
