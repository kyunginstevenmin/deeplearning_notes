## Problem statement of NN:
The general objective of NN is to estimate a function that maps input to output. 

For classification, the function can map the probability of an output given an input, and have a linear decision boundary given the probability predicted by the NN.

**For single dimension input and binary outcome, the probability of the outcome can be modelled by a sigmoid function.** 
The sigmoid function models/approximates the probability of Y=1 given the input value X.
The classifier is a threshold classifier.
![[Pasted image 20240719082725.png]]
![[Pasted image 20240719082616.png]]



**For multidimensional input and binary output, the probability can also be modelled as a sigmoid function. With multidimensional input, the decision boundary is a hyperplane classifier. For example, for 2D input, the hyperplane is a 1D line.**
![[Pasted image 20240719083022.png]]


## Estimating the model: Maximum Likelihood

**We estimate or train the parameters of the model with the objective of maximizing the likelihood of observing the data given the model parameters, which is mathematically equivalent to minimizing the KL divergence.**
The function that we use to model the probability of Yi given Xi is a sigmoid function with parameters W0 + W1.T @ Xi.

The joint probability of observing all N pairs of (Xi, Yi)  is the product of probability of each observation. Each observation is assumed to be drawn from a probability distribution that is identical and independent. 

Using bayes rule, the joint probability of X y pair is the product of P(X) and P(Y|X). And we model P(Y|X) using the sigmoid function with parameters.

![[Pasted image 20240719083200.png]]


The probability of the entire training data is decomposed to the product of the probability of observing the data Xi * the conditional probability of observing Yi given Xi. **We want to find the parameters that maximize this probability.**
Maximizing a probability is the same as maximizing a log of the probability because log function is monotonic. We do this for convenience.
![[Pasted image 20240719083831.png]]


To maximize the log(P(training data)) w.r.t. to the parameters, we only need to maximize the second term since the first term has no parameters. Maximizing the log probability is the same as minimizing the -log probability.
![[Pasted image 20240719083959.png]]


**This term is equal to minimizing the KL divergence.**
![[Pasted image 20240719084545.png]]


## For non-linear decision boundaries when the data is represented in original X dimensions, with linear classifier
In the original dimensions of X, the data is not linearly separable. The decision boundary is non-linear. However, if the final output is a linear separator, such as an SVM,  and if the function perfectly separates the data, then then penultimate layer must represent the data in a linearly separable way. The network before the penultimate layer moves data around to make data linearly separable.
![[Pasted image 20240719084803.png]]

For mathematical summary:
![[Pasted image 20240719085246.png]]

**For non-linearly separable data, we can use logistic regression or multinomial regression (instead of a threshold classifier, such as SVM) to model the P(Y|X), which is the posterior probabilities given X. We want to maximize the posterior probabilities given the training data, and for multi-class classification, Y is a vector of probabilities.**

Because the network also consists of a non-linear function f(X) that converts the input space of X, P(Y|X) = P(Y|f(X)).

When we optimize the network with a KL divergence, we learn the network parameters of 1) the final softmax/logistic classifier, and 2) the rest of the network, which finds the optimal representation of data.


**Poll 2**
Select all that are true
- **A (classification) neural network is just a statistical model that computes the a posteriori probabilities of the classes given the inputs**
- **Training the network to minimize the KL divergence (Xentropy loss) is the same as maximum likelihood training of the network**
- Training the network by minimizing KL divergence gives us a maximum likelihood estimate of the network parameters only when the classes are separable 
- **It is valid, and possibly beneficial, to train the network, and subsequently replace the final (output) layer by any other linear classifier**


## Manifold Hypothesis: Neural networks learn the best feature space transformation to make the data linearly separable
![[Pasted image 20240719091044.png]]
	figure: The first layer is the input layer, which represents data in 2 dimensions. The matrix of the 2nd layer projections the 2D data as a hyperplane in 3D pace. Subsequently, the activation function applied on the hyperplane performs non-linear transformation. And the final layer transforms the data into a 1D line for a linear separation.


### Increasingly more separable data as you go down deeper layers, and thus more complex projections.
![[Pasted image 20240719091440.png]]
	As you go deeper into the neural network and plot the intermediate layer outputs, they are more linearly separable. This plot is a 3D plot where the data of each layer is represented in terms of 3 principal components. 


## What do the intermediate layers do?

### Understanding what individual perceptrons do: correlation filter/pattern recognition.
A basic perceptron fires if the sum of the product between its weight and input exceeds a threshold. In matrix notation, this is the dot product between the weight vector and input vector. And the dot product between two vectors are the measure of similarity of the vectors in terms of the cosine of the angle. So the perceptron will fire if the weight vector and the input vector is sufficiently correlated. Thus the weight vector detects the pattern in the input. 
	![[Pasted image 20240719095120.png]]
As another illustration of the pattern correlation measuring role of perceptrong, we can represent/visualize the weights and input as a grid of values. The dot product of the weight and input would be a element wise product and summation of the product. This dot product is the measure of the correlation.
	![[Pasted image 20240719095018.png]]

The Weights are learned through backpropagation of the derivative w.r.t. the objective function of the network. Thus the features are learned to optimize predicting the target value. So in digit classification, the features learned by each perceptrons are features useful for predicting digits.


### Can we reconstruct the input exactly using the intermediate features we learned?
We may not be able to reconstruct the input exactly, but it can reconstruct it using the features that the network learned to be important in distinguish the target class.

Network designed to reconstruct the input using the features learned from it is known as the auto-encoder.
	![[Pasted image 20240719095918.png]]

Question: is the auto encoder trained to a classification task, then change the output head to make it to reproject to the dimensions of the input? 
	How do we learn W? whats the objective function? Whats the target class?
Answer: 
	The objective function is the L2 divergence of X and X_hat.
	The target is a p dimensional vector X_hat, an approximation of the input.
	The weights are a tied weight between the encoder and the decoder?
	![[Pasted image 20240719100008.png]]



### Auto-encoder as principal component
If finds the direction/axis of maximal energy, which is the direction of maximal variance of the data.
WX also maps all input vectors onto the principal axis. WX = Z.
Values on this principal axis are the values of the hidden representation. Change in this value will change the value along this axis.
The direction at which the error is minimal is the principal eigen vector.
	Whats the definition of error?
![[Pasted image 20240719101211.png]]


The error here is defined. Its L2 norm of X-X_hat.
![[Pasted image 20240719101659.png]]


The Encoder is the "analysis" net that maps the input vectors into the principal subspace.
The decoder is the "synthesis" net that maps from the principal values into the input vector space.
And the principal space is optimized by minimizing average error from mapping the input vector unto the principal space and remapping unto the input vector space. 
	![[Pasted image 20240719101848.png]]

**A non-linear principal component axis can be made when you use non-linear activation function in the hidden layers.**
![[Pasted image 20240719102257.png]]


Examples:
	![[Pasted image 20240719102422.png]]
	- Extending the hidden "z" value beyond range of Z unto which the training data project onto doesn't continue the helix (as seen in blue).
	- generalization doesn't happen beyond the manifold.

**Bottleneck Auto-encoders: this architectures have hidden representations that have lower dimensionality than the input.**
- However, the dimensions of the input is progressively reduced.
![[Pasted image 20240719102759.png]]

### Decoder as a generative model.
The decoder can generate data with dimensions of the input data from values on the manifold (ideal subspace/vector that minimizes average error). 
Its a source specific generative dictionary.

How it can be used as signal separator: given multiple signals, separate out sources.

Any mixed signal is a combination of signals coming from different dictionaries/auto-encoders.
![[Pasted image 20240719103548.png]]
	The prediction set is a combination of outputs from each decoders.
	Use backpropagation of the derivative of the cost function w.r.t. the intermediate values. The parameters of the decoder has already been learned.
	The intermediate signals can then be passed through the decoder to interpret them.

Train auto-encoder for individual signals.
