---
Class: 
Title: 
Topics: 
tags: 
url: https://www.youtube.com/watch?v=2-c1kaxUnmk
---
Recap:
- NN is a universal function approximator
- how to train NN. 
- Convolution neural network
- Recurrent network(current)

Series of input to produce a series of output.
	Each input is a vector.
	Each output is a vector.
	A series of input is different from just the sum of individual elements. Or atleast, we don't want them to be. Because we want the input at time t of the series to affect output at other time t'.


### Finite response networks: something that happens today affects the output of the system for N days in the future.
The finite response network is an network that takes historical data into account to make current predictions. It does so by using the input data up to N time back in the past, up to now. So the network takes as its input data, all X's from t-N to t, where t is the current time, and N is a parameter we can choose. 

The limitation with this finite response networks is that its difficult to model longer term trends into the network. 
	We would have to set a value of N that is appropriate. 
	If N gets too big, the network becomes too big?

- but there are multiple longer term trends each with their own reasons.
	- for example, daily trends, weekly trends, monthly trends.
- If then how do you set an appropriate N for a finite response network?
	- How many days do you go back to use the information to predict todays response?
CNNs is a finite response system. aka Time-delay neural network when one dimensional data. 
- there is a finite window of input that affects the model.
- We shift with fixed window of input along the time axis.

![image](</Images/Pasted image 20240627083328.png>)
	The network looks back upto T-N input vectors, where N = 3.
![image](</Images/Pasted image 20240627083511.png>)
	The network looks at X's from current t to t-N, to product an output.

### Infinite response systems: Everything in the past lies in Y.
- To make the past have continuing effect on future, we can make the output of today as a function of (output of yesterday, other features of today). (as shown in the equation below)
- As long as t continues, the memory continues to be propagated via Y. 
- Even if Xt = 0 or input does not existed, Yt can be computed with from Yt-1.
- Few requirements:
	- initial state of Yt-1, when t=0 must be defined.
![image](</Images/Pasted image 20240627084144.png>)



#### Narx Networks: "nonlinear autoregressive network with exogenous inputs"
- Infinite response is an instance of NARX network.
- popular for time series prediction.
	- used in weather, stock markets, language(LLMs).
- any phenomena with distinct innovations that 'drive' and output.
- memory is captured in the output, not in the network.
	- The output at t becomes the input of the next network at t+1. 
- More generic Narx Network can also have incorporate previous input data into current predictions as well as previous output data.
	- Yt would be a function of (Xt, ..., Xt-L) and (Yt-1, ..., Yt-K)
![image](</Images/Pasted image 20240626132020.png>)

### How do you make memory a more explicit component of the network?
There are several networks that were developed to make memory an explicit component of the network.  
	In Narx network, past data is not explicitly stored as a variable of the network. Its just the output of the network.

We create a memory variable at mt, with is a function of previous y, previous hidden values, and previous memory variable value.
We make the current network a function of current Xt, and the memory variable mt.
![image](</Images/Pasted image 20240627085555.png>)


Two of which are: Jordan network, and Elman Network.
#### Jordan network
- simply retain running average of past outputs.
	- Memory variable is a running average of past outputs.
- but still, memory unit is fixed structure: no learning of parameters.
	- What do we mean by there is no learning parameters?
	- How the past outputs affect current output is not learnable. There's no associated parameter.
![image](</Images/Pasted image 20240627085633.png>)


#### Elman Networks:
- context unit: carries historical state of hidden units.
	- contains some information of inner working of network in the past.
	- Its different from the jordan network which only used the output of the past networks. Elman uses the hidden units of the past networks.
- problem:
	- we segment into separate units of MLP at times t's. 
	- Means that error isn't back propagated from memory unit to previous hidden unit.
		- The derivative of the hidden unit(that also becomes the memory unit) does not take into account its pathway via the memory unit to the hidden unit at next time frame.
		- So the derivative of loss w.r.t. the green hidden unit should be the sum of derivative via y(t) and via the hidden unit at t+1. State space model does this. 
- solution:
	- But the State-space model is an model that carries historical state of hidden units to current state of hidden units, and allows back propagation of current error to previous hidden states. 
![image](</Images/Pasted image 20240627090022.png>)

### The state-space model
Fully recurrent networks.
- current state(ht) is a function of previous hidden state and current input.
	- this is the recurrent element, since the definition of the current state is defined by the the same definition at t-1.
- current output, yt, is a function of the current state. 
![image](</Images/Pasted image 20240627090702.png>)
![image](</Images/Pasted image 20240627090803.png>)

	Current state is affected by input at t, and state at t-1.
	Input at t=0 affects outputs forever.
	All columns are identical.

generalizations with other recurrences:
- current hidden state can affect not just next hidden state, but next next hidden state.
### Equation of RNN:
![image](</Images/Pasted image 20240627093710.png>)
	W(1): weight of X(t) on intermediate value which is used for h(t).
	W(1,1): weight of how hidden state h(t-1) on h(t)
	Affine transformation of W(1) and X(t) and W(1,1) and h(t-1)  + bias.
	The affine transformation + bias is then put in an activation function f1: usually tanh()
	Output is a function of affinte transformation of h1(t) with W(2), followed by activation function.
		The activation function could be a vector activation like softmax.

### Variants on recurrent nets
![image](</Images/Pasted image 20240627094227.png>)
![image](</Images/Pasted image 20240627094254.png>)


### how do we train the network?

Each columns are identical in structure. They also share the parameters.
- Question: does this mean how previous hidden layer affects current hidden layer is constant? 
Each columns look like an MLP.

- Back propagation through time:
- given a collection of sequence of inputs:
	- (Xi, Di)
- Network makes sequence of outputs Y
	- The output is a sequence, and a sequence it not the same as the sum of individual outputs.
	- Each element of sequence is the output at time t, which is a vector.


Divergence between sequence of outputs and desired sequence of outputs.
- how do you get a divergence between two sequences? especially when they are different lengths? When there are no  one to one corresponsdence of output and desired sequence.
- you need to manipulate output so that it looks like form of desired output.

First step: derivative of divergence w.r.t. Y at all T.
- we need all the derivatives for back prop.


![image](</Images/Pasted image 20240626143010.png>)

### Bidirectional RNN
![image](</Images/Pasted image 20240627094356.png>)
- We can concatenate outputs of forward pass and backward pass.
- 