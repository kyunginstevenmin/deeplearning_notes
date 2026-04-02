# topics:
1. MLP as universal boolean function
2. MLP as universal classifier
3. MLP as universal approximator
4. Optimal depth vs width
5. rbf networks.

# perceptron are the basic units of a neural network
![image](</Images/Pasted image 20240519071343.png>)
They have a 1) affine function of the inputs, 2) activation function. There are many activation functions. Here is a threshold gate activation, where the perceptron outputs 1 if the affine function is greater than 0.
**There are other softer activation functions:**
**Squashing function**: squashes affine function outputs, which ranges from negative to positive infinity to bounded y value.
- sigmoid: 0 to -1
- tanh: -1 to 1
![image](<Images/Pasted image 20240519071718.png>)
## definitions:
**depth of neural network:** 
- depth of any source and sink is the length of the longest path.
- deep neural network: depth is greater 3.
**Layer:**
- neurons on the same depth from the source.


# What functions can multi-layer perceptrons approximate?
MLP's are 1) universal boolean functions, 2) universal classifiers, 3) universal function approximators

# MLP as universal boolean function
Single layer perceptrons can model all boolean operators except XOR.
MLP's can compute XOR with 3 perceptrons.

### 1 hidden layer MLP is a universal boolean function
![image](<Images/Pasted image 20240519072746.png>)
We can represent all boolean function using Truth tables. Truth table shows all input combinations for which the output is 1. We can create 1 perceptron for each row of the truth table/combination of variables for which output is 1. 

The greatest number of total combinations is 2^N for N number of variables.


**But we can reduce the number of perceptrons needed for representing the boolean function, by expressing boolean function interms of DNF.**
![image](<Images/Pasted image 20240519072746.png>)
Adjacent boxes can be grouped into DNF formulas for the table. And we create perceptrons to model each DNF formulas. 


**Whats the largest irreducible number of DNF?
![image](<Images/Pasted image 20240519073201.png>)



**For N number of variables, the largest irreducible number of DNF occurs when there are no adjacent boxes. When there are no combinations that differ by just one bit.**
![image](<Images/Pasted image 20240519073201.png>)
You need as many as (2^N)/2 = (total number of combinations)/2 = 2^(N-1)
This is exponential in N.


### multi-layer perceptron reduces the number of perceptron needed to compute boolean functions.
![image](<Images/Pasted image 20240519073412.png>)


**The number of perceptron becomes linear in N when you allow depth in the network.**
![image](<Images/Pasted image 20240519074106.png>)
The largest irreducible number of DNF is the same as performing XOR on all variables. A single XOR can be computed using 3 neurons with 2 layers. So for N=6, we need 3(N-1) perceptron. = 3 perceptrons x (number of XOR operations). 

The depth of the network is 2log2(N) layers. XOR on all two variable pairs reduces the number of variables by 2. Which means each XOR layer needs to be performed log2(N) times, to have 1 output. Each layer has 2 layers of perceptrons, hence the times 2.


**If you don't have enough depth, the rest of operations will be performed in single layer, with requires exponential amount of neurons, with respect to the number of remaining variables.**
![image](<Images/Pasted image 20240519080038.png>)
- the final layer in this network is a single layer with exponential number of neurons to compute the XOR of the remaining number of variables. 

### The need for depth:
Not having sufficient depth means that he number of perceptrons required from that point on grows exponentially. Does this mean having few extra layers acts as a buffer to prevent this from happening?


# MLP as universal classifier
**MLPs can draw any decision boundaries by using AND operations to draw polygons, and OR operators to combine the polygon reginons.**
![image](<Images/Pasted image 20240519083736.png>)
**A Single hidden layer network can also be used to approximate any boundary to arbitrary precision. But it needs alot of neurons.**
![image](<Images/Pasted image 20240519083906.png>)
- We can do this by composing an infinite sided polygon using a AND operator, and setting the threshold of the output neuron as N or N/2,by subtracting the bias. 
- You can have many circles at any locations, so they can approximate any decision boundary.
![image](<Images/Pasted image 20240519084051.png>)

**In classification as well, having more depth reduces the number of perceptrons needed.**![image](<Images/Pasted image 20240519084240.png>)

# MLP as universal approximator
**Any function can be approximated with arbitrary precision, by creating K sets of perceptrons that model the function output of that input region. **
- The value at that region is set scaled by h. 
- The final output neuron would be the sum of the sets of neurons.
![image](<Images/Pasted image 20240519093529.png>)


**But most output neurons are activation functions, whose domain and range are fixed. So MLP's are universal approximators within the domain of the input neurons and range of the activation output neuron**
![image](<Images/Pasted image 20240519094104.png>)
# Sufficient Depth and Width
**Not all neural networks can sufficiently approximate all functions. The architecture of the neural network needs sufficient capacity, which is depth and width of the neural network.**

**Because the networks are directed, all the information passed on from one layer to another is limited by the width of the layer. Thus every layer must be sufficiently wide to capture the function.** 
For example, a neural network with only 8 neurons in perceptron can only capture 8 boundaries. The layers further down will only have information about these 8 boundaries: whether the points crossed the boundary. It also doesn't give any information about distance from the boundary. 
![image](<Images/Pasted image 20240519094548.png>)

**Graded Activation functions: Threshold activation functions give only limited information about whether threshold has been crossed. It doesn't give information about distance from the boundary. Other activation functions such as ReLU and sigmoid give information about distance from boundary.**
- Graded activation functions can compensate for the lack of width of a layer, but you would need more depth. 
 **What are the benefits of using graded activation functions compared to threshold activation functions?**
potential answer.
1. Because graded outputs provides information about how much we missed the boundary by. It passes this information to the next layer, which can learn on the remaining information (similar to residuals). 
	1. Sigmoid provides information about distance from threshold(arbitrarily set to 0.5). But the information becomes more saturated the further away from the threshold. ReLU preserves information about distance from boundary. 
2. Graded functions are differentiable. We can minimize the neural network deviance if the functions of the layers are differentiable.
