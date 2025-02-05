---
Class: Intro to Deep Learning
Title: L5 Intro to Deep Learning
Topics:
  - "[[Back Propagation>)"
tags:
  - "#gradientdescent"
  - convergence
---
 
## Notes:
see: [[Jacobian>)
[[Back Propagation>)

## **Jacobians are used for both scalar activation and vector activation of a neural network.**
Activation functions where the entire vector output of the activation is dependent on the entire vector of the input. ex) softmax functions
### **Vector activation vs scalar activation**
![image](<Pasted image 20240525102206.png>)
**Jacobian for scalar activation would be a diagonal jacobian matrix, and just a complete jacobian matrix for vector activation.**
![image](<Pasted image 20240525102703.png>)
## **Example of Vector activation: Softmax in vector form vs non-vector form **
**Non-vector form**
![image](<Pasted image 20240525103149.png>)
![image](<Pasted image 20240525103204.png>)
- THe notations for the two slides are different:
- X-> Y is the vector activation part. xi affects all j elements of vector Y. 
- When we derive z w.r.t. xi, we sum the derivatives of all paths xi take to influence z. Xi->Yj-> Z. for all j. i is constant.

**Vector form**
- Transpose of Jacobian of activation function multiplied by partial derivative of Z w.r.t. y.
![image](<Pasted image 20240525103646.png>)

## Algorithm
![image](<Pasted image 20240525112552.png>)
Initilizations:
- intialize Wk, bk:
- initialize variable for empirical loss.
- Initialize Derivative of loss w.r.t. Wk and bK for all k.
Empirical loss calculation and empirical gradient calculations for training examples.
- For all t = 1:T # for all training instances
	- forward pass:
		- calculate output Y.
		- Calculate divergence w.r.t. output y.
		- loss += divergenc. # loss is the sum of all divergence of all T examples?
	- Backward pass: for all K starting with N-1:
		- 
We calculate the average empirical loss for all training examples of size T (batch size). 
- 
