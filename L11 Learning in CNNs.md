## Resampling
Up-sampling:
- Upsampling followed by convolution is akin to convolution with fractional slide.
- for calculating back prop, better to think of as separate steps.

## CNN: end to end

## Back propagation through convolution
- Given that we have derivative of Div. w.r.t. activation map Y, we need to compute derivative w.r.t.
	- Affine maps: derivative of activation function. Pointwise operation.
	- Previous activation map Y(l-1): 
	- Convolution filters.
![[Pasted image 20240622211858.png]]

## Backpropagating through activation
- Activation is a component wise operation on the affine map. Thus, backpropagation is also a component wise operation, multiplying the divergence w.r.t. Y with the derivative of Y w.r.t. Z at every component. 
![[Pasted image 20240622212128.png]]

## Back propagation through convolution: 
### Dependency diagram for previous activation map.
- We need to sum over all the dependency pathway that Y goes through. 
- A single input map influences divergence through all the affine maps it contributes to.
	- The m'th input map influences the affine maps via the m'th kernel of all n filters. 
![[Pasted image 20240622213003.png]]
### This is a influence diagram of how a single y of m'th input affects all n affine maps.
- A single Y affects multiple components of N'th map of Z. A single Y affects all N maps of Z. 
- We thus need to sum over influence of Y on all components of Z, for all N maps of Z.
![[Pasted image 20240622213441.png]]



### How a single Y(l-1, m, x, y) influences z(l, n, x', y')
- A single Y(l-2, m, x,y) influences z(l, n, x', y') via w(m, n, x-x', y-y').
- So the entire influence of a single Y on Z is the sum of all z(x',y') it influences.

![[Pasted image 20240623093616.png]]
### To calculate the total dDiv/dY, we perform convolution operation with weight kernel thats been flipped horizontally on the zero-padded derivative map of the output.
The convolution operation is composed of finding the dot product of the kernel matrix(flipped horizontally and vertically) and the derivative map of Z. The flipped kernel matrix becomes the matrix of dZ/Dy. Dot product of dDiv/dZ and dZ/dy, which does an component wise product and sums the result, give the sum of all dDiv/dZ x dZ/dY over a Z's that a particular Y influences. 

This gives us how a Y(l-1, m) influences a single Z(l,n) map. But Y influences D Z maps where D is the number of filters. So perform this operation for all D output maps. 
![[Pasted image 20240624080819.png]]

### We perform this transposed convolution for all n'th output maps that a m input map influences. For this we select the m'th kernel for all N filters, then convolve over the entire derivative maps of the N outputs.


![[Pasted image 20240624081633.png]]

![[Pasted image 20240624082000.png]]

![[Pasted image 20240624082839.png]]
## The filter derivative:
- One weight kernel maps one input map to one output map.
- The derivative of the weight kernel is given by the convolution of derivative of divergence w.r.t. dz on the input map. 