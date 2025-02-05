## Convolutional Layer:
There are two stages for computing output map:
1. Compute affine map with convolution of a filter.
2. Elementwise activation of output of covolution.
![[Pasted image 20240623120059.png]]


### Convolution with filter


![[Pasted image 20240620092543.png]]
**Ill try to organize in terms of top-down hierarchy.**

**A single Convolutional layer:**
- performs same convolutional operation with same hyperparameters:
	- hyperparameters include:
		- Stride
		- size of kernel/weights
- has as many output neurons as number of filters.
	- The number of output neurons is equal to the number of filters.
	- Each filter has as many unique kernels as the number of input channels/maps/neurons in previous layer.
![[Pasted image 20240623115040.png]]


**A single output neuron/filter:**
- outputs a single scalar value, but since we are scanning, we produce a map of output.
	- In application, we don't scan through entire input, but create duplicates of affine neurons that share the parameters, to fill up the entire input space.
- A output neuron is same as a filter.
	- a filter has as many kernels as the number of input maps.
	- We do a dot product of kernel weights and input channels, then sum the result from all channels and add bias term. 
	- Striding the filter across the input field gives us the output map.


**How is the output map of a layer calculated:**
- For each output map/filter/neuron:
	- For each/all input channel/map:
		- each input channel has a unique kernel for a output neuron. we sum the results of the dot product of input channel  and unique kernel. 
		- For each segment with stride:
			- a dot product is performed with a unique kernel(unique for input map and output neuron/filter).
			- Sum the dot product output of all channels + bias. This is a scalar value for a single position in the convolved feature map. Each filter creates a single 2D output map.

For each filter, we do an affine transformation for each input channel then sum with a bias. Each filter has a set of kernels, with unique weights for each input channel. But a filter has a constant/shared weight that scans across the entire input space of the input channels. 

The affine transformation and summing over the results of individual channels combines information from various channels. This is feature combination, which creates a new more abstract/complex feature.


![[Pasted image 20240621195322.png]]



**How many filters are there in a layer?
- The number of filters is the number of output neurons.
- Its the number of output channels of a layer.

**How many unique kernels in a layer?**
	Each filter/output neuron has k$_{j-1}$, which is the number of input channels/neurons in previous layer, number of unique kernels.
	Each layer has k$_{j}$ number of filters/outputs.
	Hence each layer has k$_{j-1}$ $*$ k$_{j}$ number of unique kernels.
**How many weights in a layer?** 
	Each layer has k(j-1) times k(j) number of unique kernels.
	Each kernel has number of weights according to its size: L^2.
	Hence, each layer has (L$^2$ $*$   k$_{j-1}$ +1) $*$ k$_{j}$ number of weights. 
	Where L$^2$ is the size of the kernel, k$_{j-1}$ is the number of maps/channels/inputs in previous layer, and k$_j$ is the number of output maps/filter/neurons.


## Hyperparameters
1. Padding
	1. utility: preserves information on the edges of the maps.
	2. method: common method is to add zeroes around the edges.
	3. example: alexnet
2. Kernel size:
	Smaller kernel size means more fine grained information. Larger kernel size means we abstract the data faster. 
	Larger kernel are better at extracting larger features.
	How does kernel size relate to the number of dimensions in a layer?
	Using multiple layers of smaller kernel size means we can create hierarchy of feature extraction.
		Layers closer to the input are more fine-grained. Subsequent layers have kernels that go over the feature maps, which results in construction of larger feature maps.
3. Stride:
	Larger stride means less granular scanning.


## resampling:
- down sampling usually follows convolution or pooling.
	- akin to stride of s in convolution.
- upsampling is followed by convolution.
	- akin to stride of 1/s.
### Down sampling
- Downsampling is merged with Convolution or pooling, since its the same with convolution with strides.
- You should conceptualize convolution with stride as convolution followed by down sampling.
- But computationally easier to calculate derivatives in terms of convolution with stride.

## up sampling layer
Introduce s-1 rows and columns for every map in layer, often values of 0. Thus it doesn't make sense to use with pooling.
Also known as transpose convolution.
Up-sampling followed by convolution. This is the same as striding by factor of 1/s.

## setting everything together
RGB can be thought of as three channels of input.

If you have k1 filters, and each filter size is LxLx3, how many planes are you computing?
	K1 channels in the 1st layer(after input).

The parameters to choose:
1. Number of filters
2. size of filter
	1. The 1x1 filter has no spatial contextual information.
3. stride of S

Learnable parameters: 3xLxL parameters for each filter.

After applying k1 filters, you have k1 channels.

Pooling layer/downsampling reduces the size of the output map of each k1 channels.
- It reduces the map by a factor of d^2.

Why you increase the number of channels as you do deeper into the layer is to prevent loosing information from down sampling.
	The number of pixels in first layer is  


### The size of the layers
Each convolution layer with stride of 1 maintains the size of image, given that we have appropriate zero padding.
- Without zero padding, the output map decreases in size.

Each convolutional layer increases the number of filters/output channels/output maps.
- Increasing the channels reduces the amount of information lost by subsequent downsampling.
- More filters also mean we detect more features.

Pooling layer with stride D decreases the size of maps by factor of D.

In general, the number of filter increases with layers.


### Design choices to make:
- number of convolutional and downsampling layers
- For each convolutional layer:
	- number of filters: ki. This is the number of output maps/channels we want
	- spatial extent of filter: LxL.
	- Stride Si: This is also how much we down sample by/ reduce the size of the feature map.
	- We don't determine the depth of each filter. This is determined by the number of filters in the previous layer.
- for each downsampling/pooling layer
	- Spatial extent of filter: pi x pi
	- stride: di
- Final MLP:
	- Number of layers and neurons in each layer.


### training:
 