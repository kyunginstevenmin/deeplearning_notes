For sequence to sequence models that are order synchronous but not time asynchronous, we need to align the output so that the output sequence has the same length as the input.

We then compute the divergence from the time-aligned sequence.
	The divergence of the entire sequence is the sum of divergence at each element.
	The derivative of the divergence at each time t is a vector with all zeros except at the component corresponding to the target at time t.
	This makes sense because the divergence is defined only by probability of only the target class. We want to maximize the log likelihood of the target class, or minimize the -log likelihood of the target class.
	Then the derivative of the divergence only depends on the value of the target class.
	The derivative is - 1/probability of target class.
		If probability of target class is small, the derivative is high. We take large steps in gradient descent.
![[Pasted image 20240702141726.png]]


**The problem with training with alignment is that its heavily dependent on the initial alignment of the target components.** 
- we optimize our parameters w.r.t. the initial alignment of the sequence, but the initial alignment of the sequence is dependent on randomly initialized parameters, which generates a probability for each class in the forward pass.


**The solution is training without alignment.**
![[Pasted image 20240702142611.png]]
- Choosing single best alignment and optimizing w.r.t. to it is like drawing single sample with highest probability from probability distribution of alignments.
	- All possible alignments with its probability is the probability distribution of alignments.
- Instead of selecting single most likely alignment, we can use the statistical expectation of all possible alignments.
	- this is using the entire distribution of alignment.


**Expectation over all alignments**
When we optimize over average alignment, we get the expectation of divergence. 
![[Pasted image 20240702143039.png]]- sum for every t,
	sum for every possible specific symbol s 
		(probability of alignment of symbol given sequence and input) x log(probability of symbol S)

**Its the weighted sum of log probability of target class, weighted by the posterior likelihood of that node being the aligned path, which is the probability that the symbol is the correct alignment at that time t.**

The likelihood of the node being the aligned path is known as the a posterior symbol probability.
	This is the total probability of all valid paths in the graph for target sequence S that go through the symbol Sr at time t. 
	![[Pasted image 20240702144602.png]]
	We can calculate this total probability using forward backward algorithm.

	![[Pasted image 20240702144833.png]]
	We normalize the posterior probability by dividing by sum of total probability of all valid paths through symbol Sr' for all Sr'. 
		Before normalization, individual  P is the joint probability that target sequence is S and the t'th symbol is equal to Sr, given input X.
		After normalization, its the probability that the correctly aligned symbol is Sr, given the target sequence is S and input X.
	This posterior probability gamma is the likelihood of the node that we weight the log probability of target class by.








**the divergence w.r.t. output Yt is no longer 0 at all except one target class since we are now not choosing one alignment but choosing the statistical expectation of the alignment.**
	
![[Pasted image 20240702145247.png]]



**If there are repeats of symbols at a particular time t, the derivative of divergence w.r.t. that symbol is the sum of derivative of divergence of all occurrence of that symbol.**
![[Pasted image 20240702145731.png]]




**Product rule for derivative: We drop second term for convenience**.
	The derivative of divergence w.r.t. particular symbol at time t becomes gamma(t,r) / output probability of target class l at time t. 

![[Pasted image 20240702145752.png]]
![[Pasted image 20240702145801.png]]
### Derivative of the expected divergence w.r.t. each output at each time.
- first, any element that doesn't appear in the graph, the derivative = 0
- for any element in graph, sum gamma of that nodes and divide by y.
	- There may be multiple nodes that has the element. There are multiple alignments that can have the particular symbol and that time t. 
	- We sum over all instances of gamma(t,r), then divide by y.
	- Y is common over both instances of gamma, because Y is not dependent on the alignment and thus its not dependent on the particular path along nodes, and thus is not dependent on gamma. Its dependent on the network.


![[Pasted image 20240702145849.png]]
- we are weighting the derivative of class probability by the posterior probability of that symbol being aligned.


## Overall steps:
1. set up RNN network.
	1. typically LSTM
2. initialize all parameters of the network.
for each training instance: 
3. forward pass: obtain symbol probabilites at each time.
![[Pasted image 20240702150934.png]]

4. Construct probability table the specific symbol sequence in the instance.
![[Pasted image 20240702151035.png]]
5. Perform forward backward algorithm to compute a(t,r) and B(t,r) at each time, for each row of nodes in the graph. Compute y(t,r) which is the posterior probabilities of each node.
6. compute derivative of divergence for each Yt.
	![[Pasted image 20240702151233.png]]

7. Propagate dDiv/dY_t_l and aggregate derivatives over minibatch and update parameters.


## A key Decoding problem
When we decode, we need to be able to distinguish true repeats of symbols and extended symbols.
	Because when we compress the extended sequences, all repeat symbols compress to one symbol.
	Thus to extend a true repeated symbol in extended form, we need to include a blank in between them.
	

## Greedy decodes are suboptimal
Greedy decodes choses alignment with the most likely time-synchronous output.
	For example in the time-synchronous outputs below, TTEEED has the highest probability. Thus greedy decodes will classify as TED.
	But if you consider the sum of all alignments, RED is more likely.
So greedy decodes optimizes on single output of time-synchronous output.
But a better choice would be to optimize on order-synchronous output, which would consider all possible time-synchronous outputs that correspond to that order synchronous output.
![[Pasted image 20240708085851.png]]

**So our objective is:**
![[Pasted image 20240708090709.png]]

To calculate the probability of a order-aligned sequence, you need to 1) calculate the probability of all time-synchronous symbols, which is denoted by alpha, 2) calculate the probability of time-synchronous symbols that corresponds to the order aligned sequences. 3) then pick the order aligned sequence whose (2) / (1) is the greatest. This sequence is denoted by S hat.
![[Pasted image 20240708091627.png]]
![[Pasted image 20240708091555.png]]

