RNN is the same as other NN in that you need to calculate the Divergence between Y_hat and Y_desired.
The structure of input and outputs are different from ordinary MLP in that input is a sequence of X's at various t's. The output is also a sequence of Y's at various t's. 
	How is this different from traditional MLP's? 
	Traditional MLP's, you have X at single t, and Y at single t. For example, n-dimensional vector X and Y can be a m-dimensional vector for probabilities for m different classes.
	But for RNN's we don't want X at single t, but sequence of X's at multiple t's, and predict either single Y, or multiple Y's. 
	**That's why we need to calculate the divergence between sequence of outputs and sequence of desired outputs.** And the sum of divergence of individual components is not equal to the divergence of the whole, atleast not always.

**The challenge with defining this divergence is that the inputs and outputs may not be time alligned or even synchronous.**
	How would you describe time aligned and synchrony?

## Variants of recurrent nets and examples
![[Pasted image 20240701152318.png]]
Many to one:
- sequence classification: predict single class for a sequence of inputs.
- examples: phrase recognition

Order synchronous, time asynchronous sequence to sequence generation: 
- example: phenome recognition
	- you have a sequence of inputs, and you need to predict sequence of output. Order matters, but time doesn't matter as much, or is less clearly defined/labelled.
- i guess this is a variant of many to one recurrent net.

![[Pasted image 20240701152608.png]]
Many to many:
- A posteriori sequence to sequence: get generate output sequence after processing input.
- example: language translation
	- you need to process entire(?) sequence of input before generating output.

one-to-many:
- single input a posteriori sequence generation.
- Input at single time used to generate output of sequence (output that has temporal structure)
- example: captioning an image.

## divergence for regular MLP for series: one to one
One-to-one is a conventional MLP with no memory component. We can still use this to analyze time series, if we don't want memory component. Just scan entire series with repeats of MLP.

If we use a regular MLP to analyze time series, we can treat the divergence of Y_out and Y_desired, which are both serieses, as the sum of divergence individual Y_outs and Y_desired at each t.
	Assumption is that we have one to one correspondence between actual and target outputs.

## Time synchronous network
Many-to-many recurrent network, with one output for one input. The one output for one input correspondence makes the network time synchronous.
example: assigning grammar tags to words.
- can be bidirectional recurrent networks, meaning that past history is used to interpret future, and future is used to interpret past.
- for example, to understand grammar of words, we need to know what comes before as well as what comes after.

Time synchronous networks can also make the useful assumption that the total divergence of the Y_out and Y_desired sequence is the sum of divergence of individual instants.
	This is because there is a one to one correspondence in input and output?
![[Pasted image 20240701154233.png]]

## Many to one recurrent nets
example: answering questions:
- input is a sequence of words.
- output is an answer at the end of the question
example: phenome recognition
- input: sequence of feature vectors
- output: phenome ID at end of sequence.
	- N-dimensional probability vector were N is the number of phenome categories.

![[Pasted image 20240701154918.png]]
many to one RNN:
![[Pasted image 20240701154958.png]]
We treat many-to-one RNN as many-to-many RNN for learning since more useful to calculate divergence.
## Inference: forward pass
In many-to-one recurrent nets, output is generated at every t, but its only read at the end of the sequence. But none-the-less we do extend the phenome(or any Y_desired) over range of t's so that we can propagate the divergence for the intermediate structures. So we are converting a many-to-one, order synchronous, time-asynchronous RNN to a order-synchronous, time synchronous RNN. (Many to one RNN's are always order synchronous because there's only one order to arrange things.)
	We also end up making the necessary simplication for calculating the divergence, since we convert the form to a time synchronous RNN:  the total divergence of the Y_out and Y_desired sequence is the sum of divergence of individual instants.
 ![[Pasted image 20240701155856.png]] 
	 We smear/merge output to untagged outputs.
![[Pasted image 20240701155910.png]]
	As we did with time-synchronous RNN models, we treat divergence of output as the sum of divergence of components.
	We can give different weights for divergence at each time t, depending on how important the prediction is at each t?
	example: answering questions. Y_out is more significant at end of the sequence. Higher weight at later T.


## Sequence to sequence problems: order-synchronous, time asynchronous
![[Pasted image 20240701160920.png]]

Problem objective: given sequence of inputs, asynchronously output a sequence of symbols.
- this is concatentation of many-to-one RNN.
- But this complicates problem:
This means that delineating the end of the sequence becomes important problem, especially because the ends of the sequences are often not labelled for a given data. For example, for a phoneme prediction, we are given a recording of a word, and a sequence of phenome label for the word. The exact time frame over the recording that corresponds to the correct phenome is not provided. So predicting the correct location for each phenome component is an important problem.

### The structure of output of network:
The structure of the entire output sequence can be described as a table.
Each column is the Y_out at each time t. Its a vector of probabilities for each class label.
	Each output Y is the posterior probability of a class label given inputs X up to time t.
	~~(We make maximum a posteriori classification to select best label at each time t.)~~
	Our objective is not to pick the MAP class label at each time t, but to pick the sequence with greatest MAP. 
Each row is the probability of a given class through time t.
![[Pasted image 20240701161004.png]]

**Our objective being picking the sequence of output Y that has the highest a posterior probability, how do we do this?**

![[Pasted image 20240701161507.png]]
method 1: 
- pick class with highest probability at each t. 
- merge adjacent classes with the same predictions and keep the last output.
- problem: you can't distinguish true repeats of class vs just merging of class. 
![[Pasted image 20240701161701.png]]

So the important questions are:
1. how do you know when to output symbols, when the network outputs at every time? (we want time asynchronous output, but training is done with time synchrony)
2. How do we train these models.
We'll address how we train these models first.


## training:
To train the model, we need the desired Y, but we don't have the exact desired Y_out. 
	**Y_out is not aligned correctly.** 
		They are order synchronous, but not time-synchronous. 
	**We need to align the labels**: expand them over t in the correct time-synchrony.
		But we don't have this information, so what are we going to do?
![[Pasted image 20240701162315.png]]

### One solution is: guess alignment, train model, and iterate
**Description:**
Given that the loss is a function of both alignment and the model parameters, we can minimize loss by iteratively fixing one and minimizing the loss w.r.t. the other parameter. Then alternate process.

**Procedure:**
Steps:
- initialize:
	- initialize initial allignment
		- problem: the initial alignment may cause the network to be stuck in local minima.
		- solution: [[Connectionist temporal classification]] for original paper, [[L16 Sequence to sequence models Connectionist Temporal Classification]] for lecture
- iterate:
	- train network using current allignment
		- minimizing loss w.r.t. model parameters
	- re-estimate alignment for each training instance.
		- How exactly do you re-estimate the allignment for each training instance?
		- each each time stamp, the model predicts the most likely symbol. that becomes the possible allignment?

**Training objective**
In words: 
	given K-length compressed sequence, and given N-lengthed input, find the expanded sequence, that is most likely. 

in math:
![[Pasted image 20240701163704.png]]


#### mechanism of aligning sequence:
1. Expansion of compressed sequence.
You align the sequence to the input by expanding the compressed sequence to a sequence of equal length with the input.
The opposite operation is compress: compress expanded sequence to a 'compressed' sequence, usually by keeping only the last label of the repeats.

2. constraining how we expand compressed sequence.
There are many ways to expand the compressed sequence. What are some good ways?

**Method 1:** mask unnecessary outputs
We only consider elements that appeared in compressed sequence and mask all the others when we choose elements with highest M.A.P. So even if the model predicts a given class as the most probably class at time t, if that class is not in Y_desired, we can not choose that as alignment class. We choose element with MAP on reduced grid.
![[Pasted image 20240701164321.png]]
![[Pasted image 20240701164404.png]]


**Method 2:** explicitly arrange constructed table
Construct table of possible output class, in the order in which each element of the sequence appears in the Y_out sequence. 
 ![[Pasted image 20240701164532.png]]

Explicitly constrain alignment: the alignment is the path along the graph.
- first element must be top left, and path must end at bottom right.
- Symbols must travel monotonically from top left to bottom right.
	- because the table has been constructed such that the order of rows are order in which elements occur in Y_desired sequence.
	- monotonically means the gradient doesn't change sign. Path doesn't go up.
- guarantees the sequence is an expansion of the target sequence.
![[Pasted image 20240701164730.png]]
Pick alignment of Maximum a posteriori, or the most probable path from source to sink using algorithm finding best path score.
- The best path score is the product of probabilities of all nodes along the path.
	- ex) Scr(path) = Y0B x Y1B x Y2IY x Y3IY x Y4F x Y5F x Y6F x Y7IY x Y8IY
- we can use algorithm such as The Viterbi algorithm.

Brief explanation of The Viterbi Algorithm
	- The best path to the current node is the path from the parent with the best score to the current node. 
	- So at every node, look at the best path score of its two parents.
	- Pick parent with best path score as its parent path. 
	- parent path score and current node probability = best path score of current node.


**The Viterbi algorithm gives us the assumed target for training.**
After fixing the target sequence, we optimize our RNN parameters with gradient descent. 
We use Div = sum over t (div(Yt,symbol_t of bestpath)). 
![[Pasted image 20240702084952.png]]
- Makes sense because of probability of Y is small, then the gradient is negative and large. We want to make large changes in our parameters, because our estimated probability is low for class of Y, which is the symbol at time t under our estimation of best path/best allignment.
- If probability of Y is high, then gradient is small. 

**Iterative estimating and training**
After training our parameters to produce highest probability given alignment, we decode the table of probabilities again to get new alignments according to new probabilities. 
![[Pasted image 20240702085515.png]]


The decode and train steps may be combined into a single block for SGD and minibatch updates?

Option 1:
- for every training instance, initialize alignment.
- use Minibatch or SGD for gradient descent on the entire training set
- iterate.
option 2:
- during SGD, for each training instance, find alignment during forward pass.
	- The forward pass gives us the vector of probabilities given training X's. 
- Use alignment in backward pass.

So in option 1 we initialize the alignments for all training instances even before training model. Then train our model for the minibatch or SGD with fixed alignment for every training data.

But in option 2, we initialize alignment for each instance and gradient descent step of SGD. So for the next instance, we get new alignment based on updated parameters in previous SGD step.
 
![[Pasted image 20240702085515.png]]