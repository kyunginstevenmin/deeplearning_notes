**Recap of the translation model:**
- The encoder RNN generates a hidden representation at each input, in recurrent manner.
- The input sequence terminates at the eos symbol, and the hidden state at the eos symbol stores all information about the sentence.
- Second RNN uses hidden activation as initial state to produce sequence of outputs.
	- Output at each time becomes input at next time.
	- Output production continues until eos is produced.
	- We pick sequence with highest probability.

Problem with this model is that all information about the input sequence is embedded in a single vector.
- Especially if the sequence is long, the single vector might not be sufficient to store all information.
- Earlier inputs might get diluted because of the distance from the final state of encoder. aka recency bias.
- Not every output related to input directly
	
**Solution: use all hidden states**
Use the average of the hidden representation of all inputs, then supply to every output word.
Problem: Same importance is given to all inputs for all outputs, but some outputs may have stronger associations with some inputs.
![image](<Pasted image 20240709194936.png>)


**Used different weighted average of inputs for each output: attention weights**
Each output computation has a unique weighted average of inputs coming in. This way each output word can have different attention on different parts of the input.
The weighted average of all the input is known as the context vector.
	The weights used are known as the attention weights. These are computed when/prior to computing the output at time t. We use the information available at time t to compute the attention weights.
![image](<Pasted image 20240709195109.png>)

How do you compute the attention weights?
1. Because they are used for weighted sum of inputs, the weights themselves must sum to 1.
2. They use information available for computing output at time t, to compute the attention weights at time t. 
3. They are computed for each input hidden representation.

Steps:
1. Calculate raw weights:
	1. They are a function of the input hidden state whose weights we are calculating, and the output hidden state at t-1.
		1. Typically this function g(hi, st-1) is a dot product of sorts. May need transformation of hi to make shapes of h and s match. 
	2. If we are using key, value, query framework, the weights are calculated by a dot product of the key and the query.
		1. Each output hidden representation has a query. Thus to calculate the context vector input the output at time t, we need to get the weights on each of this input into this context vector.
		2. This weight is computed by dot product of the query of the output and key of each input. The dot product gives the similarity between the two vectors. Think of broadcasting the dot product with the query vector across the array of key vectors of input. 
		3. The dot product is followed by a softmax over alll q dot k for all input, which means that we prioritize the most relevant input to the output in terms of similarity calculated by the dot product. Softmax also means that all the weights sum to 1, which means that the context vector becomes an weight average of all inputs.
2. softmax of raw weights.
![image](<Pasted image 20240709200755.png>)

## converting an input: inference
1. Compute hidden representation of all of input sequence.
2. Being computing output sequences.
	1. Compute attention weight for first output element. The raw weight for each input is the function of hidden representation of input and s-1. Then the raw weights will be put through a SoftMax.
	2. Using the weights, compute the weighted sum of all hidden states.
		1. All the hidden states are identical shape, since RNN's have shared parameters.
3. 

![image](<Pasted image 20240709201139.png>)
![image](<Pasted image 20240709204729.png>)
## Query key value:
The encoder outputs a key and value at each input time.
The decoder outputs a query at each output time.

The key and the query is used to compute the weights for the context vector. Its based on the dot product between the two, which computes the similarity in vector angle.

The context is weighted sum of the values, whose weights are computed as above.
![image](<Pasted image 20240709205030.png>)
### Example of attention weights:
![image](<Pasted image 20240709205636.png>)
	For each output, most important input is highlighted. General trend is diagonal, showing order synchrony in the input and the output.


## There are two training paradigms.
### Teacher forced learning/maximum likelihood approach
In teacher forced learning, we give the model the desired sequence until t-1 to prompt prediction at time t. Then compute the divergence between the output, which is a probability distribution, and the desired output. We propagate this divergence through the network, including through the context vectors.

This differs from traditional mechanisms for training where the desired output sequence until t-1 is not provided, and just the hidden representation of the input is used to generate output. But this produces initial outputs that are so random and far from the desired output that we can't compute loss.

A midway approach is to **occasionally** use drawn output instead of ground truth at t-1 to input into inference at t.
![image](<Pasted image 20240709210424.png>)


## Multihead-attention
There may be multiple contexts for each input. Each input might have different reasons why its important. For example, one context may represent the syntactic context, another context may be semantic. We represent this by creating multiple query/key/value sets. Each attender focuses on different aspects of the input.

Question:
- How are the different contexts combined? Summed? concatenated? 
![image](<Pasted image 20240709211241.png>)

### Getting rid of Recurrence in input representation?
Our encoder is used to produce hidden representation of inputs sequence. Because of its recurrent structure, the hidden representation of inputs at time t is influenced by all preceding elements. This is not necessary since our attention model attends to all inputs of the sequence anyways?

But by removing the recurrence in the encoder, we lose the context-specificity of input embeddings. The representation of a single word in a sequence does depend on other inputs. 

Solution is to use attention framework on the input to provide contextual information. The representation of inputs utilizes contextual information. This is known as self-attention.
![image](<Pasted image 20240709211644.png>)

	We get rid of the recurrent structure and replace them with the self attention structure for contextual information on the inputs. Allows 1) wider contextual representation, 2) prevents recency bias, 3) more granular contextual representation. 


### Self attention
Compute new output for each word, that has context information.
- The output dimensions is equal to the dimensions of the value, since we take a weighted average of the value outputted by each word. 
- What dimensions of V is usually smaller than the dimensions of h.

Compute query-key-value set for every word.
- the query-key-value set is learned aswell. It has weights.

For each word:
- using the query for that word, and key for all other words(also including the key of itself), compute the raw weights, using a function such as a dot product. 
- Put the raw weights through a Softmax function. This normalizes the weights assigned to every word in the input context.
- Compute the weighted sum of the values of all words using the normalized weights.


![image](<Pasted image 20240709213658.png>)

Multi-head attention uses multiple attention heads.
- each head with a unique set of query-key-value sets.
- with independent attentions weights
- with independent outputs.
Final output is an concatenation of the outputs of the attention heads. You end up with a single vector representation for each input. They have the multihead attention representation the input at time i.
![image](<Pasted image 20240709214600.png>)


multi-head attention modules can be followed by an MLP module, which is known as a multi-head self-attention block. 