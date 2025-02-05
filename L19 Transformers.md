# Table of contents
1. The transformer architecture
2. Pre-training and fine tuning
3. Transformer applications
4. case-study: large language models

## 1. Transformer architecture
![image](</Images/Pasted image 20240716163553.png>)
**The general architecture of a transformer is composed of two parts: encoder and decoder.**
Encoder is mainly for creating hidden representations of input sequences in a meaningful way, particularly by employing attention mechansims.

Transformers since 2018 are encoder only and decoder only architectures.

The attention mechanism in encoder is a self-attention mechanism that gives contextual information elements of the sequence, with respect to other elements of the sequence.

### Task statement: machine translation
Our task is to process translate some input sequence and generate an output sequence.

### Input processing:
![image](</Images/Pasted image 20240717083837.png>)
The initial steps of input processing is:
1. Input embeddings
2. positional encoding.

#### Input embeddings
![image](</Images/Pasted image 20240716164319.png>)
Sequence of words is first tokenized, or split into individual words.
Individual tokens are represented as one-hot encoded vectors.
They are passed through embedding layer to create embeddings for each words.

**Question:** 
1. Is the input embedding layer a learnable layer?
	Potential Answer: Yes, since it has tied weights with the output layer that performs the reverse operation of embedding to output transformation.
 2. Is the learning process of input embedding different for different transformer architecture? E.g. for Encoder-decoder models, encoder only models, and decoder models?
3. What embedding technique do you use for each model? word2vec, cbow, next word predicting language models? 

#### positional encoding
- move input embedding by same distance but in different angles, depending on the position.
- Inputs close together in time are similar in perturbation angle.

### Attention:
![image](</Images/Pasted image 20240717083901.png>)
	![image](</Images/Pasted image 20240716165435.png>)
	The context weight assigned to each word representation V is computed by taking the Softmax of the dot product of the i'th Query vector and j'th Key vector for each j'th word.
![image](</Images/Pasted image 20240717154903.png>)

**Query**: builds question
**Key**: interacts with queries, distinguishes objects from one another, identify which object is relevant to query and by how much.
**value:** actual details of object

Similarity measure between Query and Key is put into a softmax to calculate the probability weights assigned to the Value of each elements of the the sequence.

The QKV have weights that are learnt via training.
	These QKV are specific to each word of an sequence.

The learning of attention weights can be parallelized, so the embeddings for words of the sequence can be processed simultaneously.



![image](</Images/Pasted image 20240716165155.png>)
	**Figure: self-attention in encoder**
	- softmax result becomes the relative weight assigned to that word embedding. 
	- The weighted sum of each word embedding of the sequence becomes a rich embedding of the an element of the sequence that includes the contextual information now, via the weighted summing over all words of the sequence.


**Self attention vs cross attention**
Self **attention**: The input element asks all other input elements, what do I mean relative to you guys?
- The input for query is from the input embedding, and not from the output embedding.

**Cross attention**: The output element asks the questions to all the input embeddings, what do you mean to me?
- The input for query is from the output, the Key and value are from the input.


<br>

**Multi-head attention**
Multiple heads allows learning of different contextual information for single token.
	Different heads learn independent projection matrices.

**Question:** is it akin to having multiple filters of CNN?

**You can keep the total number of parameters of post attention embedding same as the pre-attention embedding dimensions.**
	Dimension of individual head projections = pre-attention input embedding dimensions / number of heads
	We want to preserve the information in the input dimensions while not making the computation too expensive. We  want the vector representation each token to roughly remain the same before and after attention. So if the input was initially embedded using 256 length vector, and we are using 2 attention heads, each attention head would output an embedding of 128, and these two embeddings will be concatenated to produce a final embedding from the multi-head attention of length 256.

### Feed forward
![image](</Images/Pasted image 20240716170959.png>)
The feed forward layer is followed by an activation function.
	- Learn non-linearities.
	- Complex relationships
	- learn from other embeddings of the sequence.

Question: 
- the feed forward layer is word specific? ie, we don't merge the embeddings of different words yet. 

### Add and norm
**Adding residuals:** 
You add identity connections from original input to the post feedforward and normalized input. This makes learning deeper network more efficient because the feedforward layer has to learn parameters that are residuals from the identity connections, which is easier to learn. (Resnet)

Question: 
1. Does residual connections also prevent vanishing gradients by providing route for gradient descent to propagate?

**Normalization of inputs has two effects.**
1. stabilize training by preventing vanishing gradients.
	1. Activation function often have bigger gradients near 0, and decreases further away from 0.
2. Regularize model by adding randomness in input data.
	1. subtracting by sample mean then dividing by sample standard deviation adds randomness because the sample mean and standard deviation is random, because it sampled. ie. each batch has differen mean and sd, and each instance gets assigned a different batch at each epoch.



## Decoder:
**Target sequence is also tokenized, embedded, and positionally encoded**.

**The main tasked performed by decoder is the language model decoding which is: given preceding output tokens, predict current output token. Aka the teacher-forced learning.**
- In a encoder-decoder transformer model, it additionally has information about the hidden representations of the input. It has access to the entire sequence of the input, and partial access to the output(only to elements of output sequence that has already been generated).
- It accesses the input sequence via cross attention mechanism, which provides contextual information of the elements of the input w.r.t. to the previous output element. 
- It also accesses the output sequences that has already been generated via masked self attention. Reasons for masking is explained below.


**A key paradigm shift from RNN to transformers is that parameter update in learning the next token given previous tokens can be done in parallel.** For the entire output sequence, we can predict the next token given the pervious tokens for all elements of the sequence in parallel. In RNN's we had wait for the previous sequence to be generated to predict the next sequence because of the autoregressive structure and the recurrent structure where the previous hidden state becomes the input to the current hidden state.

This is true when we train our models, because we have access to the entire desired output sequence. However during inference, we don't have the desired output sequence, but we have our predictions of the output sequence. In order to predict the next token, we need all previous predictions of the sequence, thus we need t wait for all tokens prior to be generated.



### Masked multi head attention: decoder self attention
This means that the output at time t has no access to information at t+1 and on ward. So during training, to implement attention mechanism to add contextual information to tokens, we need to mask the attention on elements of training sequence that are not available at time t. This is because during inference mode, we will not have access to tokens at t+1 and onward.

We implement masking by adding a negative infinity to the pre-softmax QK/sqrt(d) terms that come after t. This means that after softmax, the attention weight on all elements that come after time t are equal to 0. This ensures that we don't assign weights to tokens of the sequence that are in the future. Although they are available during training, they will not be available during inference mode. Only output tokens prior to t are available during inference, so we will only assign weights to them, and the weights will sum to 1 because we added the negative infinity before the softmax function.

![image](</Images/Pasted image 20240716181130.png>)
	Figure: Masking of attention weights of output tokens that will not be available during inference of t'th output.

![image](</Images/Pasted image 20240716181534.png>)
	Figure: We mask the attention by adding a negative infinity term to the pre-softmax QK^T values.

![image](</Images/Pasted image 20240716181554.png>)
	Figure: The weighted sum of all of the output value. At t'th output, we get the weighted sum of the output vectors (with dimensions dh) from time 0 to t-1. 
### Encoder decoder cross attention
Encoder decoder cross attention is when the output element asks contextual information to all elements of the input sequence.
The elements of the decoder Supplies the input for the Query matrix, and the Encoder supplies the input for the Key and Value matrix.

### Block structure of encoders and decoders
Inputs and outputs of decoders and encoders have the same structure. This means that encoders/decoders can be stacked such that the output of one can become the input of the next block.

**For cross attention, every decoder block receives the final encoder output.** 
	If the encoder has a multi-head attention, the output of each attention is concatenated, then you can also add a MLP block. The final output of the MLP layer will be used as the hidden state representation of each input token. Cross attention can be applied on this final output, by generating the Key and Value from the encoder, and the Query from the decoder. These dimensions must match. 


### Linear:
This layer maps the output of the decoder to the output token dimension. The dimensions of the final output is number of elements in sequence x number of possible tokens/output.
There are one node for each possible output token. The weight matrix for each node is Vxdmodel where V is the number of possible output token, and d_model is the number of dimensions of the embeddings.

Weight tying with input embedding matrix:
- the weights of the this linear layer is tied with the input embedding matrix.
- This linear operation is the inverse operation of the embedding matrix. The embedding matrix projects the one-hot encoded token id into an embedding vector. This operation converts the embedding vector to a token vector.
- Question: How do you do weight tying if the input token and the output tokens are not the same? For example when you translate german to english?
	- https://stackoverflow.com/questions/49299609/tying-weights-in-neural-machine-translation
	- This stack overflow has:
		- embedding dimension of 300.
		- hidden size of 600.
		- output: 50000 categories(V). 
	- Linear layer has dimensions of: 50000 x 600
	- embedding dimensions have: 300 x 50000, under the assumption that the input vocabulary has the same size.
	- In this case, we would have to make the linear dimension 50000 x 300, so that its tied with the embedding dimensions 
	- Prior to the final linear layer that projects to the output dimensions, we can add another linear layer that projects from 600 hidden dimensions of the decoder to 300 hidden dimension. 
	- 
- 

![image](</Images/Pasted image 20240716184907.png>)
	Figure: The linear layer projects from the dimensions of the decoder to the output dimensions V. Multiply Td x D_model matrix by (V x D_model).transpose gives Td x V matrix, as shown in the softmax figure below.

### Softmax:
This layer converts the unnormalized probabilities and normalizes them. 

![image](</Images/Pasted image 20240716184833.png>)
	Figure: Softmax. Td is the number of output sequence. V is the output dimensions. Softmax is applied across V for each Td.


## Transformers can be decoder and encoder only
**Decoders can be encoders.**
	Because decoders can generate hidden representations prior to output layer.
	It generates hidden representation by predicting the hidden representation of the next token given all previous tokens it has generated.
	The hidden representation of all previous tokens are generated using the causal attention masks, which limits attention to preceding tokens.

Question: does this mean that the hidden representation of a given token may differ based on which output token we are currently inferring?
	Potential Answer: Yes. Because during masked self attention, weights are generated only for currently available tokens, and weights sum to 1. For example, the weights assigned for <sos> token will be 1 by default when predicting the next output token, because <sos> token is the only token available. When more tokens are generated, the <sos> token will no longer have a weight of 1. 

Question: Where is the Q generated from during masked self attention?
	Answer: Its generated from the output token for which we are generating the self-attention weight for. K are from all other output tokens that currently exist, includig the self. 
		![image](</Images/Pasted image 20240716165155.png>)


**Encoders can also generate tokens.**
	add linear layer and softmax layer to map hidden state dimensionality to vocab size.


**Question: But then whats the difference between encoder only and decoder only transformers?**
	The length of output sequence of an encoder-only transformer is fixed and equal to the length of the input sequence.
	
The decoder only transformer is trained on fixed length output, but during inference, it generates output tokens until <eos> token is outputted? There may be various heuristics associated with this.

Decoder has auto-regressive structure, where the previous output token prompts the next output, and also a causal attention where it attends to all previous tokens only. Encoder doesn't have this auto-regressive structure. It needs access to all input tokens, even those in future.

Decoder's attention is often masked to see only output tokens currently available/generated. Encoders attention has access to the entire input sequence.


**Question: Whats the difference between a encoder and a RNN?**
	The difference is the recurrence? Encoder takes away the recurrent nature where the model feeds the hidden representation of the input at time t to generating the hidden representation of the input at time t+1. This is replaced with a self-attention mechanism with positionally encoding, where each word attends to all words within a sequence with learned weights.
		
	![image](</Images/Pasted image 20240709211644.png>)

![image](</Images/Pasted image 20240716222042.png>)

# part 2. training and inference
Training of transformers can be divided into three steps:
1. Pre-training
	During this phase, you train the transformer without a specific task in mind.
	You train with lots of data.
2. Fine-tuning
	This is the phase where you adapt to specific tasks.
	You need task specific data set as a result.
3. Inference


### Parameter efficient fine tuning techniques
These are strategies to fine tune the model efficiently by training smaller amounts of parameters, but that still would tune the model to the task.

For example LoRA (lower rank adaptation) decomposes the parameter weights matrix into a product of its lower rank matrices. Then you train only the lower rank matrices to compose the weight matrix.

**Bitfit:**
Bitfit freezes the weight matrix term and just trains the bias term. THeres lots of leverage in just changing the bias term because of the later non-linear activation layers.
![image](</Images/Pasted image 20240716223046.png>)
# part 3. Transformer applications
### computer vision:
Vision transformers: (ViT)
1. Split image into fixed size patches.
2. tokenize each patch: linear projection of flattened patches
3. add position embedding
	1. This provides grib like structure of image patches?
4. feed resulting sequence of vectors with patch and position embedding to a transformer encoder.
5. For classification, add linear layer that project from transformer hidden representation dimensions to output class dimensions.
	What's the output sequence structure for ViT?
	
![image](</Images/Pasted image 20240716223454.png>)

### Multimodel transformers
separate encoder blocks for different modalities.
Concatenate the outputs of different encoders.
Jointer decoder with query embeddings specific for tasks.
task specific heads makes final outputs for each task.

# part 4. LLM
2017: encoder-decoder language models
2018: LLM. encoder and decoder only models
	BERT is the popular encoder only model.
	GPT is the popular decoder only LLM.

## Bidirectional encoder representations
The biggest challenge they wanted to solve was the lack of task specific data. There is a bottle neck in labelled training data. To fine tune larger models you need large amounts of task-specific labelled training data.

Solution they tried was: learning an effective enough representation of word embeddings so that the variety of tasks can be solved without task-specific data. This relates to [[L20>) which discusses how in LLM's, the task is represented not as hot-one encoded vectors, as traditional classification tasks are, but as natural languages. This means the problem is an open set, as opposed to a closed set. 

**To build better word representations, they had three main methods. 1) large training corpus, 2) pre-training task aimed at learning word embeddings, 3) and large number of parameters.**

**Whats the output structure of BERT models?**
What's the input? Its a sequence of tokens, that are one-hot encoded, according to the vocabulary size.
The output is a sequence of tokens, that have the dimension specified by the model. This is the dimension of the word embedding.
Question: Because its an encoder only model, its output sequence length is equal to the input sequence length?

**We learn this embedding of words via Masked Language Modeling.**

![image](</Images/Pasted image 20240717105050.png>)
	Figure: Masked language model. You mask a token in a input sequence. You train the model to predict the token of the masked token, by adding a prediction head only to the masked token. The prediction head would project from the embedding dimensions to the vocabulary dimensions. 

**Next sentence prediction**
![image](</Images/Pasted image 20240717105925.png>)
	Give bert sequence of two sentences that can be adjacent to each other in the original corpus, or random two sentences. Then train the model to predict whether the two sentences are adajcent or not via attaching a prediction head on the <cls> classifier token that is at the beginning of the two sentence sequence. This prediction task trains the embeddings to include the relationships between words in multiple sentence structure. 

**Question: How does learning for other word embeddings occur when the prediction head is attached to the classifier head?**
	The hidden representation of the <cls> token has a self attention mechanism that makes it a weighted sum of all other tokens within the sequence. Then the backprop flows through all other tokens of the sequence.

**Fine tuning of Bert**
For classification task, add prediction head on cls token.
For question and answering, add two vectors, start of question and end of question at input vectors of question?

## Takeaways from BERT
1. Pretraining tasks can be general.
	1. you can learn represenations of words from large corpus that is not necessarily task specific, but the representation can be used to solve specific tasks.
2. Different NLP tasks are transferable.
3. Scaling works

 ## GPT
 Generative pretrained transformer: decoder only model.
 Motivation: large amounts of unlabelled data to train LM.

task: Predict next token given previous tokens.

How to use GPT?




 