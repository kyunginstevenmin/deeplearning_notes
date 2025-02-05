RNN can be used not only for time series data but also for recursion problems?
- much more efficient that simply MLP.
	for parity problem, MLP requires extremely large amounts of data to train model.
	RNN uses recurrent structure of problem to solve problem with less data.


Story so far: recurrent network learns by minimizing divergence between sequence of output and sequence of desired output.

in recurrence, past events have continuing influence on future.

## BIBO stability: bounded input bounded output
For CNNs, if the inputs are bounded, then the output is also bounded.

for RNN's, because inputs are bounded, we only need to look at recurrent layers to see BIBO stability.
	further assume linear activation at recurrent layer.
	![[Pasted image 20240629121549.png]]


![[Pasted image 20240629121626.png]]
- weights Wh are fixed. Wx are also fixed.


Paper by Bengio et al. on why RNN has problem with  long term memory.
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=279181
