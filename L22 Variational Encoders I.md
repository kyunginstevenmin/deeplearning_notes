### Generative models:
We want to make networks that can generate data. Given an training data, can we make a NN that can generate new samples?

The structure of generative models in statistics is a model distribution of data that we draw samples from according to that distribution. The probability model that best describes the sample distribution is the probability distribution we use to generate more samples from. Examples of probability distributions are multinomial, gaussian distribution.

**How do we know what parameters, such as mean, and variance to use if we want our probability model to be most like the sample that we observed?**
Given set of observed data X = {x}, we learn probability model P(x;theta) such that the model best fits the observations.

**The model that best 'fits' the data is defined in terms of Maximum likelihood.**
	We select the model that has the highest probability of generating the observed sample data. 
![image](<Pasted image 20240719142626.png>)


Best estimators for the parameters for multinomial and gaussian distributions are in red.
![image](<Pasted image 20240719142659.png>)


## Maximum likelihood estimation with incomplete data
But sometimes there are missing values that makes it difficult to use these estimators. There may be incomplete data because of 1) missing data, 2) the structure of probability model we are trying to model. For example, mixture models, we don't a priori know which distribution each data is generated from.

### Missing data:
![image](<Pasted image 20240719150502.png>)


![image](<Pasted image 20240719150707.png>)


### Missing value due to Gaussian mixture model structure
![image](<Pasted image 20240719151006.png>)
![image](<Pasted image 20240719151050.png>)

So to estimate the model parameters, you would need to know which gaussian distribution the data is from, as well as the data vector.
You would separate data according to the k'th gaussian they belong to, then find the mean and variance that with the maximum likelihood of producing the data.
	![image](<Pasted image 20240719151433.png>)

But we don't know to which gaussian model each observation belongs to.
	The probability of an individual vector is the marginalized probability of the observation over all gaussian models.
	The maximum likelihood function is maximizing the sum of log probability of every observation, and the probability of each observation has a summation term. This log of sum is difficult to optimize.
	![image](<Pasted image 20240719151622.png>)

The general form of the problem:
- When there are missing values or variables, we calculate the probability of the observed values by marginalizing over the missing values or variable by summation or integral of the joint probability of missing and hidden values.
- But this means the formula contains a log sum term that makes optimization difficult.
- We estimate optimization of the ML estimate by method of variations.
![image](<Pasted image 20240719151943.png>)

### optimize variational lower bound as a proxy/alternative for the true likelihood function
Because its difficult to optimize log(P(O)) directly, we optimize a variational lower bound function of it. This technique utilizes the property of log functions.
![image](<Pasted image 20240719152714.png>)
![image](<Pasted image 20240719152834.png>)
![image](<Pasted image 20240719152859.png>)
Our objective is to maximize the likelihood fuction, but instead maximize the variational lower bound function. We do this in two step process.
1) Determine Q(h) that makes the lower bound tighter, while the parameter is fixed.
	1) the ideal Q(h) without constraints is P(h|o;theta). We can use our initial estimate of theta as Q(h).  
2) with Q(h) fixed, maximize the lower bound w.r.t. to theta.
	1) This finds the theta at which the variational lower bound function is maximized.
	2) Use this theta as the new parameter for Q(h). 



## Missing values again
![image](<Pasted image 20240719154050.png>)
Q. if we expand incomplete vectors while not expanding complete vectors, won't this change the distribution of our data?


![image](<Pasted image 20240719161705.png>)
- Use our current estimation of theta, to fill in missing value.
	- The integral term is the conditional expectation of the (m,o) given the observed value o. 
- We sum the conditional expectation for all observed values, which is the same as the unconditional expectation of our data. Its the sample mean as well.
	- This is our estimate of the mean after filling in the missing values according to P(m|o) according to our previous best estimate of theta.
	- This parameter will be used to get a new estimate of the missing values. 


How to solve this iteratively?

	![image](<Pasted image 20240719161811.png>)


### GMM Problem of incomplete data: missing gaussian identity
If we knew which gaussian each data point belonged to we would be able to separate data according to the gaussian to which they belong, then then use sample mean and variance estimator to estimate the parameters for each gaussian. But we don't have that. 
![image](<Pasted image 20240719162241.png>)
Simple solution: assign k to each data according to the proportion P(k|o) = Whats the likelihood of being assigned to k'th gaussian given the observation vector. 

We can compute P(k|o) using previous model parameters. P(O|K) is the gaussain parameter. P(K) is the weight parameter. P(K|o) = P(k,o)/P(o)

With completed data, re-estimate the parameters.
	![image](<Pasted image 20240719163047.png>)
	
question: does drawing multiple samples from original observation distort the distribution?
Answer: No, as long as you draw equal number of complete data from each observations.

Then you segregate vectors according to the gaussian. The proportional of observation that belongs to k'th gaussian will be P(K|O). 

![image](<Pasted image 20240719164326.png>)

Problem: How do we estimate P(m|o;theta') if its not tractable?
solution: use neural network estimate of p(m|o).
- how do you get a neural network estimate of p(m|o)?

## Principal Component analysis
Find principal subspace such that when all the vectors are approximated as lying on that subspace, the approximation is minimal.
	Why is the objective minimizing approximation/total squared error?
	Its the subspace direction such that if you move along it, and project back onto the input dimensions, we are most likely to produce least error?
	Conversely, it is also the direction at which the variance in the original space is greatest. But the direction at which the variance in the orthogonal directional is the lowest. But why does it have to be this way?
![image](<Pasted image 20240720082347.png>)

