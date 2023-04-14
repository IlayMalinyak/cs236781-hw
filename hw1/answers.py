r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1. False. the in sample error comes from the data that the model was trained on - train set. thet test set represent the out of sample error.
2. False. the amount of data in the train set determin the in-sample error (or bias error) which reflect the ability of the model to learn the relation between features and samples. different splits results in different in-sample error, different weights parameters and therefore different performance.
3.True. during cross validation we should not use the test set in order to prevent leakage. the test set should be use only for model evaluation.
4. False. the validation error at each fold only determine the model parameters inside the cross validation procedure (as each fold convert one train set to a validation set, using validation error as proxy for generalization error is biased)
"""

part1_q2 = r"""
**Your answer:**
no. choosing values for model hyperparameters should be done using training-set and cross validation procedure. using the test-set for that would cause leakage of the test-set into the training procedure. this way, any evaluation of the model using the test-set would be biased and would not reflect a true generalization error.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
in KNN, K is related to bias variance trade-off. when K=1 the model 
fits exactly to the training data but will likely fail to generalize on new data.
in terms of bias-variance this case correspond to low bias and high variance.
the other extreme is the when K=N. now the model is underfitted - it will likely fail to learn the features 
from the training set and therefore will also fail to generalize (under the assumption that the new samples comes from the same 
distribution). this case correspond to high bias and low variance. the "sweet spot" is the minimum if the total error 
$err_{tot} = bias^2+vairance+noise$. we conclude that increasing K will reduce to generalization
error up to a point where the total error starts to increase. 

"""

part2_q2 = r"""
**Your answer:**

1. training on the entire train set and selecting the model with respect to the same dataset can lead to overfitting. we want to select the model based on out-of-sample error, and by using the same dataset for training and for evaluation we will not get this. using CV we will get that since in each fold there is strict seperation between train and validation sets.
2. using test-set to select the model will cause data leakage. after that, we cannot make real evaluation for the model since our test-set was contaminated when we used it for model selection. using CV, we use only the train set (splitted into train and validation) for the model selection. this leaves the test set uncontaminated and thus we can use it to evaluate the model. 
```
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
$\Delta$ is irrelevant because we have the regularization term, $\lambda$, which we can tune. as $\Delta$ sepcify the margin size, $\lambda$ allows samples to to be inside the margin. the "cost" of sample being inside the margin determined by $\lambda$. for a given dataset we can either change $\Delta$ and fix $\lambda$ or change $\lambda$ and fix $\Delta$ and there is a correspondence between the two cases (for each value of $\Delta$ with fixed $\lambda$ there's a value of $\lambda$ with fixed $\Delta$ that will have the same accuracy). in our case $\lambda$ id the hyperparameter so $\Delta$ can be chosen arbitrarily


"""

part3_q2 = r"""
**Your answer:**

1. Based on the visualitation we can notice that each class (row) of weights learns a representation of 
the shape of the corresponding class label. the representation is based in the training set and if we have
in the test set an image which is very different from the representation of its label in the train set, but 
more close to a different representation, the model will miscalsiffied the image as the later class. we can see it happened between
digits 7 and 2 and 7 and 9 (these are similiar digits so the probability for this to occur in those digits is higher).
2. In KNN there is no actual learning. the classification in based only on neighbouring samples. In LinearClassification model there is no affect of neighbouring samples. The model learn the unique features of each label and classify based on the similiarity between sample's features and the learned features.  

"""

part3_q3 = r"""
**Your answer:**

1. We would say the learning rate is good. If it was too high, the loss would bounce between approximatley the same values and 
would not decrease. in case of too low learning rate the loss would decrease very slowly and we would not reach saturation (plateau in the loss and accuracy). since in our graphs we see steep decrease (or increase in the case of accuracy) and than a more flat area, in conclude that the learning rate is good.

2. based on the graphs, the model is slightly overfit. this can be seen from the graphs - the training accuracy (loss) is slightly higher (lower) than the validation and test accuracies (losses). the reason we are overfitting has might be the fact that the trainint-set is not representative enough - the train-set samples distribution is different than the test-set samples distribution. ther eare samples in the test-set that looks like samples from different class in the training set. this can lead to lower accuracy on the training-set and overfitting  

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


A graph of the $y-\hat{y}$ as a function of $\hat{y}$ makes it possible to see the error itself directly and clearly.

Conversely, a graph of $\hat{y}$ as a function of $y$ contains more information: it shows the prediction model itself, thus allowing it to be compared to the desired values $y$. This comparison allows us to understand where the error comes from. For example, we can see whether $y$ relate linearly to the $X$ values, and thus we will know whether the problem can be solved by mapping to another feature space, and also to which feature space (e.g. polynomial mapping), or perhaps the relationship between $X$ and $y$ is indeed linear but our model is far from the desired prediction because the slope is not appropriate, then maybe the regulation is too strong and we should weaken it.

Another difference is that a graph of $\hat{y}$ as a function of $y$ can only be displayed for one feature at a time, or at most 2 features (if it's a 3D graph).

So in summary, the graph of $y-\hat{y}$ as a function of $\hat{y}$ makes it possible to clearly see the error itself, while a graph of $\hat{y}$ as a function of $y$ makes it easier to see the cause of the error, but harder to see the error itself. Therefore, if we are still tuning the hyperparameters in the model we would prefer to see the graph of $\hat{y}$ as a function of $y$, and if we have already finished training the model and we just want to see how accurate it is, we would prefer the graph of $y-\hat{y}$ as a function of $\hat{y}$.

"""

part4_q2 = r"""
1) The model is no longer a linear regression model, because it does not find a linear relationship between $X$ and $y$. It does use linear regression as part of the model, but this use is inside a "black box": the model accepts inputs $X$, and returns $\hat{y}$ that are not linearly related to them.

2) In principle, we can fit any finite-time writable function. But in practice, the more parameters the function requires, the more difficult it will be to fit it without causing overfitting. Therefore we will prefer simpler functions.

3) Now the model will be a non-straight "hyperplane": a curved separator that still divides the space in two, but the boundary of the separation is not a linear hyperplane but curved in space.

"""

part4_q3 = r"""
**Your answer:**


1) The cross-validation process takes a long time. Therefore, we will prefer to shorten the search time by means of logarithmic compression of the search space.

2) I adjusted each of the hyperparameters separately, so the total number of fit operations is the sum of the parameter ranges:
$3+20+10+10+7 = 50$

If I were to do a grid search, I would get a much higher number:
$3*20*10*10*7 = 42000$

"""

# ==============
