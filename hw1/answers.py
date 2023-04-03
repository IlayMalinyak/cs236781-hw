r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
The loss is computed only on wrong clasiffications and its goal is to make sure
all wrong classes will get loss higher by at least $Delta$ from the true class. as long 
as this work, the true class will be classified correctly and therefore the size of $Delta$ 
is not important. its only role is to seperate between wrong classes and true class

"""

part3_q2 = r"""
**Your answer:**

1. Based on the visualitation we can notice that each class (row) of weights learns a representation of 
the shape of the corresponding class label. the representation is based in the training set and if we have
in the test set an image which is very different from the representation of its label in the train set, but 
more close to a different representation, the model will miscalsiffied the image as the later class. we can see it happened between
digits 7 and 1, digits 7 and 9 and digits 5 and 2 (these are similiar digits so the probability for this to occur in those digits is higher).
2. In KNN there is no actual learning. the classification in based only on neighbouring samples. In LinearClassification model there is no affect of neighbouring samples. The model learn the unique features of each label and classify based on the similiarity between sample's features and the learned features.  

"""

part3_q3 = r"""
**Your answer:**

1. We would say the learning rate is good. If it was too high, the loss would bounce between approximatley the same values and 
would not decrease. in case of too low learning rate the loss would decrease very slowly and we would not reach saturation (plateau in the loss and accuracy). since in our graphs we see steep decrease (or increase in the case of accuracy) and than a more flat area, in conclude that the learning rate is good.

2. **wait for better results**

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
