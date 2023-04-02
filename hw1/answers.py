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

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
