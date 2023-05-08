r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.15
    lr = 0.05
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg, = 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    if opt_name == 'vanilla':
        wstd = 0.1
        lr = 0.001
        reg = 0.001
    if opt_name == 'momentum':
        wstd = 0.1
        lr = 0.001
        reg = 0.001
    if opt_name == 'rmsprop':
        wstd = 0.1
        lr = 0.001
        reg = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 1e-4
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. The results we got are in line with what we expected.
It can be seen that the order of the loss values decreases as the dropout intensity is reduced (both in train and in test).
In the same way, the order of the accuracy values increases.
It can be seen that for a dropout value of 0.4, results closer to the model without dropout were obtained for the test set, which means that the generalization ability was good.
For a dropout value of 0.8 the regularization was too strong and harmed the model's ability to learn and generalize. </br>
2. As we wrote in the previous question, dropout with a probability of 0.8 damaged the model's ability to learn and generalize and this can also be seen in the high loss value in the test set (compared to values of 0 and 0.4).
Similarly, the accuracy is almost 10% lower compared to the other 2 models.
"""

part2_q2 = r"""
Yes, it is possible that the test loss and test accuracy values will both increase because these two measurements do not measure the same thing.
While Accuracy measures the proportion of correct (discrete) predictions, the CE loss measures the "magnitude" of the error expressed in the log "distance" for the correct label (continuous since the log function is continuous).
If so, there may be situations in which it will be possible to increase the model error (by reducing the score of the correct label for a certain sample) but still keep the sample correctly classified (the score will still be the highest) and on the other hand another sample that was on the threshold and was not classified correct, will get a higher score which will change its classification to the correct one and thus the accuracy alongside the loss which can increase as well.
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


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
