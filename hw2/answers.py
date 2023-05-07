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
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg, = 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    if opt_name == 'vanilla':
        raise NotImplementedError()
    if opt_name == 'momentum':
        raise NotImplementedError()
    if opt_name == 'rmsprop':
        raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
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
# Part 3 answers

part3_q1 = r"""
1. Setting L=2 leads to the highest accuracy for K=32 while setting L=4 leads to the highest accuracy for K=64.
   It seams that low L is the most suitable hyperparameter for this network since the training is stable and the loss is decreasing monotonically.
2. Setting L to 8/16 wasn't trainable - the accuracy was static (~10%) and the loss was high. As we learned in the course, it might happen due to vanishing gradients because the loss is very static.
   We can solve it by adding skip connections to the network.
"""

part3_q2 = r"""
For L=2, we can observe that K=64 achieves the best results with ~80% accuracy on the training set.
For L=4, we can see the same but with K=128.
As we saw in the previous experiment, for L=8/16 the net wasn't trainable.
"""

part3_q3 = r"""
We can observe that for this setting of filters, only L=1 an L=2 were trainable while L=4 wan't as opposed to previous experiments.
Also L=2 results slightly higher accuracy than L=1 on the test set though the opposite on the training set.
This means that the deeper network was less overfitted and generalizes better. 
"""


part3_q4 = r"""
In our custom network we added skip connections to tackle the vanishing gradient that we encountered in the previous experiments.
Also, we added some regularizations:
1) droupouts with p=0.5
2) batch normalization
We used hidden_dims=[100,100] and filters_per_layer=[64,128,256,512].

Overall, we still see the vanishing\exploding gradient phenomenon and therefore the test accuracy is pretty the same as in 1.3 experiment.
"""
