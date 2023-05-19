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
In our custom network, we added:
1) Dropouts with p=0.5 - in order to add regularization for mitigating the overfitting
2) Batch normalization - in order to tackle the vanishing\exploding gradient
We used hidden_dims=[512,512] and filters_per_layer=[64,128,256,512].

Our architecture reflects its improvements in 2 main results:
1) We were able to train the model with L>2, as opposed to the previous experiment.
2) We achieved test accuracy of 82% which is more than 10% improvement.
"""
