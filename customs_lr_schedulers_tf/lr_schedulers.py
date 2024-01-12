import tensorflow as tf
import numpy as np


def estabilidad(eigenvals):
    eig_1 = eigenvals[0]
    eig_2 = eigenvals[1]
    if eig_1.imag == 0:
        if eig_1.real < 1:
            return 'estable'
        elif eig_1.real == 1:
            return 'neutral'
        elif eig_1.real > 1 and eig_2.real <= 1:
            return 'saddle point'
        else:
            return 'unstable'
    else:
        if abs(eig_1) < 1:
            return 'estable spiral focus'
        elif abs(eig_1) == 1:
            return 'neutral center'
        else:
            return 'unstable spiral focus'


class StepLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, init_lr, step_size, gamma, verbose=0):
        self.step_size = step_size
        self.gamma = gamma
        self.verbose = verbose
        self.current_lr = init_lr

    def __call__(self, step):
        if step % self.step_size == 0:
            self.current_lr *= self.gamma ** (step // self.step_size)
            if self.verbose:
                print(f"Updated lr in step:{step}")
        return self.current_lr


class ReduceLROnPlateau(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Args:
        factor: Float. Factor by which the learning rate will be reduced.
            `new_lr = lr * factor`.
        patience: Integer. Number of epochs with no improvement after which
            learning rate will be reduced.
        verbose: Integer. 0: quiet, 1: update messages.
        min_delta: Float. Threshold for measuring the new optimum, to only focus
            on significant changes.
        cooldown: Integer. Number of epochs to wait before resuming normal
            operation after the learning rate has been reduced.
        min_lr: Float. Lower bound on the learning rate.
    """

    def __init__(self, init_lr=0.01, factor=0.1, patience=2, verbose=0,
                 min_delta=1e-4, cooldown=0, min_lr=0):

        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr

        self.cooldown_counter = 0
        self.wait = 0
        self.best = np.Inf

        self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
        self.metric = None
        self.current_lr = init_lr

    def __call__(self, step):
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0
            return self.current_lr
        # si mejora la metrica en al menps min_delta
        if self.monitor_op(self.metric, self.best):
            self.best = self.metric
            self.wait = 0
            return self.current_lr
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = self.current_lr
                if old_lr > np.float32(self.min_lr):
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
                    if self.verbose:
                        print(f"Updated lr in step: {step}")
                    self.current_lr = new_lr
                    return new_lr
            return self.current_lr

    def in_cooldown(self):
        return self.cooldown_counter > 0
