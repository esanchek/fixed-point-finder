'''
Pytorch FixedPointFinder
Written for Python 3.8.17 and Pytorch 2.0.1
@ Matt Golub, June 2023
@ with contributions from Alexander Ladd, 2022

If you are using FixedPointFinder in research to be published, 
please cite our accompanying paper in your publication:

Golub and Sussillo (2018), "FixedPointFinder: A Tensorflow toolbox for 
identifying and characterizing fixed points in recurrent neural networks," 
Journal of Open Source Software, 3(31), 1003.
https://doi.org/10.21105/joss.01003

Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb

import numpy as np
import time
from copy import deepcopy

# import torch
# from torch.autograd.functional import jacobian

import tensorflow as tf

from FixedPointFinderBase import FixedPointFinderBase
from FixedPoints import FixedPoints


class StepLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, init_lr, step_size, gamma):
        self.init_lr = init_lr
        self.step_size = step_size
        self.gamma = gamma
        self.current_lr = init_lr
        self.metric = None

    def __call__(self, step):
        if step % self.step_size == 0:
            new_lr = self.current_lr * self.gamma ** (step // self.step_size)
            self.current_lr = new_lr
            print(f"In step {step} new learning rate is {new_lr} due to q_scalar: {self.metric}")
        return self.current_lr
        #    (self.init_lr * self.gamma ** (step // self.step_size))


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

    def __init__(self, init_lr=0.01, factor=0.1, patience=10, verbose=0,
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
                    # print(f"In step {step} new learning rate is {new_lr} due to q_scalar: {self.metric}")
                    self.current_lr = new_lr
                    return new_lr
            return self.current_lr

    def in_cooldown(self):
        return self.cooldown_counter > 0


class FixedPointFinderTF2(FixedPointFinderBase):

    def __init__(self, rnn,
                 lr_init=1.0,
                 lr_patience=5,
                 lr_factor=0.95,
                 lr_cooldown=0,
                 **kwargs):
        '''Creates a FixedPointFinder object.

        Args:
            rnn: A Pytorch RNN object. The following are supported: nn.RNN, nn.GRU,
            or any wrapper class that matches the input/output argument 
            specifications of output, h_n = nn.RNN(input, h0). 
            
            lr_init: Scalar, initial learning rate. Default: 1.0.

            lr_patience: The 'patience' arg provided to ReduceLROnPlateau().
            Default: 5.

            lr_factor: The 'factor' arg provided to ReduceLROnPlateau().
            Default: 0.95.

            lr_cooldown: The 'cooldown' arg provided to ReduceLROnPlateau().
            Default: 0.

            See FixedPointFinderBase.py for additional keyword arguments.
        '''
        self.rnn = rnn
        # self.device = next(rnn.parameters()).device

        self.lr_init = lr_init
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_cooldown = lr_cooldown

        super().__init__(rnn, **kwargs)
        self.tf_dtype = getattr(tf, self.dtype)

    def _run_joint_optimization(self, initial_states, inputs, cond_ids=None):
        '''Finds multiple fixed points via a joint optimization over multiple
        state vectors.

        Args:
            initial_states: An [n x n_states] numpy array specifying the initial
            states of the RNN, from which the optimization will search for
            fixed points.

            inputs: A [n x n_inputs] numpy array specifying a set of constant
            inputs into the RNN.

        Returns:
            fps: A FixedPoints object containing the optimized fixed points
            and associated metadata.
        '''

        n_batch = inputs.shape[0]

        # Ensure that fixed point optimization does not alter RNN parameters.
        print('\tFreezing model parameters so model is not affected by fixed point optimization.')
        self.rnn.trainable = False

        self._print_if_verbose('\tFinding fixed points via joint optimization.')

        with tf.device('GPU'):
            inputs_bxd = tf.constant(inputs, dtype=self.tf_dtype)
            x_bxd = tf.Variable(initial_states, dtype=self.tf_dtype, trainable=True)  # to(self.device)

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_init)   # to(self.device)
            # scheduler = StepLR(init_lr=self.lr_init, step_size=500, gamma=0.95)  # step_size=500, gamma=0.7)
            scheduler = ReduceLROnPlateau(init_lr=self.lr_init, factor=0.95, patience=2, cooldown=0, min_lr=1e-5)

            q_prev_b = tf.zeros((n_batch,), dtype=self.tf_dtype)   # device=self.device)

        iter_count = 1
        t_start = time.time()

        while True:
            with tf.device('GPU'):
                with tf.GradientTape() as tape:
                    # for each batch row: state and its associated input:
                    _, F_x_bxd = self.rnn(inputs_bxd, x_bxd)
                    dx_bxd = x_bxd - F_x_bxd
                    # for each batch row: 1/2 (x - Fx)^2
                    q_b = 0.5 * tf.reduce_sum(tf.square(dx_bxd), axis=1)
                    # mean for all batch
                    q_scalar = tf.math.reduce_mean(q_b)

                # optimizer step vs joint q: q_scalar
                grads = tape.gradient(q_scalar, [x_bxd])
                optimizer.apply_gradients(zip(grads, [x_bxd]))

                scheduler.metric = q_scalar
                iter_learning_rate = scheduler(step=iter_count-1)
                optimizer.learning_rate = iter_learning_rate

            # convert to numpy
            dq_b = tf.math.abs(q_b - q_prev_b)
            ev_q_b = q_b.numpy()              # to_cpu
            ev_dq_b = dq_b.numpy()            # to_cpu

            if self.super_verbose and np.mod(iter_count, self.n_iters_per_print_update) == 0:
                self._print_iter_update(iter_count, t_start, ev_q_b, ev_dq_b, iter_learning_rate)

            # stop condition 1: (tolerances) in all batch rows tolerance has been reached
            if iter_count > 1 and np.all(np.logical_or(ev_dq_b < self.tol_dq * iter_learning_rate,
                                                       ev_q_b < self.tol_q)):
                '''Here dq is scaled by the learning rate. Otherwise very small steps due to very 
                small learning rates would spuriously indicate convergence. This scaling is roughly 
                equivalent to measuring the gradient norm.'''
                self._print_if_verbose('\tOptimization complete to desired tolerance.')
                break

            # stop condition 2: reached max num iterations
            if iter_count + 1 > self.max_iters:
                self._print_if_verbose('\tMaximum iteration count reached. Terminating.')
                break

            q_prev_b = q_b
            iter_count += 1

        if self.verbose:
            self._print_iter_update(iter_count, t_start, ev_q_b, ev_dq_b, iter_learning_rate, is_final=True)

        xstar = x_bxd.numpy()       # to_cpu?
        F_xstar = F_x_bxd.numpy()   # to_cpu?

        # Indicate same n_iters for each initialization (i.e., joint optimization)        
        n_iters = np.tile(iter_count, reps=F_xstar.shape[0])

        fps = FixedPoints(
            xstar=xstar,
            x_init=initial_states,
            inputs=inputs,
            cond_id=cond_ids,
            F_xstar=F_xstar,
            qstar=ev_q_b,
            dq=ev_dq_b,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.np_dtype)

        return fps

    def _run_single_optimization(self, initial_state, inputs, cond_id=None):
        '''Finds a single fixed point from a single initial state.

        Args:
            initial_state: A [1 x n_states] numpy array specifying an initial
            state of the RNN, from which the optimization will search for
            a single fixed point. 

            inputs: A [1 x n_inputs] numpy array specifying the inputs to the
            RNN for this fixed point optimization.

        Returns:
            A FixedPoints object containing the optimized fixed point and
            associated metadata.
        '''

        return self._run_joint_optimization(initial_state, inputs, cond_ids=cond_id)

    def _compute_recurrent_jacobians(self, fps):
        '''Computes the Jacobian of the RNN state transition function at the
        specified fixed points assuming fixed inputs for each fixed point
        (i.e., dF/dx, the partial derivatives with respect to the hidden 
        states). Implementend as a batch Jacobian

        Args:
            fps: A FixedPoints object containing the RNN states (fps.xstar)
            and inputs (fps.inputs) at which to compute the Jacobians.

        Returns:
            J_np: An [n x n_states x n_states] numpy array containing the
            Jacobian of the RNN state transition function at the states
            specified in fps, given the inputs in fps.
        '''

        inputs_np = fps.inputs
        states_np = fps.xstar

        inputs_bxd = tf.constant(inputs_np, dtype=self.tf_dtype)             # to(self.device)
        x_bxd = tf.Variable(states_np, dtype=self.tf_dtype, trainable=True)  # to(self.device)

        with tf.GradientTape(persistent=True) as tape:
            output, F_x_bxd = self.rnn(inputs_bxd, x_bxd)

        J_bxdxd = tape.batch_jacobian(F_x_bxd, x_bxd)
        J_np = J_bxdxd.numpy()
        return J_np

    def _compute_input_jacobians(self, fps):
        ''' Computes the partial derivatives of the RNN state transition
        function with respect to the RNN's inputs, assuming fixed hidden states.

        Args:
            fps: A FixedPoints object containing the RNN states (fps.xstar)
            and inputs (fps.inputs) at which to compute the Jacobians.

        Returns:
            J_np: An [n x n_states x n_inputs] numpy array containing the
            partial derivatives of the RNN state transition function at the
            inputs specified in fps, given the states in fps.
        '''

        return None
