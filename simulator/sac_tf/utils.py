import numpy as np

import keras as K
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Flatten, Softmax, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp


class SAC__BasePictureProcessor(tf.keras.Model):
    """
    simple picure processing, pretrained model + conv-conv-maxpool
    """

    def __init__(self, input_shape=(84, 84, 3), hidden_size=128):
        super(SAC__BasePictureProcessor, self).__init__()

        self.conv1_1 = Conv2D(
            filters=hidden_size,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            dtype=tf.float32
        )
        self.conv1_2 = Conv2D(
            filters=hidden_size,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            dtype=tf.float32
        )
        self.pool1 = MaxPool2D(pool_size=(4, 4))

        self.conv2_1 = Conv2D(
            filters=hidden_size,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            dtype=tf.float32
        )
        self.conv2_2 = Conv2D(
            filters=hidden_size,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            dtype=tf.float32
        )
        self.pool2 = MaxPool2D(pool_size=(2, 2))

        self.conv3_1 = Conv2D(
            filters=int(hidden_size / 2),
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            dtype=tf.float32
        )
        self.conv3_2 = Conv2D(
            filters=int(hidden_size / 4),
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            dtype=tf.float32
        )
        self.pool3 = MaxPool2D(pool_size=(2, 2))

        self.flatten = Flatten()

    def apply_picture_processing(self, picture):
        return self.flatten(
            self.pool3(self.conv3_2(self.conv3_1(
                self.pool2(self.conv2_2(self.conv2_1(
                    self.pool1(self.conv1_2(self.conv1_1(
                        picture,
                    )))
                )))
            )))
        )


class SAC__ValueNet(SAC__BasePictureProcessor):
    '''
    Implementaion of V function
    '''

    def __init__(self, picture_shape, extra_state_size, action_size, hidden_size=128, name='_v1'):
        super(SAC__ValueNet, self).__init__(picture_shape, hidden_size)
        self.d0 = Dense(units=hidden_size, activation='relu', dtype=tf.float32)
        self.d1 = Dense(units=hidden_size, activation='relu', dtype=tf.float32)
        self.d2 = Dense(units=hidden_size, activation='relu', dtype=tf.float32)
        self.value = Dense(units=1, activation=None, dtype=tf.float32)

        self.optimizer = Adam(0.003)

    def __call__(self, state):
        state_picture, state_extra = state
        return self.value(
            self.d2(self.d1(
                Concatenate(axis=1)([
                    self.apply_picture_processing(state_picture),
                    self.d0(state_extra),
                ])
            ))
        )


class SAC__QNet(SAC__BasePictureProcessor):
    '''
    Implementation of Q function.
    '''

    def __init__(self, picture_shape, extra_state_size, action_size, hidden_size=128, name='_v1'):
        super(SAC__QNet, self).__init__(picture_shape, hidden_size)

        self.d_extra_state = Dense(units=hidden_size, activation='relu', dtype=tf.float32)
        self.d_action = Dense(units=hidden_size, activation='relu', dtype=tf.float32)

        # self.concat = tf.concat([x_state, x_action], axis=1)
        self.d1 = Dense(units=hidden_size, activation='relu', dtype=tf.float32)
        self.d2 = Dense(units=hidden_size, activation='relu', dtype=tf.float32)
        # final shape [None, 1]
        self.qvalue = Dense(units=1, dtype=tf.float32)

        self.optimizer = Adam(0.003)

    def __call__(self, state, action):
        state_picture, state_extra = state
        x = Concatenate(axis=1)([
            self.apply_picture_processing(state_picture),
            self.d_extra_state(state_extra),
            self.d_action(action),
        ])
        x = self.d1(x)
        x = self.d2(x)
        return self.qvalue(x)


class SAC__Policy(SAC__BasePictureProcessor):
    '''
    Implementation of Policy function.
    '''

    def __init__(self, picture_shape, extra_state_size, action_size, hidden_size=128, name='_v1'):
        super(SAC__Policy, self).__init__(picture_shape, hidden_size)

        self.d_extra_state = Dense(units=hidden_size, activation='relu', dtype=tf.float32)
        self.state_concat = Concatenate()
        self.d1 = Dense(units=hidden_size, activation='relu', dtype=tf.float32)
        self.d2 = Dense(units=hidden_size, activation='relu', dtype=tf.float32)
        self.d3 = Dense(units=action_size, dtype=tf.float32)
        self.soft_max = Softmax(axis=1)
        self.soft_max_gumbel = Softmax(axis=1)

        self.optimizer = Adam(learning_rate=0.003)

    def gumbel_softmax(self, prob, temperature=0.5):
        u = np.random.uniform(low=0.0, high=1.0, size=(prob.shape)).astype(np.float32)
        u = -np.log(-np.log(u))

        return self.soft_max_gumbel((prob + u) / temperature)

    def __call__(self, state, temperature=0.5, use_gumbel=False):
        state_picture, state_extra = state
        probs = self.soft_max(
            self.d3(self.d2(self.d1(
                self.state_concat([
                    self.d_extra_state(state_extra),
                    self.apply_picture_processing(state_picture),
                ])
            )))
        )
        if not use_gumbel:
            return probs
        else:
            return self.gumbel_softmax(probs, temperature)


class SAC__Agent:
    '''
    Class of agent, which control update steps of all sub-net and can be used in exploitation.
    '''

    def __init__(self,
                 picture_shape,
                 extra_size,
                 action_size,
                 hidden_size=256,
                 name='agent_1',
                 info=''):
        # save meta info
        self.name = name
        self.info = info

        # save env hyper params
        self.action_size = action_size
        self.picture_shape = picture_shape
        self.extra_size = extra_size

        # here init agent nets
        self._Q1 = SAC__QNet(picture_shape, extra_size, action_size, hidden_size, '_q1')
        self._Q2 = SAC__QNet(picture_shape, extra_size, action_size, hidden_size, '_q2')
        self._V = SAC__ValueNet(picture_shape, extra_size, action_size, hidden_size, '_v')
        self._V_Smooth = SAC__ValueNet(picture_shape, extra_size, action_size, hidden_size, '_v_smooth')
        self._Policy = SAC__Policy(picture_shape, extra_size, action_size, hidden_size, '_p1')

    @staticmethod
    def prepare_state(state):
        if isinstance(state, tuple):
            return tuple([
                np.array([state[0]]),
                np.array([state[1]]),
            ])
        if isinstance(state, (list, np.ndarray, np.array)):
            return tuple([
                np.array([x[0] for x in state]),
                np.array([x[1] for x in state])
            ])
        raise ValueError("state type don't understud")

    def get_batch_actions(self, state, need_argmax=False, use_gumbel=True, temperature=0.5):
        # state: [batch_size, state_size]
        actions = self._Policy(
            SAC__Agent.prepare_state(state),
            use_gumbel=use_gumbel,
            temperature=temperature
        )
        if need_argmax:
            return np.argmax(actions, axis=1)
        return actions

    def get_single_action(self, state, need_argmax=False, use_gumbel=True, temperature=0.5):
        # state: [state_size, ]
        # return [action_szie, ]
        action = self._Policy(
            SAC__Agent.prepare_state(state),
            temperature=temperature,
            use_gumbel=use_gumbel
        )[0]
        if need_argmax:
            return np.argmax(action)
        return action

    def save(self, folder):
        import os
        if not os.path.exists(folder):
            os.makedirs(folder)
        self._Q1.save_weights(os.path.join(folder, 'Q1'))
        self._Q2.save_weights(os.path.join(folder, 'Q2'))
        self._Policy.save_weights(os.path.join(folder, 'Policy'))
        self._V.save_weights(os.path.join(folder, 'V'))

    def _get_grads(
            self,
            replay_batch,
            temperature=0.5,
            gamma=0.7,
    ):
        # shape of treplay_batch : tuple of (
        #     [batch_size, tuple(picture, extra_features)], - state
        #     [batch_size, actoin_size],- action
        #     [batch_size, 1],          - revard
        #     [batch_size, tuple(picture, extra_features)], - new state
        #     [batch_size, 1]           - is it done? (1 for done, 0 for not yet)
        # )
        state, action, reward, new_state, done_flag = replay_batch
        state = SAC__Agent.prepare_state(state)
        new_state = SAC__Agent.prepare_state(new_state)
        batch_size = len(done_flag)

        with tf.GradientTape() as tape:
            q_func_target = tf.cast(
                reward + gamma * (1 - done_flag) * self._V_Smooth(new_state),
                tf.float32
            )

            # two gradient update of Q-functions
            loss_q1 = tf.reduce_mean(
                (self._Q1(state, action) - tf.stop_gradient(q_func_target)) ** 2
            )
            loss_q2 = tf.reduce_mean(
                (self._Q2(state, action) - tf.stop_gradient(q_func_target)) ** 2
            )

            # PROBS, shape : [batch_size, action_size]
            new_actions_prob = self._Policy(state, use_gumbel=True, temperature=temperature)
            # max probs for each item in batch, [batch_size, 1]
            new_actions_max_log_probs = tf.reshape(
                tf.math.log(tf.reduce_max(new_actions_prob, axis=1)),
                (batch_size, 1),
            )
            # action for priviosly predicted probs, ONEHOT, [batch_size, action_size]
            new_actions = tf.reshape(
                tf.one_hot(
                    tf.argmax(new_actions_prob, axis=1),
                    self.action_size,
                ),
                (batch_size, self.action_size),
            )

            # shape: [batch_size, 1], get min of Q function in accordance with ariginal article
            new_q_func = tf.reduce_min([
                self._Q1(state, new_actions),
                self._Q2(state, new_actions),
            ],
                axis=0,
            )

            # update Value Net
            loss_v = tf.reduce_mean(
                0.5 * (self._V(state) - tf.stop_gradient(new_q_func - new_actions_max_log_probs)) ** 2
            )

            # update Policy
            loss_policy = tf.reduce_mean(
                (new_actions_max_log_probs - tf.stop_gradient(new_q_func)) ** 2
            )

            # compute gradients
            all_grads = tape.gradient(
                [loss_q1, loss_q2, loss_v, loss_policy],
                [
                    *self._Q1.trainable_variables,
                    *self._Q2.trainable_variables,
                    *self._V.trainable_variables,
                    *self._Policy.trainable_variables,
                ],
            )
            # clip gradients
            all_grads = [
                tf.clip_by_value(grad, -5.0, 5.0)
                for grad in all_grads
            ]

            # separate grads by models
            len_q1_tw = len(self._Q1.trainable_variables)
            len_q2_tw = len(self._Q2.trainable_variables)
            len_v_tw = len(self._V.trainable_variables)
            len_policy_tw = len(self._Policy.trainable_variables)

            grad_q1 = all_grads[:len_q1_tw]
            grad_q2 = all_grads[len_q1_tw: len_q1_tw + len_q2_tw]
            grad_v = all_grads[len_q1_tw + len_q2_tw: len_q1_tw + len_q2_tw + len_v_tw]
            grad_policy = all_grads[-len_policy_tw:]

        return grad_q1, grad_q2, grad_v, grad_policy

    def update_step(
            self,
            replay_batch,
            temperature=0.5,
            gamma=0.7,
            v_exp_smooth_factor=0.8,
    ):
        MINI_BATCH = 4
        # shape of treplay_batch : tuple of (
        #     [batch_size, tuple(picture, extra_features)], - state
        #     [batch_size, actoin_size],- action
        #     [batch_size, 1],          - revard
        #     [batch_size, tuple(picture, extra_features)], - new state
        #     [batch_size, 1]           - is it done? (1 for done, 0 for not yet)
        # )
        batch_size = len(replay_batch[-1])

        grad_q1_vector, grad_q2_vector, grad_v_vector, grad_policy_vector = [], [], [], []
        for mini_batch_start_index in range(0, batch_size, MINI_BATCH):
            grad_q1, grad_q2, grad_v, grad_policy = self._get_grads(
                replay_batch=(
                    x[mini_batch_start_index: mini_batch_start_index + MINI_BATCH]
                    for x in replay_batch
                ),
                temperature=temperature,
                gamma=gamma,
            )
            grad_q1_vector.append(grad_q1)
            grad_q2_vector.append(grad_q2)
            grad_v_vector.append(grad_v)
            grad_policy_vector.append(grad_policy)

        self._Policy.optimizer.apply_gradients(zip(
            tf.reduce_mean(grad_policy_vector, axis=0),
            self._Policy.trainable_variables,
        ))
        self._V.optimizer.apply_gradients(zip(
            tf.reduce_mean(grad_v_vector, axis=0),
            self._V.trainable_variables,
        ))
        self._Q2.optimizer.apply_gradients(zip(
            tf.reduce_mean(grad_q2, axis=0),
            self._Q2.trainable_variables,
        ))
        self._Q1.optimizer.apply_gradients(zip(
            tf.reduce_mean(grad_q1, axis=0),
            self._Q1.trainable_variables,
        ))

        # update Smooth Value Net
        for smooth_value, new_value in zip(self._V_Smooth.trainable_variables, self._V.trainable_variables):
            smooth_value.assign(
                smooth_value * v_exp_smooth_factor + (1 - v_exp_smooth_factor) * new_value
            )
