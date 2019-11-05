from abc import ABC

import numpy as np
import tensorflow as tf
from chainerrl.agent import BatchAgent
from tensorflow.contrib.distributions.python.ops import relaxed_onehot_categorical
from typing import Sequence, Any


# base class for all SAC parts
class SAC__Base:

    def __init__(self):
        self.session = None
        self.scope_name = None

    def copy_weights_from_model(self, other_model, transform_func=None):
        '''
        Copy to current model weights from other_model.

        transform_func allow you to assing to the model transformed weights.
        It should be (new_weight, old_weight) -> weight_to_assing
        '''
        if transform_func is None:
            transform_func = lambda w_new, w_old: w_new
        update_weights = [
            tf.assign(old, transform_func(new, old))
            for (old, new)
            in zip(
                tf.trainable_variables(self.scope_name),
                tf.trainable_variables(other_model.scope_name)
            )
        ]
        self.session.run(update_weights)

    def get_name(self):
        return self.scope_name

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_name)

    def load_from_file(self, path, file_name):
        saver = tf.train.import_meta_graph(file_name + '.meta')
        saver.restore(
            self.session,
            tf.train.latest_checkpoint(path),
        )

    def save_to_file(self, file_name):
        saver = tf.train.Saver(self.get_variables())
        saver.save(self.session, file_name)


class SAC__ValueNet(SAC__Base):
    '''
    Implementaion of V function
    '''

    def __init__(self, session, state_size, action_size, hidden_size=128, name='_v1'):
        super().__init__()
        self.session = session
        self.scope_name = 'SAC__ValueNet' + name

        with tf.variable_scope(self.scope_name):
            # V-value architecture from just two FC layer with relu activation
            self.state = tf.placeholder(
                dtype=tf.float32,
                shape=[None, state_size],
                name='ValueNet_state'
            )
            # (None, state_size) -> (None, hidden_size)
            x = tf.layers.Dense(units=hidden_size, activation='relu')(self.state)
            # (hidden_size, state_size) -> (None, hidden_size)
            x = tf.layers.Dense(units=hidden_size, activation='relu')(x)
            # (hidden_size, state_size) -> (None, 1)
            self.value = tf.layers.Dense(units=1, activation=None)(x)

            # place holder.py for `Q(s, a) - log(pi(a|s))`
            self.target = tf.placeholder(
                dtype=tf.float32,
                shape=[None, 1],
                name='ValueNet_target'
            )
            # use stop gradient to prevent the gradient spreading to target, that is `Q(s, a) - log(pi(a|s))`
            loss = 0.5 * tf.reduce_mean((self.value - tf.stop_gradient(self.target)) ** 2)
            self.optimizer = tf.train.AdamOptimizer(0.003)

            # minimizing step over trainable params in this scope
            self.train_step = self.optimizer.minimize(
                loss,
                var_list=self.get_variables(),
            )

    def get_value(self, state):
        return self.session.run(
            self.value,
            feed_dict={
                self.state: state,
            },
        )

    def make_update_step(self, state, target):
        self.session.run(
            self.train_step,
            feed_dict={
                self.state: state,
                self.target: target,
            }
        )


class SAC__QNet(SAC__Base):
    '''
    Implementation of Q function.
    '''

    def __init__(self, session, state_size, action_size, hidden_size=128, name='_v1'):
        super().__init__()
        self.session = session
        self.scope_name = 'SAC__QNet' + name

        with tf.variable_scope(self.scope_name):
            # net architecture
            self.state = tf.placeholder(
                dtype=tf.float32,
                shape=[None, state_size],
                name='QNet_state'
            )
            self.action = tf.placeholder(
                dtype=tf.float32,
                shape=[None, action_size],
                name='QNet_action'
            )

            x_state = tf.layers.Dense(units=hidden_size, activation='relu')(self.state)
            x_action = tf.layers.Dense(units=hidden_size, activation='relu')(self.action)

            x = tf.concat([x_state, x_action], axis=1)
            x = tf.layers.Dense(units=hidden_size, activation='relu')(x)
            x = tf.layers.Dense(units=hidden_size, activation='relu')(x)
            # final shape [None, 1]
            self.qvalue = tf.layers.Dense(units=1)(x)

            self.optimizer = tf.train.AdamOptimizer(0.003)

            # here one should set target as `r(s_t, a_t) + gamma * V(s_{t+1})`
            self.target = tf.placeholder(
                dtype=tf.float32,
                shape=[None, 1],
                name='QNet_target'
            )
            loss = tf.reduce_mean((self.qvalue - tf.stop_gradient(self.target)) ** 2)
            self.train_step = self.optimizer.minimize(
                loss,
                var_list=self.get_variables(),
            )

    def get_q(self, state, action):
        return self.session.run(
            self.qvalue,
            feed_dict={
                self.state: state,
                self.action: action,
            },
        )

    def make_update_step(self, state, action, target):
        self.session.run(
            self.train_step,
            feed_dict={
                self.state: state,
                self.action: action,
                self.target: target,
            },
        )


class SAC__Policy(SAC__Base):
    '''
    Implementation of Policy function.
    '''

    def __init__(self, session, state_size, action_size, hidden_size=128, name='_v1'):
        super().__init__()
        self.session = session
        self.scope_name = 'SAC__Policy' + name

        with tf.variable_scope(self.scope_name):
            # net architecture
            self.state = tf.placeholder(
                dtype=tf.float32,
                shape=[None, state_size],
                name='Policy_state'
            )
            self.action = tf.placeholder(
                dtype=tf.float32,
                shape=[None, action_size],
                name='Policy_action'
            )

            x = tf.layers.Dense(units=hidden_size, activation='relu')(self.state)
            x = tf.layers.Dense(units=hidden_size, activation='relu')(x)
            # final shape [None, 1]
            x = tf.layers.Dense(units=action_size)(x)
            self.policy_probs = tf.math.softmax(x, axis=1)

            # temperature to control Gumbel-Softmax
            self.temperature = tf.placeholder(
                dtype=tf.float32,
                name='Policy_temperature'
            )
            self.dist = relaxed_onehot_categorical.RelaxedOneHotCategorical(
                temperature=self.temperature,
                probs=self.policy_probs
            )
            # shape [None, action_size]
            self.generated_actions_probs = self.dist.sample()
            # shape [None, 1]
            self.generated_action = tf.one_hot(
                tf.math.argmax(self.generated_actions_probs, axis=1),
                action_size,
            )

            self.optimizer = tf.train.AdamOptimizer(0.003)

            # target should be Q(state, all_actions),
            #     to match bu shape generated_actions_probs: [None, action_size]
            self.target = tf.placeholder(
                dtype=tf.float32,
                shape=[None, 1],
                name='Policy_target'
            )
            loss = tf.reduce_mean(tf.log(self.generated_actions_probs) - tf.stop_gradient(self.target))
            self.train_step = self.optimizer.minimize(
                loss,
                var_list=self.get_variables(),
            )

    def make_update_step(self, state, target, temperature=0.5):
        self.session.run(
            self.train_step,
            feed_dict={
                self.state: state,
                self.target: target,
                self.temperature: temperature,
            }
        )

    def get_policy_probs(self, state, temperature=0.5):
        return self.session.run(
            self.generated_actions_probs,
            feed_dict={
                self.state: state,
                self.temperature: temperature,
            }
        )

    def get_policy_action(self, state, temperature=0.5):
        return self.session.run(
            self.generated_action,
            feed_dict={
                self.state: state,
                self.temperature: temperature,
            }
        )

    def get_policy_actions_with_probs(self, state, temperature=0.5):
        return self.session.run([
            self.generated_action,
            self.generated_actions_probs,
        ],
            feed_dict={
                self.state: state,
                self.temperature: temperature,
            }
        )


class SAC__Agent(BatchAgent):
    '''
    Class of agent, whitch controll update steps of all sub-net and can be used in expluatation.
    '''

    def batch_act(self, batch_obs: Sequence[Any]):
        return self.get_batch_actions(batch_obs, need_argmax=True)

    def batch_act_and_train(self, batch_obs):
        pass

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        pass

    def batch_observe_and_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        pass

    def __init__(self,
                 session,
                 state_size,
                 action_size,
                 hidden_size=128,
                 name='agent_1',
                 info=''):
        # save meta info
        self.name = name
        self.info = info

        # save env hyper params
        self.action_size = action_size
        self.state_size = state_size

        # here init agent nets
        self._Q1 = SAC__QNet(session, state_size, action_size, hidden_size, '_q1')
        self._Q2 = SAC__QNet(session, state_size, action_size, hidden_size, '_q2')
        self._V = SAC__ValueNet(session, state_size, action_size, hidden_size, '_v1')
        self._V_ExpSmooth = SAC__ValueNet(session, state_size, action_size, hidden_size, '_VSmooth')
        self._Policy = SAC__Policy(session, state_size, action_size, hidden_size, '_p1')

        # init weights
        session.run(tf.initialize_all_variables())

        # make V_exp_smooth net equal to V net
        self._V_ExpSmooth.copy_weights_from_model(self._V)

    def update_step(
            self,
            replay_batch,
            temperature=0.5,
            gamma=0.7,
            v_exp_smooth_factor=0.8,
            need_update_VSmooth=False,
    ):
        # shape of replay_batch : tuple of (
        #     [batch_size, state_size], - state
        #     [batch_size, actoin_size],- action
        #     [batch_size, 1],          - revard
        #     [batch_size, state_size], - new state
        #     [batch_size, 1]           - is it done? (1 for done, 0 for not yet)
        # )
        state, action, reward, new_state, done_flag = replay_batch
        batch_size = len(state)

        # cur_policy_actions is [batch_size, action_szie]
        cur_policy_actions, cur_policy_probs = self._Policy.get_policy_actions_with_probs(
            state=state,
            temperature=temperature,
        )
        # compute log prods for the most probabel action
        #     and reshape it to [batch_size, 1]
        cur_actions_log_probs = np.reshape(
            np.log(np.mean(cur_policy_probs, axis=1)),
            (batch_size, 1),
        )

        # shape: [batch_size, 1], get min of Q function in accordance with ariginal article
        q_func_current = np.min(
            np.array([
                self._Q1.get_q(state, cur_policy_actions),
                self._Q2.get_q(state, cur_policy_actions),
            ]),
            axis=0
        )

        # update Value function
        self._V.make_update_step(
            state=state,
            target=q_func_current - cur_actions_log_probs
        )

        #  update both Q functions
        q_func_target = reward + gamma * (1 - done_flag) * self._V_ExpSmooth.get_value(new_state)
        self._Q1.make_update_step(
            state=state,
            action=action,
            target=q_func_target,
        )
        self._Q2.make_update_step(
            state=state,
            action=action,
            target=q_func_target,
        )

        # update Policy function
        self._Policy.make_update_step(
            state=state,
            target=q_func_current,
        )

        # we donn't need update V_exp_smmoth on each step
        if need_update_VSmooth:
            self.update_V_ExpSmooth(v_exp_smooth_factor)

    def update_V_ExpSmooth(self, v_exp_smooth_factor):
        # update V_exp_smooth
        self._V_ExpSmooth.copy_weights_from_model(
            self._V,
            transform_func=
            lambda w_new, w_old: w_new * v_exp_smooth_factor + (1 - v_exp_smooth_factor) * w_old,
        )

    def save(self, folder):
        import os
        if not os.path.exists(folder):
            os.makedirs(folder)
        for model in [self._Q1, self._Q2, self._V, self._V_ExpSmooth, self._Policy]:
            model.save_to_file(os.path.join(folder, model.get_name()))

    def get_batch_actions(self, state, need_argmax=False, temperature=0.5):
        # state: [batch_size, state_size]
        actions = self._Policy.get_policy_action(
            state=state,
            temperature=temperature,
        )
        if need_argmax:
            return np.argmax(actions, axis=1)
        return actions

    def get_single_action(self, state, need_argmax=False, temperature=0.5):
        # state: [state_size, ]
        # return [action_szie, ]
        action = self._Policy.get_policy_action(
            state=[state],
            temperature=temperature,
        )[0]
        if need_argmax:
            return np.argmax(action)
        return action
