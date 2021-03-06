from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import os

import gym
from chainerrl.wrappers import atari_wrappers

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import functools
from builtins import *  # NOQA

from future import standard_library

standard_library.install_aliases()  # NOQA
import argparse
import os

import chainer
import numpy as np

import chainerrl
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.q_functions import DistributionalDuelingDQN
from chainerrl import replay_buffer

from envs.common_envs_utils import *
from envs.gym_car_intersect import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CarIntersect-v3')
    parser.add_argument('--outdir', type=str, default='train/results', help='Directory path to save output files.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=None)
    parser.add_argument('--train', action='store_true', default=None)
    parser.add_argument('--eval-epsilon', type=float, default=0.0)
    parser.add_argument('--noisy-net-sigma', type=float, default=0.5)
    parser.add_argument('--steps', type=int, default=2 * 10 ** 6)
    parser.add_argument('--replay-start-size', type=int, default=2 * 10 ** 4)
    parser.add_argument('--eval-n-episodes', type=int, default=5)
    parser.add_argument('--eval-interval', type=int, default=10 ** 4)
    parser.add_argument('--logging-level', type=int, default=20, help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--monitor', action='store_true', default=False, help='Monitor env.')
    parser.add_argument('--num-envs', type=int, default=40)
    parser.add_argument('--final-epsilon', type=float, default=0.01)
    parser.add_argument('--final-exploration-frames', type=int, default=2 * 10 ** 4)
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs

    def make_car_env_discrete(max_frames=30 * 30, env_seed=42, random_suffix=None):
        print('CarIntersect-v3')
        env = gym.make('CarIntersect-v3')
        env = chainerrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=max_frames)
        env = MaxAndSkipEnv(env, skip=4)
        env = DiscreteWrapper(env)
        print('save_wrapper')
        env = SaveWrapper(env, random_suffix=random_suffix)
        env = WarpFrame(env)
        env.seed(env_seed)
        return env

    def make_batch_env(test):
        vec_env = chainerrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_car_env_discrete) for _, _ in enumerate(range(args.num_envs))])
        vec_env = chainerrl.wrappers.VectorFrameStack(vec_env, 4)
        # print(vec_env.observation_space)
        return vec_env

    env = make_batch_env(test=False)

    n_actions = env.action_space.n

    n_atoms = 51
    v_max = 10
    v_min = -10
    q_func = DistributionalDuelingDQN(n_actions, n_atoms, v_min, v_max, n_input_channels=12)

    # Noisy nets
    links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
    # Turn off explorer
    explorer = explorers.LinearDecayEpsilonGreedy(
        0.3, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    # Draw the computational graph and save it in the output directory.
    # chainerrl.misc.draw_computational_graph(
    #     [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
    #     os.path.join(args.outdir, 'model'))

    # Use the same hyper parameters as https://arxiv.org/abs/1707.06887
    opt = chainer.optimizers.Adam(0.00025, eps=1.5 * 10 ** -4)
    opt.setup(q_func)

    # Prioritized Replay
    # Anneal beta from beta0 to 1 throughout training
    update_interval = 4
    betasteps = args.steps / update_interval
    rbuf = replay_buffer.PrioritizedReplayBuffer(
        10 ** 5, alpha=0.5, beta0=0.4, betasteps=betasteps,
        num_steps=10)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    Agent = agents.CategoricalDoubleDQN
    print(args.replay_start_size)

    agent = Agent(
        q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
        explorer=explorer, minibatch_size=64,
        replay_start_size=args.replay_start_size,
        target_update_interval=3 * 10 ** 3,
        update_interval=update_interval,
        batch_accumulator='mean',
        phi=phi,
    )

    if args.load is True:
        print('evaluation started')
        dir_of_best_network = os.path.join("train/", "best")
        agent.load(dir_of_best_network)

        stats = experiments.evaluator.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=10,
            logger=None)
        print(stats)

    if args.train or not args.load:
        print('training started')
        experiments.train_agent_batch_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_episodes,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            log_interval=1000,
        )


if __name__ == '__main__':
    main()
