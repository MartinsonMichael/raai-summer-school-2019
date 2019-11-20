import os
import argparse

import time
import gym
from gym.envs.classic_control.rendering import SimpleImageViewer
from pyglet.window import key

from envs import *

action = 0
restart = False
KEYS = {key.LEFT, key.RIGHT, key.UP, key.DOWN}
KEY_MAP = {
    key.LEFT: 3,
    key.RIGHT: 4,
    key.UP: 1,
    key.DOWN: 2,
}


def key_press(k, modifier):
    global restart, action
    if k == key.ESCAPE:
        restart = True
    if k in KEYS:
        action = KEY_MAP[k]


def key_release(k, modifier):
    global action
    if k in KEYS:
        action = 0


def main():
    global restart, action
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots-number", type=int, default=0, help="Number of bot cars in environment.")
    parser.add_argument("--env-name", type=str, default=None, help="Name of env to show.")
    parser.add_argument("--discrete", type=bool, default=False, help="Name of env to show.")
    parser.add_argument("--sleep", type=float, default=None, help="Name of env to show.")
    args = parser.parse_args()

    if args.env_name is None:
        print('Specify env name')
        return

    print(f'will be used \'{args.env_name}\'')

    env = gym.make(args.env_name)
    if args.discrete:
        print('use discrete wrapper')
        env = DiscreteWrapper(env)

    env.reset()

    viewer = SimpleImageViewer()
    viewer.imshow(env.render())
    viewer.window.on_key_press = key_press
    viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s = None
            done = None
            info = {}
            for _ in range(4):
                s, r, done, info = env.step(action)
                total_reward += r
            if (steps % 10 == 0 or done) and args.sleep is None:
                print("\naction " + str(action))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                print(info)

            steps += 1
            viewer.imshow(s)

            if args.sleep is not None:
                print("\naction " + str(action))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                print(info)
                time.sleep(args.sleep)

            if done or restart:
                print('restart')
                break


if __name__ == "__main__":
    main()
