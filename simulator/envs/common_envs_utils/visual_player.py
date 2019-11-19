import argparse

from gym.envs.classic_control.rendering import SimpleImageViewer
from pyglet.window import key

# from gym_car_intersect.envs.hack_env__latest import CarRacingHackatonContinuous2

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
    parser.add_argument("--bots_number", type=int, default=0, help="Number of bot cars in environment.")
    args = parser.parse_args()

    env = CarRacingHackatonContinuous2()
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
            for _ in range(20):
                s, r, done, info = env.step(action)
                total_reward += r
            if steps % 50 == 0 or done:
                print("\naction " + str(action))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                print(info)
            steps += 1
            viewer.imshow(s[0])

            if done or restart:
                break


if __name__ == "__main__":
    main()
