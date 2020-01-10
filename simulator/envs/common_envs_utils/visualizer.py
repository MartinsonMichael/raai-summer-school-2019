import os
from matplotlib import animation
from IPython.display import display, HTML
import datetime
import matplotlib.pyplot as plt


def visualize(env, action_picker, name='test', folder='save_animation_folder'):
    folder_full_path = os.path.join(folder, name)
    if not os.path.exists(folder_full_path):
        os.makedirs(folder_full_path)

    state = env.reset()
    im_array = [state]
    total_reward = 0.0
    while True:
        new_state, reward, done, info = action_picker(state)
        im_array.append(new_state)
        state = new_state
        total_reward += reward
        if done:
            break
    plot_sequence_images(im_array, need_disaply=False, need_save=os.path.join(
        folder_full_path, f'R_{total_reward}__Time_{datetime.datetime.now()}_.mp4'
    ))


# f'./save_animation_folder/{datetime.datetime.now()}.mp4'

def plot_sequence_images(image_array, need_disaply=False, need_save=None):
    ''' Display images sequence as an animation in jupyter notebook

    Args:
        image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
    '''
    dpi = 72.0
    xpixels, ypixels = image_array[0].shape[:2]
    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(image_array),
        interval=33,
        repeat_delay=1,
        repeat=True
    )
    if need_save is not None:
        anim.save(need_save)

    if need_disaply:
        display(HTML(anim.to_html5_video()))
