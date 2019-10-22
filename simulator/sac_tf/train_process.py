# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
import tensorflow as tf
from os.path import join
import pickle

from .holder import Holder

tf.reset_default_graph()
session = tf.Session()
holder = Holder(session)

holder.insert_N_sample_to_replay_memory(20 * 1000, temperature=0.01)

for i in range(100 * 1000):
    gamma = min(0.9, 0.1 + i**0.5 / 1000)
    temperature = min(0.7, 0.2 + i ** 0.5 / 1000)

    print(f'step: {i}, gamma: {gamma}, temperature: {temperature}')

    holder.insert_N_sample_to_replay_memory(1000, temperature=temperature)
    holder.update_agent(update_step_num=500, temperature=temperature, gamma=gamma)
    holder.agent.update_V_ExpSmooth(0.8)

    holder.get_test_game_total_revard()

    if i % 500 == 499:
        folder = join('saved_models', f'step_{i}')
        holder.agent.save(folder)
        pickle.dump(
            holder.history,
            open(join(folder, f'history.pkl', 'wb'))
        )

    # clear_output(wait=True)
    # ax.plot(holder.get_history()[:, 0], holder.get_history()[:, 1])
    # display(fig)
    #
    # plt.pause(0.5)
