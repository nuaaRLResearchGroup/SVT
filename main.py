import math
import random
import time
import numpy as np

from plot import *
from SLDQN_module import DQN
import scipy.io as sio


################################################################################
def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm) * 1.0
            d = np.convolve(y, d, "same") / np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data



def gain_all(rsu1, rsu2, rsu3, rsu4, uav, power, noise_power, B):
    d1 = math.sqrt((uav[0] - rsu1[0]) ** 2 + (uav[1] - rsu1[1]) ** 2 + (uav[2] - rsu1[2]) ** 2)
    d2 = math.sqrt((uav[0] - rsu2[0]) ** 2 + (uav[1] - rsu2[1]) ** 2 + (uav[2] - rsu2[2]) ** 2)
    d3 = math.sqrt((uav[0] - rsu3[0]) ** 2 + (uav[1] - rsu3[1]) ** 2 + (uav[2] - rsu3[2]) ** 2)
    d4 = math.sqrt((uav[0] - rsu4[0]) ** 2 + (uav[1] - rsu4[1]) ** 2 + (uav[2] - rsu4[2]) ** 2)

    ch_j1 = channel_gain(d1)
    ch_j2 = channel_gain(d2)
    ch_j3 = channel_gain(d3)
    ch_j4 = channel_gain(d4)

    return (ch_j1, ch_j2, ch_j3, ch_j4)




def main():
    algos = ['DQN']
    y_labels = ['reward', 'latency (s)', 'energy consumption (J)', 'BER', 'Eavesdropping (bps)',
                'risk level', 'SINR', 'power (W)']
    results = {ylabel: [] for ylabel in y_labels}
    summary = {ylabel: {algo: 0 for algo in algos} for ylabel in y_labels}

    max_episode = 10
    max_time = 50  # 800
    memory_capacity = max_time
    window = 3
    gamma = 0.6
    batch_size = 16
    learning_begin = 32
    beta = 0.3

    for algo in algos:



        rewards = []
        times = []
        consumptions = []
        bers = []
        qtls = []
        reputations = []
        risk_level = []
        # eave_level = []
        sinrs = []
        chosen_power = []
        # ================= Save =================
        Save_utility = np.zeros((max_episode, max_time))
        Save_qtl = np.zeros((max_episode, max_time))
        Save_ber = np.zeros((max_episode, max_time))
        Save_time = np.zeros((max_episode, max_time))
        Save_consumption = np.zeros((max_episode, max_time))
        Save_power = np.zeros((max_episode, max_time))
        Save_risk_level = np.zeros((max_episode, max_time))

        # ================= Learning =================
        agent = DQN(12, len(action), memory_capacity, window=window, GAMMA=gamma, learning_begin=learning_begin,
                    beta=beta, safe_mode=False)
        for episode in range(max_episode):
            b = gain_all(rsu1, rsu2, rsu3, rsu4, uav, u_power, noise_power, B)  # b: 为无人机与四个通信节点间的最大数据传输速率

            rewardd = []
            timee = []
            consumptionn = []
            berr = []
            qtll = []
            risk_levell = []
            sinrr = []
            powerr = []


            for time_step in range(max_time):
                state = np.array([b, t, l], dtype=np.float32)

                # ================= choose_action =================
                act_idx = agent.select_action(state, time_step)

                act1 = action[act_idx][0]
                node1 = compute_node[act1]
                act2 = action[act_idx][1]
                node2 = compute_node[act2]

                power_node1 = action[act_idx][2] * 0.3
                power_node2 = u_power - power_node1

                d_node1 = math.sqrt((uav[0] - node1[0]) ** 2 + (uav[1] - node1[1]) ** 2 + (uav[2] - node1[2]) ** 2)
                d_node2 = math.sqrt((uav[0] - node2[0]) ** 2 + (uav[1] - node2[1]) ** 2 + (uav[2] - node2[2]) ** 2)

                ch_node1 = channel_gain(d_node1)  # 计算选出的两个节点的信道增益
                ch_node2 = channel_gain(d_node2)

                d_qt = math.sqrt((uav[0] - qt[0]) ** 2 + (uav[1] - qt[1]) ** 2 + (uav[2] - qt[2]) ** 2)
                ch_qt = channel_gain(d_qt)  # * 0.3  # 窃听信道的信道增益

                dynamic_jam_power_node1 = jam_power * (power_node1 * 10) ** 2
                dynamic_jam_power_node2 = jam_power * (power_node2 * 10) ** 2
                sinr_node1 = SINR(d_node1, qt, node1, power_node1, dynamic_jam_power_node1, noise_power)
                sinr_node2 = SINR(d_node2, qt, node2, power_node2, dynamic_jam_power_node2, noise_power)

                ber_node1 = 1 / 2 * math.erfc(math.sqrt(sinr_node1 / 2))
                ber_node2 = 1 / 2 * math.erfc(math.sqrt(sinr_node2 / 2))
                ber = (ber_node1 + ber_node2) / 2

                datarate1 = B * math.log((1 + sinr_node1), 2)
                datarate2 = B * math.log((1 + sinr_node2), 2)

                time1 = o[0] * 8192 / datarate1
                time2 = o[1] * 8192 / datarate2
                next_t = [0, 0, 0, 0]
                next_t[act1] = time1
                next_t[act2] = time2
                next_t = tuple(next_t)
                time = time1 + time2
                latency = time

                consumption1 = power_node1 * time1
                consumption2 = power_node2 * time2
                next_l = [0, 0, 0, 0]
                next_l[act1] = consumption1
                next_l[act2] = consumption2
                next_l = tuple(next_l)
                consumption = consumption1 + consumption2

                qtl_node1 = B * math.log((1 + power_node1 * ch_qt / noise_power), 2)
                qtl_node2 = B * math.log((1 + power_node2 * ch_qt / noise_power), 2)
                qtl = (qtl_node1 + qtl_node2) / 2

                if time_step < 420:
                    uav[0] += (random.choice(speed))
                    qt[0] -= (random.choice(speed))

                rewardd.append(reward)
                timee.append(latency)
                consumptionn.append(consumption)
                berr.append(ber)
                qtll.append(qtl)
                # risk_levell.append(level)
                sinrr.append((sinr_node1 + sinr_node2) / 2)
                powerr.append(power_node1)

                # =======更新==========
                b = gain_all(rsu1, rsu2, rsu3, rsu4, uav, u_power, noise_power, B)
                next_b = b
                o1 = random.choice(o)
                next_o = (o1, o1, o1, o1)
                next_r = (qtl, qtl, qtl, qtl)

                # state_next = np.array([next_b, next_resource, next_v, next_o, next_c, next_r, next_t, next_l])
                state_next = np.array([next_b, next_t, next_l])

                # ================= Sauav =================
                Save_utility[episode, time_step] = reward
                Save_qtl[episode, time_step] = qtl
                Save_ber[episode, time_step] = ber
                Save_time[episode, time_step] = latency
                Save_consumption[episode, time_step] = consumption
                Save_power[episode, time_step] = power_node1
                # Save_risk_level[episode, time_step] = level

                # ================= Update & iterate =================

                agent.optimize_model(state, state_next, act_idx, reward, gamma_start=0.7, gamma_end=0.3,
                                     anneal_step=max_time, step=time_step, learning_begin=learning_begin,
                                     BATCH_SIZE=batch_size)
                b = next_b
                o = next_o
                r = next_r
                t = next_t
                l = next_l

            agent.reset()

            rewards.append(rewardd)
            times.append(timee)
            consumptions.append(consumptionn)
            bers.append(berr)
            qtls.append(qtll)
            # risk_level.append(risk_levell)

            sinrs.append(sinrr)
            chosen_power.append(powerr)
            for result, y_label in zip(
                    [rewardd, timee, consumptionn, berr, qtll, sinrr, powerr],
                    y_labels):
                summary[y_label][algo] += result[-1]

            # average for one algo
        for result, y_label in zip(
                [rewards, times, consumptions, bers, qtls, sinrs, chosen_power],
                y_labels):
            results[y_label].append(np.mean(np.array(result), axis=0))
            summary[y_label][algo] = np.mean(np.array(result), axis=0)[-1]

        filename = f'./{algo}.mat'

        sio.savemat(filename,
                    {f'{algo}_Utility': Save_utility,
                     f'{algo}_qtl': Save_qtl,
                     f'{algo}_BER': Save_ber,
                     f'{algo}_time': Save_time,
                     f'{algo}_consumption': Save_consumption, })

        # ================= Save & plot =================
        # now = datetime.datetime.now()
        # step = np.arange(1, max_time + 1)
        # filefoldname = 'SLQ' + now.strftime('%Y_%m_%d_%H_%M_%S')
        # # os.mkdir(filefoldname)
        # fname1 = filefoldname + '/reward'

    for i, y_label in enumerate(results.keys()):
        plt_alg_contrast(smooth(results[y_label], 10), y_label, algos, save_path='./data/debug/', index=i)

    plt.close('all')


if __name__ == '__main__':
    main()
