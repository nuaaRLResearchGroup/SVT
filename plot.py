import os

import matplotlib.pyplot as plt

INDEX = 0

fontsize = 12


def plt_alg_contrast(datas, y_label, algs, save_path=None, index=None):
    plt.style.use('ggplot')
    plt.set_cmap("Accent")
    plt.rcParams['font.family'] = 'Times New Roman'

    global INDEX
    index = index if index is not None else INDEX
    INDEX += 1

    fig = plt.figure(num=index)
    for data, alg in zip(datas, algs):
        plt.plot(data, label=alg)
    plt.xlabel('Time slots', fontsize=fontsize)
    plt.title(y_label, fontsize=fontsize)
    plt.legend(fontsize=fontsize-3)
    plt.tight_layout()

    if save_path is not None:
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        fig_name = f'/{y_label}' + '.png'
        plt.savefig(save_path + fig_name, format='png', dpi=300)

    plt.close()


def plt_alg_contrast_one_fig(datas, y_labels, algs, save_path=None, index=None):
    plt.style.use('ggplot')
    plt.set_cmap("Accent")
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    axes = axes.reshape(-1)
    for i, (key, data) in enumerate(zip(y_labels, datas)):
        for j, (d, alg) in enumerate(zip(data, algs)):
            axes[i].plot(d, label=alg)

        if i == 4:
            axes[i].set_xlabel('Time slots', fontsize=fontsize)
            axes[i].legend(fontsize=12)
        # axes[i].set_ylabel(key)
        axes[i].set_title(key, fontsize=fontsize)

    plt.tight_layout()

    if save_path is not None:
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        fig_name = f'/result' + '.png'
        plt.savefig(save_path + fig_name, format='png', dpi=300)

    plt.close()
