import os
import re
import json
import hiplot as hip


def is_wanted(root: str, filename: str) -> bool:
    """
    Filter out folders that we don't want to analyze
    :param path:
    :return:
    """
    return 'checkpoint' in filename and filename.endswith('pt') and ('old' not in root and '.ipy' not in root and 'with' not in root)


def find_best_checkpoint(experiments_path):
    """
    Looks at all the folders and checkpoints and returns the combination of hyper-parameters and loss in inverse order
    :param experiments_path:
    :return:
    """
    d = {}
    for root, dirs, files in os.walk(experiments_path):
        for file in files:
            if is_wanted(root, file):
                filename = file.split('.')[0] + '.' + file.split('.')[1]
                loss_value = filename.split('_')[1]
                loss_value = float(loss_value)
                # print("Loss: {}\tFolder: {}".format(loss_value, root))
                d[loss_value] = root
                # if loss_value < min_loss:
                #     min_loss = loss_value
                #     best_hyper = os.path.basename(root)
    l = list(d.keys())
    l.sort()
    for el in l:
        print("Loss: {}\tFolder: {}".format(el, d[el]))


if __name__ == '__main__':
    experiments_path = 'training/experiments/'
    find_best_checkpoint(experiments_path)
    # plot_hyperparameters(experiments_path)
