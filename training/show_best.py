import os


def find_best_checkpoint(experiments_path):
    """
    Looks at all the folders and checkpoints and returns the combination of hyper-parameters that performed better
    :param experiments_path:
    :return:
    """
    min_loss = 100
    best_hyper = ''
    for root, dirs, files in os.walk(experiments_path):
        for file in files:
            if 'checkpoint' in file:
                filename = file.split('.')[0] + '.' + file.split('.')[1]
                loss_value = filename.split('_')[1]
                loss_value = float(loss_value)
                if loss_value < min_loss:
                    min_loss = loss_value
                    best_hyper = os.path.basename(root)

    print("Best loss: {}".format(min_loss))
    print("Best hyper-parameters: {}".format(best_hyper))


if __name__ == '__main__':
    experiments_path = 'training/experiments'
    find_best_checkpoint(experiments_path)
