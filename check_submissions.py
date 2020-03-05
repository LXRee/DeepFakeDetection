import numpy as np
import pandas as pd
import os


def check(internet_path, local_path, dest):
    di = pd.read_csv(internet_path)
    dl = pd.read_csv(local_path)

    ddi = {}
    ddl = {}

    for i in range(di['label'].shape[0]):
        ddi[di['filename'].loc[i]] = di['label'].loc[i]

    for i in range(di['label'].shape[0]):
        ddl[dl['filename'].loc[i]] = dl['label'].loc[i]

    df = pd.DataFrame(columns=['filename', 'local_p', 'internet_p'])

    diff = 0
    for i, key in enumerate(ddl.keys()):
        df.loc[i] = [key, ddl[key], ddi[key]]
        diff += np.abs(ddl[key]) - np.abs(ddi[key]).sum()
    diff /= len(ddl.keys())
    print("The two embeddings differ for {}".format(diff))

    df.to_csv(dest)


def check_embeddings(internet_path, local_path, dest):
    di = pd.read_pickle(internet_path)
    dl = pd.read_pickle(local_path)

    ddi = {}
    ddl = {}

    for i in range(di['filename'].shape[0]):
        ddi[os.path.basename(di['filename'].loc[i])] = di['video_embedding'].loc[i]

    for i in range(di['filename'].shape[0]):
        ddl[os.path.basename(dl['filename'].loc[i])] = dl['video_embedding'].loc[i]

    # df = pd.DataFrame(columns=['filename', 'local_v', 'internet_v'])
    ud = {}
    for i, key in enumerate(ddl.keys()):
        ud[key] = [ddl[key], ddi[key]]

    diff = 0
    for key, value in ud.items():
        diff += (np.abs(value[0]) - np.abs(value[1])).mean()
    diff /= len(ud.keys())
    print("The difference between embeddings is, on average: {}".format(diff))
    # df.to_csv(dest)


if __name__ == '__main__':
    internet_path = '../submission_internet.csv'
    local_path = '../submission.csv'
    check(internet_path, local_path, '../confronto.csv')

