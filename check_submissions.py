import numpy as np
import pandas as pd

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

    for i, key in enumerate(ddl.keys()):
        df.loc[i] = [key, ddl[key], ddi[key]]

    df.to_csv(dest)
