
import pandas as pd
import torch

def get_gt(path):
    df = pd.read_csv(path)

    all_label = []
    for i in range(50):
        lb = df[df['target']==i]['category'].iloc[0]
        all_label.append(lb)
    
    return all_label

def get_target(path):
    df = pd.read_csv(path)

    target = list(df['target'])

    return torch.tensor(target)


def get_audio_path(root, path):
    df = pd.read_csv(path)
    df['file'] = df['filename'].apply(lambda x: root+x)

    return list(df['file'])

    