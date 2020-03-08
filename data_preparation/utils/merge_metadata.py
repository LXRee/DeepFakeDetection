import json
import os
import random

from tqdm import tqdm

""" Merge all metadata.json from dataset splits """


def merge_metadata(source_path, dest_path):
    """

    :param source_path: path/to/folder containing dataset splits and metadata(s)
    :param dest_path: path/to/folder containing destination metadata(s)
    :return: nothing. Writes balanced metadata and one with skipped videos
    """
    # Read all metadata.json from each folder
    metadata_list = {}
    splits_folder = os.path.join(source_path)
    for root, dirs, files in tqdm(os.walk(splits_folder), desc="Reading metadata"):
        for file in files:
            if 'meta' in file:
                metadata_path = os.path.join(root, file)
                metadata_list[metadata_path] = root

    # Create new json with path to each image.
    all_meta = {}
    for metadata_path, root_dir in tqdm(metadata_list.items(), desc='Writing metadata'):
        if metadata_path is not '':
            metadata = json.load(open(metadata_path, 'r'))
            for key, value in metadata.items():
                p = os.path.join(root_dir, key)
                if os.path.exists(p):
                    all_meta[p] = value
                else:
                    print('{} not found!'.format(p))

    # Select REAL videos and their information and check if exists
    real_metadata = {k: v for k, v in all_meta.items() if v['label'] == 'REAL'}
    # Select FAKE videos and their information and check if exists. Plus, extract keys and shuffle them.
    fake_metadata = {k: v for k, v in all_meta.items() if v['label'] == 'FAKE'}
    fake_meta_keys = list(fake_metadata.keys())
    random.shuffle(fake_meta_keys)

    # Create new dict with balanced metadata
    balanced_meta = {k: v for k, v in real_metadata.items()}
    skipped_meta = {}
    for i, k in enumerate(fake_meta_keys):
        if i < len(real_metadata.keys()):
            balanced_meta[k] = fake_metadata[k]
        else:
            skipped_meta[k] = fake_metadata[k]

    # Sanity check
    assert len([k for k, v in balanced_meta.items() if v['label'] == 'FAKE']), len([k for k, v in balanced_meta.items() if v['label'] == 'REAL'])

    # Dump balanced dataset
    json.dump(balanced_meta, open(os.path.join(dest_path, 'balanced_metadata.json'), 'w'))

    # Dump skipped items
    json.dump(skipped_meta, open(os.path.join(dest_path, 'skipped_items.json'), 'w'))


if __name__ == '__main__':
    source_path = 'D:\\deepfakedataset'
    dest_path = 'data\\train_data'
    merge_metadata(source_path, dest_path)
