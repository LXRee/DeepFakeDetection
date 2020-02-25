import os, json
from tqdm import tqdm

# merge all metadata.json from splits

if __name__ == '__main__':
    # read all metadata.json from each folder
    metadata_list = {}
    splits_folder = os.path.join('data')
    for root, dirs, files in tqdm(os.walk(splits_folder), desc="Reading metadata"):
        metadata_path = ''
        for file in files:
            if 'meta' in file:
                metadata_path = os.path.join(root, file)
        metadata_list[metadata_path] = root

    # create new json with path to each image.
    all_meta = {}
    for metadata_path, root_dir in tqdm(metadata_list.items(), desc='Writing metadata'):
        if metadata_path is not '':
            metadata = json.load(open(metadata_path, 'r'))
            for key, value in metadata.items():
                all_meta[os.path.join(root_dir, key)] = value

    json.dump(all_meta, open(os.path.join(splits_folder, 'data.json'), 'w'))
