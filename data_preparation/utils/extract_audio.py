import multiprocessing as mp
import os
import shlex
import subprocess
import json
from tqdm import tqdm

from data_preparation.utils.make_chunks import list_chunks


def extract_audio_from_chunk(file_paths: list, output_path):
    """
    Extract audio from files in file_paths list
    :param file_paths: list of paths/to/audio.mp4
    :return: None. Extract audio from file and writes them in output_path
    """
    for file_path in tqdm(file_paths, desc="Processing audios..."):
        # print("Processing {}...".format(os.path.splitext(os.path.basename(file_path))[0]))
        wav_dest = os.path.join(output_path, os.path.basename(file_path).split('.')[0])
        cmd = 'ffmpeg -i "{}" -ab 160K -ac 1 -ar 44100 -vn "{}".wav'.format(file_path, wav_dest)
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _ = p.communicate()


def extract_audios_from_videos(metadata_path, output_path):
    """
    Extract audio from video that are listed in metadata_path
    :param metadata_path: path/to/metadata.json containing training information
    :param output_path: path/to/dest/folder for the created wav files
    :return:
    """
    # Load metadata file
    metadata = json.load(open(metadata_path, 'r'))
    # Retrieve all the paths contained in metadata
    path_list = list(metadata.keys())
    processors = mp.cpu_count()
    pool = mp.Pool(processes=processors)
    paths_chunks = list_chunks(path_list, len(path_list) // processors)
    results = [pool.apply_async(extract_audio_from_chunk, args=[chunk, output_path]) for chunk in paths_chunks]
    _ = [p.get() for p in results]


if __name__ == '__main__':
    print("\nExtracting WAW audio files from videos...\n")
    metadata_path = os.path.join('data', 'train_data', 'balanced_metadata.json')
    dest_path = os.path.join('data', 'train_audio')
    extract_audios_from_videos(metadata_path, dest_path)
