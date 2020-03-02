import multiprocessing as mp
import os
import shlex
import subprocess

from data_preparation.utils.make_chunks import list_chunks

OS_TYPE = "linux"
ABSOLUTE_PATH = "/home/matteo/Projects/Github/DeepFakeDetection"
RELATIVE_INPUT_PATH = "dataset/deepfake-detection-challenge/train_sample_videos"
RELATIVE_OUTPUT_PATH = "dataset/deepfake-detection-challenge/train_sample_extracted_audio"
INPUT_PATH = os.path.join(ABSOLUTE_PATH, RELATIVE_INPUT_PATH)
OUTPUT_PATH = os.path.join(ABSOLUTE_PATH, RELATIVE_OUTPUT_PATH)


def extract_audio_from_chunk(file_paths: list):
    for file_path in file_paths:
        if os.path.splitext(os.path.basename(file_path))[1] == '.mp4':
            print("Processing {}...".format(os.path.splitext(os.path.basename(file_path))[0]))
            wav_dest = os.path.join(OUTPUT_PATH, os.path.basename(file_path).split('.')[0])
            main_cmd = "ffmpeg.exe" if OS_TYPE == "windows" else "ffmpeg"
            cmd = main_cmd + " -i {} -ab 160K -ac 1 -ar 44100 -vn {}.wav".format(file_path, wav_dest)
            p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _ = p.communicate()


def extract_audios_from_videos():
    paths = [os.path.join(INPUT_PATH, file_name) for file_name in os.listdir(INPUT_PATH)]
    processors = mp.cpu_count()
    pool = mp.Pool(processes=processors)
    paths_chunks = list_chunks(paths, len(paths) // processors)
    results = [pool.apply_async(extract_audio_from_chunk, args=[chunk]) for chunk in paths_chunks]
    _ = [p.get() for p in results]


if __name__ == '__main__':
    print("\nExtracting WAW audio files from videos...\n")
    extract_audios_from_videos()
    print("\nExtracted {na}/{nv} WAW audio files!".format(na=len(os.listdir(OUTPUT_PATH)),
                                                          nv=len(os.listdir(INPUT_PATH))))
