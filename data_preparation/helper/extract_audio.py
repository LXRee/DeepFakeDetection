import multiprocessing as mp
import os
import subprocess

from data_preparation.helper.make_chunks import list_chunks


def extract_audio_from_paths(in_paths, dest_path):
    for in_path in in_paths:
        if os.path.splitext(os.path.basename(in_path))[1] == '.mp4':
            wav_dest = os.path.join(dest_path, os.path.basename(in_path).split('.')[0])
            command = "ffmpeg.exe -i {} -ab 160K -ac 1 -ar 44100 -vn {}.wav".format(in_path, wav_dest)
            # in order to suppress output and to not overload processor with it
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _ = p.communicate()


def extract_audio_from_source(input_path, dest_path):
    paths = []
    for root, dirs, files in os.walk(input_path):
        # Retrieve all paths in input_path folder
        paths.extend([os.path.join(root, file) for file in files])

    # Needed for multiprocessing
    paths = list(paths)
    processors = 6
    pool = mp.Pool(processes=processors)
    arguments = list_chunks(paths, len(paths) // processors)
    results = [pool.apply_async(extract_audio_from_paths, args=(a, dest_path)) for a in arguments]
    _ = [p.get() for p in results]
    # extract_audio_from_paths(paths, dest_path)


if __name__ == '__main__':
    in_path = 'C:\\Users\\mawanda\\PyCharmProjects\\DeepFakeCompetition\\test_data'
    out_path = 'C:\\Users\\mawanda\\PyCharmProjects\\DeepFakeCompetition\\test_audio'
    extract_audio_from_source(in_path, out_path)
