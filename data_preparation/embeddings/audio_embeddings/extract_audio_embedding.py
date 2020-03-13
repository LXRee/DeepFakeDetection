from resemblyzer import VoiceEncoder, preprocess_wav
from utils.make_chunks import list_chunks
import pandas as pd
import multiprocessing as mp
import os
from tqdm import tqdm

# Use resemblyzer in order to extract the audio embeddings from the wav files


def extract_audio_embedding(paths, dest_path):
    """
    Extract audio embedding from each wav file in paths.
    :param paths: list of paths to wav audio files
    :param dest_path: output path in which save the df.pickle file
    :return:
    """
    # Define encoder for audio. This is a demanding task, but CUDA does not support multiprocessing from Python
    encoder = VoiceEncoder()
    for path in tqdm(paths):
        filename = os.path.basename(path).split('.')[0]
        # Extract wav features
        wav = preprocess_wav(path)
        # Actually creates audio embedding
        embed = encoder.embed_utterance(wav)
        df = pd.DataFrame(columns=['filename', 'audio_embedding'])
        # Put info inside dataframe. Keep the name as "filename.mp4" so we keep compatibility with video embeddings
        df.loc[0] = [filename + '.mp4', embed]
        df.to_pickle(os.path.join(dest_path, filename + '.csv'))


if __name__ == '__main__':
    input_folder = os.path.join('data', 'train_audio')
    output_folder = os.path.join('dataset', 'audio_embeddings')
    # Load paths for wav audio files
    paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder)]
    # Define workers for task
    processors = mp.cpu_count()
    processors = 4
    pool = mp.Pool(processes=processors)
    paths_chunks = list_chunks(paths, len(paths) // processors)
    results = [pool.apply_async(extract_audio_embedding, args=[chunk, output_folder]) for chunk in paths_chunks]
    # Process functions
    _ = [p.get() for p in results]
