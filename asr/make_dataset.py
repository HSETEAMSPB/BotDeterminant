from config import *
from features import mfcc

import os
import json
import glob
import subprocess
import tarfile
import soundfile as sf
import numpy as np
import tensorflow as tf

def make_vocabulary():
    """
    current example only works for regular english texts, 
    create your own if you want to teach for your language
    """

    characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
    characters.append("<unk>")
    with open(vocab_file, "w") as vocab:
        vocab.write("{")
        for num, char in enumerate(characters):
            if num == len(characters)-1:
                vocab.write(f"\"{char}\": {num}")
            else:
                vocab.write(f"\"{char}\": {num}, ")
        vocab.write("}")

def download_and_split():
    """
    Downloading LibriSpeech corps, splitting into test, train and dev directories
    """
    cwd = os.getcwd()

    def split_dir(num, path):
        name = path.name
        l_dir = os.listdir(f"LibriSpeech/{name}")
        size = len(l_dir)
        train = l_dir[:int(size*0.7)]
        dev = l_dir[int(size*0.7):int(size*0.85)]
        test = l_dir[int(size*0.85):]

        for i in train:
            subprocess.run(["mv", f"LibriSpeech/{name}/{i}", f"LibriSpeech/train/"])
        for i in dev:
            subprocess.run(["mv", f"LibriSpeech/{name}/{i}", f"LibriSpeech/dev/"])
        for i in test:
            subprocess.run(["mv", f"LibriSpeech/{name}/{i}", f"LibriSpeech/test/"])

    for num, link in enumerate(links):
        data_path = tf.keras.utils.get_file(
            fname=f"{cwd}/{num}.tar.gz",
            origin=link, 
            untar=True
        )
        untar = tarfile.open(f"{cwd}/{num}.tar.gz")
        untar.extractall('./')
        untar.close()

    rootdir = cwd + "/LibriSpeech"
    for num, item in enumerate(os.scandir(rootdir)):
        if item.is_dir():
            split_dir(num, item)


def dataset(data: str):
    """
    data: path to train, dev or test directory
    """
    v_file = open(vocab_file)
    v_file = json.load(v_file)

    def prepare(audio_file, text):
        audio, _ = sf.read(audio_file)
        feature = mfcc(audio)
        t_steps = np.array(feature.shape[0], dtype=np.int32)
        text = text.decode("utf-8")

        # mapping
        maps = [v_file[char] if char in v_file else v_file["<unk>"] for char in text]
        np_maps = np.array(maps, dtype=np.int32)
        len_maps = np.array(len(maps), dtype=np.int32)

        return feature, np_maps, t_steps, len_maps

    # Generate examples from a Librispeech directory
    audios, texts = [], []
    transcripts_glob = os.path.join(librispeech_dir, f"{data}*/*/*/*.txt")
    for f in glob.glob(transcripts_glob):
        path = os.path.dirname(f)
        for line in open(f).read().strip().splitlines():
            line = line.strip()
            number, transcription = line.split(" ", 1)
            audio_file = os.path.join(path, f"{number}.flac")
            audios += [audio_file]
            texts += [transcription]

    dataset = tf.data.Dataset.from_tensor_slices((audios, texts))
    dataset = dataset.shuffle(buffer_size=shuffle_size)
    dataset = dataset.map(
        lambda audio_file, transcript: tf.numpy_function(
            prepare,
            [audio_file, transcript],
            (tf.float32, tf.int32, tf.int32, tf.int32),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Filter utterances with <= 150 chars
    dataset = dataset.filter(lambda x, y, _, y_len: y_len <= 250)

    # Regroup the dataset by length and set the batch size for each group
    dataset = dataset.apply(
        tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda x, y, x_len, _: x_len,
            bucket_boundaries=[400, 1000, 1300, 1600, 2000],
            bucket_batch_sizes=[bs, bs//2, bs//2, bs//4, bs//4, bs//8],
            padded_shapes=([None, num_feature_filters], [None], [], []),
        )
    )

    return dataset
