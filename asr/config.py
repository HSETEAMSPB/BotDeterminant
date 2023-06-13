# features extraction
sample_rate = 16000
window_len = 0.025
step_len = 0.010
num_feature_filters = 40

# path to directory containing librispeech data
librispeech_dir = "LibriSpeech"
# path to contain train / dev / test
data_key = "LibriSpeech"
# path to map file e.g {"a": 1, "b": ...}
vocab_file = "vocab.txt"

# datasetthree librispeech buildings for 960 hours in total
# you can also use smaller ones for example
links = ["https://www.openslr.org/resources/12/train-clean-100.tar.gz",
         "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
         "https://www.openslr.org/resources/12/train-other-500.tar.gz",
         ]
shuffle_size = 10000

# model size, in the original article alpha in [1, 2]
# best results with alpha = 2
alpha = 0.125

# warming up the learning rate so that the model doesn't 
# learn false dependencies from the very beginning
warmup_steps = 15000
init_learning_rate = 0.0025
peak_lr = 0.0025

# default values as in the article
num_units = 640
num_vocab = 32
num_lstms = 1
lstm_units = 2048
out_dim = 640
num_features = 40

# the best epoch is unknown :(
epochs = 100
# batch size (recommend 32-64)
bs = 32
