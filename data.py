import re
import itertools
import collections

import numpy as np


def clean_str(string):
    """
    Tokenization & string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates the labels. Returns the 
    split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding="utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding="utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Convert to numpy array
    x_text = np.array(x_text)
    # Generate one-hot encoded label arrays
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_word2vec(filepath, vocabulary, embedding_dim, tf_VP=False):
    """
    Returns the embedding matrix for vocabulary from filepath.
    """

    if tf_VP:
        null_index = 0
    else:
        null_index = None

    # Initialize embedding matrix from pre-trained word2vec embeddings. 0.25 is chosen so that unknown
    # vectors have (approximately) the same variance as pre-trained ones.
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocabulary), embedding_dim))

    words_found = 0
    with open(filepath, "rb") as f:
        header = f.readline()
        word2vec_vocab_size, embedding_size = map(int, header.split())
        binary_len = np.dtype("float32").itemsize * embedding_size
        for line in range(word2vec_vocab_size):
            word = []
            while True:
                ch = f.read(1).decode("latin-1")
                if ch == " ":
                    word = "".join(word)
                    break
                if ch != "\n":
                    word.append(ch)

            if tf_VP:
                idx = vocabulary.get(word)
            else:
                idx = vocabulary.get(word, None)

            if idx != null_index:
                embeddings[idx] = np.fromstring(f.read(binary_len), dtype="float32")
                words_found += 1
            else:
                f.read(binary_len)

    print("Word Embeddings Extracted: {}".format(words_found))
    print("Word Embeddings Randomly Initialized: {}".format(len(vocabulary) - words_found))

    return embeddings


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for the dataset.
    """
    data = np.array(data)
    data_size = len(data)
    indices = collections.deque()
    num_iterations = int(num_epochs * data_size / batch_size)
    for step in range(1, num_iterations + 1):
        if len(indices) < batch_size:
            if shuffle:
                indices.extend(np.random.permutation(data_size))
            else:
                indices.extend(np.arange(data_size))
        idx = [indices.popleft() for i in range(batch_size)]
        yield data[idx]
