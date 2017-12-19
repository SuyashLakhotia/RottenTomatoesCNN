import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import heapq
import data


def cosine_similarity(embeddings, a, b):
    """
    embeddings: Embedding matrix.
    a, b: Indices to be compared.
    """
    return np.dot(embeddings[a], embeddings[b]) / (np.linalg.norm(embeddings[a]) * np.linalg.norm(embeddings[b]))


def calculate_similarities(embeddings, idx):
    """
    embeddings: Embedding matrix.
    idx: Index of word whose similarities are to be calculated.
    """
    similarities = []
    for i in range(len(embeddings)):
        sim = cosine_similarity(embeddings, idx, i)
        similarities.append(sim)
    return similarities


def print_most_similar(n, idx, embeddings, vocabulary):
    """
    n: Number of most similar words to print.
    idx: Index of word to be compared to.
    embeddings: Embedding matrix.
    vocabulary: (Index, Word) dictionary.
    """
    similarities = calculate_similarities(embeddings, idx)
    most_similar = map(similarities.index, heapq.nlargest(n, similarities))
    print([vocabulary[i] for i in most_similar])


def compare_similar_words(n, word, vocab, embeddings_arr):
    """
    n: Number of most similar words to compare.
    word: Word to be compared.
    vocab: The vocabulary dictionary used to extract the index of word.
    embeddings_arr: Array of word embeddings to be compared.
    """
    idx = vocab.get(word)
    for i in range(len(embeddings_arr)):
        print("%d:" % (i))
        print_most_similar(n, idx, embeddings_arr[i], vocabulary)


# Load data
print("Loading data...")
x_text, y = data.load_data_and_labels("data/rt-polarity.pos", "data/rt-polarity.neg")

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Extract vocabulary from vocab_processor
vocab_dict = vocab_processor.vocabulary_._mapping
sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
vocabulary = list(list(zip(*sorted_vocab))[0])  # list of words in vocabulary

# Restore model v2
sess = tf.Session()
saver = tf.train.import_meta_graph("runs/v2/1507798871/checkpoints/model-7100.meta")
saver.restore(sess, tf.train.latest_checkpoint("runs/v2/1507798871/checkpoints/."))

# Get embedding matrix
embeddings_v2 = sess.run("embedding/W:0")

# Restore model v2.1
sess = tf.Session()
saver = tf.train.import_meta_graph("runs/v2/1513692784/checkpoints/model-5000.meta")
saver.restore(sess, tf.train.latest_checkpoint("runs/v2/1513692784/checkpoints/."))

# Get embedding matrix
embeddings_v2_1 = sess.run("embedding/W:0")

# Embeddings array
embeddings_arr = []
embeddings_arr.append(embeddings_v2)
embeddings_arr.append(embeddings_v2_1)

# Compare words
compare_similar_words(10, "good", vocab_processor.vocabulary_, embeddings_arr)
compare_similar_words(10, "bad", vocab_processor.vocabulary_, embeddings_arr)
compare_similar_words(10, "clooney", vocab_processor.vocabulary_, embeddings_arr)
