import os
import time
import datetime
import data

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import scipy.sparse
import sklearn.datasets
import sklearn.metrics

from text_gcnn import GraphCNN
from gcnn import graph, coarsening


# Parameters
# ==================================================

# Data params
test_sample_percentage = 0.1  # percentage of training data to use for validation
positive_data_file = "data/rt-polarity.pos"
negative_data_file = "data/rt-polarity.neg"

# Pre-trained word embeddings
embedding_dim = 300  # dimensionality of embedding
embedding_file = "data/GoogleNews-vectors-negative300.bin"  # word embeddings file

# Preprocessing params
num_frequent_words = 5000  # number of frequent words to retain

# Model hyperparameters
polynomial_orders = [4]  # Chebyshev polynomial orders (i.e. filter sizes)
num_features = [128]  # no. of features for every GCL
pooling_sizes = [1]  # pooling sizes (1 (no pooling) or power of 2)
dropout_keep_prob = 0.5  # dropout keep probability
l2_reg_lambda = 0.0  # L2 regularization lambda

# gCNN parameters
number_edges = 16
coarsening_levels = 0

# Training parameters
learning_rate = 1e-3
batch_size = 64
num_epochs = 200

# TF training params
evaluate_every = 100  # evaluate model on validation set after this many steps
checkpoint_every = 100  # save model after this many steps
num_checkpoints = 5  # number of checkpoints to store

# Misc. parameters
allow_soft_placement = True  # allow device soft device placement i.e. fall back on available device
log_device_placement = False  # log placement of operations on devices


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x, y = data.load_data_and_labels(positive_data_file, negative_data_file)
max_document_length = max([len(x_i.split(" ")) for x_i in x])
print("Max. Sentence Length: {}".format(max_document_length))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x = x[shuffle_indices]
y = y[shuffle_indices]

# Split train/test set
test_sample_index = -1 * int(test_sample_percentage * float(len(y)))
x_train, x_test = x[:test_sample_index], x[test_sample_index:]
y_train, y_test = y[:test_sample_index], y[test_sample_index:]

del x, y  # don't need these anymore

# Build vocabulary and extract term-document matrices
# TODO: tf-idf? Also change test data vectorizer.
# TODO: sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english')
x_train = vectorizer.fit_transform(x_train)
train_vocab = vectorizer.get_feature_names()

print("Vocabulary Size: {}".format(len(train_vocab)))
print("Train/Test Split: {}/{}".format(len(y_train), len(y_test)))

# Keep only N most-frequent words in data, vocabulary & embeddings
word_freqs = x_train.sum(axis=0)
word_freqs = np.squeeze(np.asarray(word_freqs))
freq_idx = np.argsort(word_freqs)[::-1]
freq_idx = freq_idx[:num_frequent_words]
x_train = x_train[:, freq_idx]
train_vocab = [train_vocab[i] for i in freq_idx]

print("Vocabulary Size (Reduced): {}".format(len(train_vocab)))

# Initialize embedding matrix from pre-trained word2vec embeddings. 0.25 is chosen so that unknown vectors
# have (approximately) the same variance as pre-trained ones.
embeddings = np.random.uniform(-0.25, 0.25, (len(train_vocab), embedding_dim))

# Construct reverse lookup vocabulary
reverse_vocab = {k: v for v, k in enumerate(train_vocab)}

# Process Google News word2vec file (in a memory-friendly way) and store relevant embeddings.
print("Loading pre-trained embeddings from {}...".format(embedding_file))
words_found = 0
with open(embedding_file, "rb") as f:
    header = f.readline()
    vocab_size, embedding_size = map(int, header.split())
    binary_len = np.dtype("float32").itemsize * embedding_size
    for line in range(vocab_size):
        word = []
        while True:
            ch = f.read(1).decode("latin-1")
            if ch == " ":
                word = "".join(word)
                break
            if ch != "\n":
                word.append(ch)
        idx = reverse_vocab[word] if word in reverse_vocab else None
        if idx != None:
            embeddings[idx] = np.fromstring(f.read(binary_len), dtype="float32")
            words_found += 1
        else:
            f.read(binary_len)
print("Word Embeddings Extracted: {}".format(words_found))
print("Word Embeddings Randomly Initialized: {}".format(len(train_vocab) - words_found))

# Process test data using train_vocab
vectorizer = sklearn.feature_extraction.text.CountVectorizer(vocabulary=train_vocab)
x_test = vectorizer.fit_transform(x_test)

# Normalize data to unit length
x_train = x_train.astype(np.float64)
x_train = sklearn.preprocessing.normalize(x_train, axis=1, norm='l1')
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float64)
x_test = sklearn.preprocessing.normalize(x_test, axis=1, norm='l1')
x_test = x_test.astype(np.float32)


# Feature Graph
# ==================================================

# Construct graph
dist, idx = graph.distance_sklearn_metrics(embeddings, k=number_edges, metric='cosine')
A = graph.adjacency(dist, idx)
A = graph.replace_random_edges(A, 0)

# Compute coarsened graphs
graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
L = [graph.laplacian(A, normalized=True) for A in graphs]

del embeddings, dist, idx, A, graphs  # don't need these anymore

# Reindex nodes to satisfy a binary tree structure
x_train = scipy.sparse.csr_matrix(coarsening.perm_data(x_train.toarray(), perm))
x_test = scipy.sparse.csr_matrix(coarsening.perm_data(x_test.toarray(), perm))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                  log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = GraphCNN(L=L, K=polynomial_orders, F=num_features, p=pooling_sizes,
                       batch_size=batch_size,
                       num_vertices=x_train.shape[1],
                       num_classes=y_train.shape[1],
                       l2_reg_lambda=l2_reg_lambda)

        # Define training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "v4", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Test Summary Writer
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpoint directory & saver
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step.
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss,
                                                           cnn.accuracy],
                                                          feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: Step {}, Loss {:g}, Accuracy {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def test_step(x_test, y_test, writer=None):
            """
            Evaluates model on a test set.
            """
            # TODO: Hacky workaround to test model since batch_size is fixed. Arbitrary batch_sizes are
            # causing issues with Tensor shapes in the model.
            step = 0
            size = x_test.shape[0]
            losses = 0
            predictions = np.empty(size)
            for begin in range(0, size, batch_size):
                end = begin + batch_size
                end = min([end, size])

                x_batch = np.zeros((batch_size, x_test.shape[1]))
                x_batch[:end - begin] = x_test[begin:end]

                y_batch = np.zeros((batch_size, y_test.shape[1]))
                y_batch[:end - begin] = y_test[begin:end]

                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, batch_pred, batch_loss = sess.run([global_step, cnn.predictions, cnn.loss],
                                                        feed_dict)

                predictions[begin:end] = batch_pred[:end - begin]
                losses += batch_loss

            accuracy = sklearn.metrics.accuracy_score(np.argmax(y_test, 1), predictions)
            loss = losses * batch_size / size

            time_str = datetime.datetime.now().isoformat()
            print("{}: Step {}, Loss {:g}, Accuracy {:g}".format(time_str, step, loss, accuracy))

            summary = tf.Summary()
            summary.value.add(tag="loss_1", simple_value=loss)
            summary.value.add(tag="accuracy_1", simple_value=accuracy)

            if writer:
                writer.add_summary(summary, step)
            return accuracy

        # Convert sparse matrices to arrays
        # TODO: Is there a workaround for this? Doesn't seem memory efficient.
        # TODO: https://github.com/tensorflow/tensorflow/issues/342#issuecomment-160354041
        # TODO: https://github.com/tensorflow/tensorflow/issues/342#issuecomment-273463729
        # TODO: https://stackoverflow.com/questions/37001686/using-sparsetensor-as-a-trainable-variable
        x_train = np.squeeze([x_i.toarray() for x_i in x_train])
        x_test = np.squeeze([x_i.toarray() for x_i in x_test])

        # Generate batches
        batches = data.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)

        # Maximum test accuracy
        max_accuracy = 0

        # Training loop
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                accuracy = test_step(x_test, y_test, writer=test_summary_writer)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                print("Max. Test Accuracy: {:g}".format(max_accuracy))
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
