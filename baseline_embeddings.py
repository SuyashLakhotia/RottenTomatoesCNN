import numpy as np
from tensorflow.contrib import learn
from sklearn.svm import LinearSVC

import data


# Parameters
# ==================================================

# Data params
test_sample_percentage = 0.1  # percentage of training data to use for validation
positive_data_file = "data/rt-polarity.pos"
negative_data_file = "data/rt-polarity.neg"

# Pre-trained word embeddings
embedding_dim = 300  # dimensionality of embedding
embedding_file = "data/GoogleNews-vectors-negative300.bin"  # word embeddings file


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = data.load_data_and_labels(positive_data_file, negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print("Max. Sentence Length: {}".format(max_document_length))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
test_sample_index = -1 * int(test_sample_percentage * float(len(y)))
x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]

del x, y, x_shuffled, y_shuffled  # don't need these anymore

print("Vocabulary Size: {}".format(len(vocab_processor.vocabulary_)))
print("Train/Test Split: {}/{}".format(len(y_train), len(y_test)))

# Initialize embedding matrix from pre-trained word2vec embeddings. 0.25 is chosen so that unknown vectors
# have (approximately) the same variance as pre-trained ones.
embeddings = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), embedding_dim))

# Process Google News word2vec file (in a memory-friendly way) and store relevant embeddings.
print("Loading pre-trained embeddings from {}...".format(embedding_file))
embeddings = data.load_word2vec(embedding_file, vocab_processor.vocabulary_, embedding_dim, tf_VP=True)

# Embed the data with the extracted embeddings
x_train = np.array([np.mean([embeddings[idx] for idx in sentence], axis=0) for sentence in x_train])
x_test = np.array([np.mean([embeddings[idx] for idx in sentence], axis=0) for sentence in x_test])

# Transform targets from arrays to labels
y_train = np.argmax(y_train, 1)
y_test = np.argmax(y_test, 1)


# Training
# ==================================================

# Train & test classifier
classifier = LinearSVC().fit(x_train, y_train)
predicted = classifier.predict(x_test)
accuracy = np.mean(predicted == y_test)

print("Accuracy: {:.4f}".format(accuracy))
