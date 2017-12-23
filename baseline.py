import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import data


# Parameters
# ==================================================

# Data params
test_sample_percentage = 0.1  # percentage of training data to use for validation
positive_data_file = "data/rt-polarity.pos"
negative_data_file = "data/rt-polarity.neg"


# Data Preparation
# ==================================================

# Load data
x, y = data.load_data_and_labels(positive_data_file, negative_data_file)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
test_sample_index = -1 * int(test_sample_percentage * float(len(y)))
x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]

# Transform targets from arrays to labels
y_train = np.argmax(y_train, 1)
y_test = np.argmax(y_test, 1)

# Linear Support Vector Classifier
svm_clf = Pipeline([('vect', TfidfVectorizer()),
                    ('clf',  LinearSVC())])
svm_clf.fit(x_train, y_train)
predicted = svm_clf.predict(x_test)
print('Linear SVC Accuracy: {:.4f}'.format(np.mean(predicted == y_test)))

# Multinomial Naive Bayes Classifier
bayes_clf = Pipeline([('vect', TfidfVectorizer()),
                      ('clf', MultinomialNB())])
bayes_clf.fit(x_train, y_train)
predicted = bayes_clf.predict(x_test)
print('Naive Bayes Accuracy: {:.4f}'.format(np.mean(predicted == y_test)))
