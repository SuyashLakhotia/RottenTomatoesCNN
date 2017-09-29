# CNN for Rotten Tomatoes Movie Reviews

> This model aims to classify movie reviews from Rotten Tomatoes as either positive or negative.

### Dataset Description

The dataset used are sentences from Pang and Lee's [movie review dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/) — `sentence polarity dataset v1.0`. There are 5,331 positive and 5,331 negative sentences, formed from a vocabulary size of ~20,000.

#### Preprocessing

> The data preprocessing code is available in `data.py` and is identical to the code used in [Kim's paper](https://arxiv.org/pdf/1408.5882.pdf).

1. Load the data from the files inside `data/`.
2. `.strip()` every sentence.
3. Replace any characters that don't match ``[A-Za-z0-9(),!?\'\`]``.
4. Insert a whitespace between a word and `'ve`, `'re` etc.
5. Insert a whitespace before punctuation marks.
6. Delete repeated whitespaces.

## Model v1

> Code based on Denny Britz's TensorFlow adaptation of Kim's model, which is blogged about [here](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).

### Data Preparation

The sentences from the dataset are fed into TensorFlow's `VocabularyProcessor` that builds a vocabulary index and maps each word to an integer between 0 and 18,757 (vocabulary size). Each sentence is padded with special padding tokens `<UNK>` (index of 0 in vocabulary) to fit the maximum sentence size of 56 words.

The data is shuffled and 10% of the dataset is used as the validation set during training.

### Model Description

> The code for the model can be found in `v1_model.py`.

The model consists of an embedding layer followed by multiple convolutional + max-pool layers before the output is classified using a softmax layer.

#### Hyperparameters

- `embedding_size`: The dimensionality of the embeddings (lower-dimensional vector representations of the vocabulary indices).
- `filter_sizes`: The number of words the convolutional filters should cover. For example, `[3, 4, 5]` will create filters that slide over 3, 4 and 5 words respectively.
- `num_filters`: The number of filters per filter size.
- `l2_reg_lambda`: L2 regularization term. Default is 0.
- `dropout_keep_prob`: Probability of keeping a neuron in the dropout layer.

### Accuracy of Model

> The code for training can be found in `v1_train.py`.

The hyperparameters used for training can be found in `v1_train.py`. Any changes to those values in the following runs are noted below.

#### Run 1

![Model Accuracy](plots/1506156971-Accuracy.png)

## References

- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- https://github.com/yoonkim/CNN_sentence
- [Implementing a CNN for Text Classification - Denny Britz](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- https://github.com/dennybritz/cnn-text-classification-tf
