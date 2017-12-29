import tensorflow as tf


class TextSimilarityCNN(object):
    """
    A CNN architecture for text classification using the cosine similarity between words. Composed of an 
    embedding layer, followed by parallel convolutional + max-pooling layer(s) and a softmax layer.
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, embeddings, filter_heights,
                 num_features, l2_reg_lambda):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of L2 regularization loss
        l2_loss = tf.constant(0.0)

        # Extract mini-batch size / test set size
        batch_size = tf.shape(self.input_x)[0]

        # Embedding layer
        with tf.name_scope("embedding"):
            if embeddings is None:
                embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                            name="W")
            else:
                embedding_mat = tf.Variable(embeddings, trainable=True, name="W")

            # Compute cosine similarities between mini-batch examples and all embeddings
            norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_mat), 1, keep_dims=True))  # l2 norm
            normalized_embeddings = embedding_mat / norm
            embedded_x = tf.nn.embedding_lookup(normalized_embeddings, self.input_x)
            normalized_embeddings = tf.reshape(tf.tile(normalized_embeddings, [batch_size, 1]),
                                               [batch_size, vocab_size, embedding_size])
            self.similarity = tf.matmul(embedded_x, normalized_embeddings, transpose_b=True)
            self.similarity = tf.expand_dims(self.similarity, -1)  # expand for .conv2d
            self.similarity = tf.cast(self.similarity, tf.float32)

        # Create a convolution + max-pool layer for each filter size (filter_height x vocab_size)
        pooled_outputs = []
        for i, filter_height in enumerate(filter_heights):
            with tf.name_scope("conv-maxpool-{}".format(filter_height)):
                # Convolution layer
                filter_shape = [filter_height, vocab_size, 1, num_features]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_features]), name="b")
                conv = tf.nn.conv2d(self.similarity,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply non-linearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(value=h,
                                        ksize=[1, sequence_length - filter_height + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_features_total = num_features * len(filter_heights)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_features_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W",
                                shape=[num_features_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Calculate accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
