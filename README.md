# CNN for Rotten Tomatoes Movie Reviews

> Code based on Denny Britz's tutorial on building a CNN for text classification of Rotten Tomatoes movie reviews.

### Accuracy of Model

![Model Accuracy](plots/1506156971-Accuracy.png)

The training accuracy is much higher than the test accuracy, which suggests that the model is overfitting the training data. This could be because of the relatively small dataset or weak regularization. I plan on revisiting this model once I have access to a better system to train the model on (currently training on my personal MacBook Pro, which is a really slow process).

#### References

- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- https://github.com/yoonkim/CNN_sentence
- [Implementing a CNN for Text Classification - Denny Britz](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- https://github.com/dennybritz/cnn-text-classification-tf
