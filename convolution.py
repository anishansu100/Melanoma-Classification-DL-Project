from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data

import os
import tensorflow as tf
import numpy as np
import random
import math
from keras_visualizer import visualizer 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.math import exp, sqrt
 
class Model(tf.keras.Model):
    def __init__(self, input_size):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size  # Z
        self.hidden = 500  # H_d
        self.dense = None

        ############################################################################################
        # TODO: Implement the fully-connected encoder architecture described in the notebook.      #
        # Specifically, self.encoder should be a network that inputs a batch of input images of    #
        # shape (N, 1, H, W) into a batch of hidden features of shape (N, H_d). Set up             #
        # self.mu_layer and self.logvar_layer to be a pair of linear layers that map the hidden    #
        # features into estimates of the mean and log-variance of the posterior over the latent    #
        # vectors; the mean and log-variance estimates will both be tensors of shape (N, Z).       #
        ############################################################################################
        # Replace "pass" statement with your code
        self.dense = Sequential([
            BatchNormalization(),
            Conv2D(3, 50, 50, activation="relu", padding="same"),
            Conv2D(32, 46, 46, activation="relu", padding="same"),
            MaxPool2D(2, padding="same"),
            Dropout(0.15),
            Conv2D(64, 21, 21, activation="relu", padding="same"),
            Conv2D(64, 19, 19, activation="relu", padding="same"),
            Dropout(0.15),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2,  activation='relu'),
        ])

        
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        



    def call(self, inputs):

        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        prob = self.dense(inputs)
        return prob
        

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    # Intializes inputs and labels
    indices = tf.random.shuffle(tf.range(0, train_inputs.shape[0], dtype=tf.int32))
    shuffled_inputs = tf.gather(train_inputs, indices)
    shuffled_labels = tf.gather(train_labels, indices)
    shuffled_inputs  = tf.image.random_flip_left_right(shuffled_inputs)
    # The division between batch sizes of 64 
    for i in range(0, len(train_inputs), model.batch_size):
        inputs = shuffled_inputs[i: i + model.batch_size]
        labels = shuffled_labels[i: i + model.batch_size]

        # Gradient descent and backpropogation takes place
        with tf.GradientTape() as tape:
            pred = model.call(inputs, False)
            loss = model.loss(pred, labels)
            model.losses.append(loss)
        gradient = tape.gradient(loss, model.trainable_variables)
        model.a_optimizer.apply_gradients(zip(gradient, model.trainable_variables))

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batchesc
    """
    return model.accuracy(model.call(test_inputs, True), test_labels)


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"
    NOTE: DO NOT EDIT
    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    train_generator, test_generator =  get_data('/Users/anishansupradhan/Desktop/CS1430/Melanoma-Classification-DL-Project/train')
    model = Sequential([
            BatchNormalization(),
            Conv2D(32, 3, 3, activation="relu", padding="same"),
            Conv2D(32, 3, 3, activation="relu", padding="same"),
            MaxPool2D(2, padding="same"),
            Dropout(0.15),
            Conv2D(64, 3, 3, activation="relu", padding="same"),
            Conv2D(64, 3, 3, activation="relu", padding="same"),
            Dropout(0.15),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1,  activation='relu'),
        ])
    model.summary() 

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 1e3), loss= tf.keras.losses.BinaryCrossentropy(), metrics = ['BinaryAccuracy'])
    model.fit(train_generator,
        batch_size = 500,
        epochs=2)
    visualizer(model, format='png', view=True)
    
    print("Evaluate model on test data")
    results = model.evaluate(test_generator, batch_size=500)
    print("test loss, test acc:", results)


    
    # # Training Inputs
    # train_inputs, train_labels = get_data("/Users/anishansupradhan/Desktop/CS1430/Melanoma-Classification-DL-Project/preprocess.py",3, 5)

    # # Testing Inputs 
    # test_inputs, test_labels = get_data("/Users/anishansupradhan/Desktop/CS1430/Melanoma-Classification-DL-Project/preprocess.py",3, 5)

    # # Model with 20 epochs
    # model = Model()
    # epoches = 20
    # for i in range(epoches):
    #     train(model, train_inputs, train_labels)
    # test(model, test_inputs, test_labels)


if __name__ == '__main__':
    main()