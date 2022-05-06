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
    #model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 1e3), loss= tf.keras.losses.BinaryCrossentropy(), metrics = ['BinaryAccuracy', 'AUC'])
    print(train_generator[1])
    # model.fit(train_generator[0][0:100][0],
    #     batch_size = 10,
    #     epochs=2)
    # # model.fit(train_generator,
    # #     batch_size = 500,
    # #     epochs=2)
    # model.summary() 
    
    # visualizer(model, format='png', view=True)
    
    # print("Evaluate model on test data")
    # results = model.evaluate(test_generator, batch_size=500)
    # print("test loss, test acc:", results)

    # inputs = test_generator[0]
    # labels = test_generator.class_indices.keys()
    # predictions = model.predict(test_generator)
    # visualize_results(inputs, predictions, labels, 'benign', 'malignant')
    
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