from __future__ import absolute_import
from functools import total_ordering
from turtle import st
from matplotlib import pyplot as plt
from preprocess import get_data
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, InputLayer
from tensorflow.math import exp, sqrt
from tensorflow.keras.applications import ResNet50
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
    plt.title('Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
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
    train_generator, test_generator =  get_data('/home/anish_pradhan/Melanoma-Classification-DL-Project/train', '/home/anish_pradhan/Melanoma-Classification-DL-Project/train', True)
    model = Sequential([
            InputLayer((200, 400, 3)),
            BatchNormalization(),
            Conv2D(4, 3, 1, activation="relu", padding="valid"),
            Conv2D(4, 3, 1, activation="relu", padding="valid"),
            Conv2D(8, 3, 1, activation="relu", padding="valid"),
            Conv2D(8, 3, 1, activation="relu", padding="valid"),
            MaxPool2D(2, padding="same"),
            Conv2D(16, 3, 1, activation="relu", padding="valid"),
            Conv2D(16, 3, 1, activation="relu", padding="valid"),
            Conv2D(32, 3, 1, activation="relu", padding="valid"),
            Conv2D(32, 3, 1, activation="relu", padding="valid"),
            MaxPool2D(2, padding="same"),
            Dropout(0.3),
            GlobalAveragePooling2D(),
            Flatten(),
            Dense(1, activation='softmax')
            ])
    tf.keras.utils.plot_model(model,to_file='model.png')
    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 1e-3), loss= tf.keras.losses.BinaryCrossentropy(), metrics = ['BinaryAccuracy', 'AUC'])
    model.summary()
    history = model.fit(train_generator,
        batch_size= 50,
        epochs=1,
        shuffle=True)
    visualize_loss(history.history['loss'])
    model.summary()
    
    print("Evaluate model on test data")
    results = model.evaluate(train_generator, batch_size=50)
    print("test loss, test acc:", results)
    


if __name__ == '__main__':
    main()