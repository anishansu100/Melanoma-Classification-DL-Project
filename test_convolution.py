from __future__ import absolute_import
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
    Read the  data  initialize our model, and then trains and 
    tests the  model for a number of epochs(a hyperparameter).
    This file is mostly for comparing results from  our 
    training and testing from convolution.py.
    
    :return: None
    '''
    # Preprocesses the train and testing datasets
    train_generator, test_generator =  get_data('/home/anish_pradhan/Melanoma-Classification-DL-Project/train', '/home/anish_pradhan/Melanoma-Classification-DL-Project/train', True)
    # This is our model
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
    # Uncomment to create a .png file of the model above
    # tf.keras.utils.plot_model(model,to_file='model.png')

    # Compiles the model with the right learning rate, loss, and AUC
    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 1e-3), loss= tf.keras.losses.BinaryCrossentropy(), metrics = ['BinaryAccuracy', 'AUC'])

    #Uncomment if you want to see the model summary
    # model.summary()

    # Model is trained here for 10 epoches
    history = model.fit(train_generator,
        batch_size= 50,
        epochs=10,
        shuffle=True)

    # Uncomment to see the loss vs epoch graph
    # visualize_loss(history.history['loss'])
    
    # This is our testing where the output would be the accuracy
    print("Evaluate model on test data")
    results = model.evaluate(test_generator, batch_size=50)
    print("test loss, test acc:", results)
    


if __name__ == '__main__':
    main()