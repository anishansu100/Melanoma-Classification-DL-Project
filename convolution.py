from __future__ import absolute_import
from pickle import TRUE
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



class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class scontain the architecture of our model  that 
        classifies whether the image is malignant or benign. We also determine the 
        learning rate as another hyperparameter with the filter size and number of 
        kernels.
        """
        super(Model, self).__init__()
        self.learning_rate = 1e-3
        self.a_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model = Sequential([
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



    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images in the sequential.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        logits = self.model(inputs)
        return logits

    def loss(self, labels, logits):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        loss_fn = BinaryCrossentropy(from_logits=False)
        return loss_fn(labels, logits)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels - no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        
        :return: the accuracy of the model as a Tensor
        """
        logits_args = tf.argmax(logits)
        logits_labels = tf.argmax(labels)
        correct_predictions = tf.equal(logits_args, logits_labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs):
    '''
    Trains the model on all of the inputs and labels for one epoch.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs includes a tuple of (batches of inputs, batches of labels)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    train_acc_metric = BinaryAccuracy()
    train_auc_metric = AUC()
    # Intializes inputs and labels
    for step in range(len(train_inputs)):
        batch, labels = train_inputs[step]
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            logits = model.call(batch) 
            # Compute the loss value for this minibatch.
            labels = np.expand_dims(labels, axis = 1)
            loss_value = model.loss(labels, logits)
        gradient = tape.gradient(loss_value, model.trainable_variables)
        model.a_optimizer.apply_gradients(zip(gradient, model.trainable_variables))

        # Update training metric.
        train_acc_metric.update_state(labels, logits)
        train_auc_metric.update_state(labels, logits)
        # Log every 200 batches.
        print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
        print("Seen so far: %s samples" % ((step + 1) * 50))
    #Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    train_auc = train_auc_metric.result()
    print("\n")
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    print("Training auc over epoch: %.4f" % (float(train_auc),))
    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    train_auc_metric.reset_states()

def test(model, test_inputs):
    """
    Tests the model on the test inputs and labels. 
    
    :param test_inputs: train inputs includes a tuple of (batches of inputs, batches of labels)
    :return: test accuracy - this should be the average accuracy across
    all batches for one epoch
    """
    total_accuracy = 0
    for step in range(len(test_inputs)):    
        x_batch_test, y_batch_test = test_inputs[step]
        logits = model.call(x_batch_test)
        labels = np.expand_dims(y_batch_test, axis = 1)
        accuracy = model.accuracy(logits, y_batch_test)
        total_accuracy = total_accuracy  + accuracy
    return total_accuracy / len(test_inputs)

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
    Read the  data  initialize our model, and then trains and 
    tests the  model for a number of epochs(a hyperparameter).
    
    :return: None
    '''
    train_generator, test_generator =  get_data('/home/anish_pradhan/Melanoma-Classification-DL-Project/train', '/home/anish_pradhan/Melanoma-Classification-DL-Project/train', True)
    model = Model()
    epoches = 10
    for i in range(epoches):
        print('epoch: ', i)
        train(model, train_generator)
    total = 0
    for j in range(epoches):
        print('epoch: ', i)
        accuracy = test(model, test_generator)
        total = total + accuracy
    total_accuracy = accuracy / epoches
    print(total_accuracy)

if __name__ == '__main__':
    main()