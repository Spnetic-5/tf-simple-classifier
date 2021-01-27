import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# pixel values in the range of 0 to 255:

# Scaling these values to a range of 0 to 1 before feeding them to the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0


'''Building the model'''
model = keras.Sequential(
   [
    # transforming the format of the images from a two-dimensional array(28*28) to a one-dimensional array(28*28)
    keras.layers.Flatten(input_shape=(28,28)),
    # first Dense layer has 128 neurons
    keras.layers.Dense(128, activation='relu'),
    # The second layer returns a logits array with length of 10
    keras.layers.Dense(10)
   ] 
)

'''Compiling the model'''
# Optimizer : model is updated based on the data
# loss : measures how accurate the model is during training
# metrics : used to monitor the training and testing steps
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

'''Training the Model'''
model.fit(train_images, train_labels, epochs=10)

# Accurancy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(np.argmax(predictions[0]))