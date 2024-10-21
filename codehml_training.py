# Importing necessary libraries from Keras
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Display the shape of the training data
print(x_train.shape, y_train.shape)

# Reshape the data to add a channel dimension (grayscale)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1) # Define input shape for the model

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Normalize the image data to values between 0 and 1(scaling)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Print information about the training and testing data shapes
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Set batch size, number of output classes (digits 0-9), and training epochs
batch_size = 128
num_classes = 10
epochs = 50

# Build the Sequential model
model = Sequential()

# Add a 2D convolutional layer with 32 filters and a 5x5 kernel, with ReLU activation
model.add(Conv2D(40, kernel_size=(5, 5),activation='relu',input_shape=input_shape))

# Add a 2x2 max pooling layer to reduce the dimensionality
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add another convolutional layer with 64 filters and a 3x3 kernel, with ReLU activation
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add another 2x2 max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output to feed into the fully connected layers
model.add(Flatten())

# Add a fully connected layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))

# Add a dropout layer to prevent overfitting (30% dropout rate)
model.add(Dropout(0.3))

# Add another fully connected layer with 64 neurons and ReLU activation
model.add(Dense(64, activation='relu'))

# Add another dropout layer (50% dropout rate)
model.add(Dropout(0.5))

# Add the output layer with 'num_classes' neurons (10) and softmax activation for classification
model.add(Dense(num_classes, activation='softmax'))


# Compile the model with categorical crossentropy loss and the Adadelta optimizer
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# Train the model with the training data, validating with the test data
hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")


# Evaluate the trained model on the test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model in Keras format
model.save('mnist.keras')  # Change from 'mnist.h5' to 'mnist.keras'
print("Saving the model as mnist.keras")

