## REFERENCES:
[MNIST ppt.pdf](https://github.com/user-attachments/files/17566336/MNIST.ppt.pdf)- this file contains the presentation which summarises the project objectives and their solutions

[MNIST report.pdf](https://github.com/user-attachments/files/17566352/MNIST.report.pdf)- this file contains the report which highlights the important features of the project.

# MNIST HANDWRITTEN DIGIT RECOGNITION 
This project aims to develop an application that utilizes the MNIST dataset, a well-known benchmark for evaluating image processing systems, to recognize handwritten digits. We implement a Convolutional Neural Network (CNN), a specialized deep learning architecture that excels in image classification tasks, to achieve high accuracy in digit recognition. 
The application features a user-friendly graphical user interface (GUI) built with the Tkinter library, allowing users to draw digits directly on the screen. The drawn digit is then processed and classified by the trained CNN model in real-time, providing immediate feedback on the recognized digit. This project 
not only demonstrates the practical application of deep learning techniques but also enhances user interaction through its intuitive interface. The successful implementation of this application showcases the potential of machine learning in automating and improving handwriting recognition tasks, paving the way for further advancements in intelligent systems.

## MNIST DATASET
https://www.kaggle.com/datasets/hojjatk/mnist-dataset
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets. The MNIST database contains 60,000 training images and 10,000 testing images, which were were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.

##LIBRARIES USED 
1. from keras.models import load_model
2. from tkinter import * 
3. import tkinter as tk
4. import win32gui
5. from PIL import ImageGrab, ImageOps
6. import numpy as np

![CODE](image.png)
![OUTPUT](image-1.png)

## RESOURCES USED
1. https://keras.io/examples/vision/handwriting_recognition/ 
2. https://www.geeksforgeeks.org/handwritten-digit-recognition-usingneural-network/ 
3. https://www.researchgate.net/publication/356535395_Handwritten_Digit_Recognition_System 
4. https://data-flair.training/blogs/python-deep-learning-projecthandwritten-digit-recognition/ 
5. https://machinelearningmastery.com/handwritten-digit-recognitionusing-convolutional-neural-networks-python-keras/ 
6. https://towardsdatascience.com/handwritten-digit-mnist-pytorch977b5338e627