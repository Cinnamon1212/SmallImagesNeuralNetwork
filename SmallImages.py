import os
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def cls():
    os.system('clear')

def main():
    smallImages_mnist = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = smallImages_mnist.load_data()
    class_names = ['airplane', 'automobile', 'bird', ' cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    train_images = x_train / 255.0
    test_images = x_test / 255.0
    cls()
    while True:
        neuronNum = input("Please enter the number of neurons you'd like to use: ")
        try:
            neuronNum = int(neuronNum)
        except ValueError:
            print("Please enter a number!")
            time.sleep(2)
            cls()
        else:
            break

    while True:
        epochsNum = input("Plesae enter the number of epochs you'd like to use: ")
        try:
            epochsNum = int(epochsNum)
        except ValueError:
            print("Please enter a number!")
            time.sleep(2)
            cls()
        else:
            break
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(neuronNum, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(tf.shape(train_images))
    model.fit(train_images, y_train, epochs=epochsNum)
    test_loss, test_acc = model.evaluate(test_images, y_test, verbose=1)
    predictions = model.predict(test_images)
    while True:
        predictFor = input("Please enter the image number you'd like to predict: ")
        try:
            predictFor = int(predictFor)
        except ValueError:
            print("Please enter a number!")
            time.sleep(1)
            cls()
        else:
            if predictFor <= 50000:
                break
            else:
                print("Please enter a value equal to or less than 50,000")
                time.sleep(1)
                cls()
    predicted = class_names[np.argmax(predictions[predictFor - 1])]
    plt.figure()
    plt.imshow(test_images[predictFor - 1])
    plt.title(f"Prediction {predicted}")
    plt.xlabel(f"Accuracy {round(test_acc, 2)}, Loss: {round(test_loss, 2)}")
    plt.colorbar()
    plt.grid(False)
    plt.savefig(f'{predicted}.png')


# https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf





if __name__ == "__main__":
    cls()
    main()
