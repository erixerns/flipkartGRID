import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf



# Main function
def main():
    trainingCSV = pd.read_csv("training.csv")

    # Showing type of data
    print("Training Set: ")
    print(trainingCSV.head())

    # Showing first 10 images with object box around them
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        img = plt.imread("images/"+trainingCSV.ix[i-1, 0])
        ax = fig.add_subplot(rows, columns, i)
        rect = patches.Rectangle((trainingCSV.ix[i-1, 1], trainingCSV.ix[i-1, 3]), trainingCSV.ix[i-1, 2]-trainingCSV.ix[i-1, 1], trainingCSV.ix[i-1, 4]-trainingCSV.ix[i-1, 3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.imshow(img)
    plt.show()

    # Note: Every image is 480 x 640

    # classes = ['object', 'nothing']
    classes = [0, 1]

    chunk=5
    lenImages=10
    # Segment the image and store it in list
    imageSegment=[]
    for i in range(10):
        imageSegment.append(getSegment(chunk))

    # Set class for each segment of image
    segmentClasses=[]
    for i in range(lenImages*25):
        if getArea(
                [imageSegment[i%25][0], imageSegment[i%25][0]+480/chunk, imageSegment[i%25][1], imageSegment[i%25][1]+640/chunk],
                trainingCSV.ix[i//25, 1:4]):
            segmentClasses.append(1)
        else:
            segmentClasses.append(0)

    # Make the model
    # Add a dropout layer as well?

    # Model Description:
    # The input layer is flattened from 480/chunk, 640/chunk, 3 to a linear array
    # There are 2 layers with relu activation (best in class)
    # The final layer has two labels, either object or not object and
    # the activation is softmax as if gives a smooth value from 0 to 1
    # which tells us how much an image belongs to a class, i.e
    # if the image is 0.9 object or 0.6 object
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(480/chunk, 640/chunk)),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(imageSegment, segmentClasses, epoch=5)





# Run Main
if __name__=="__main__":
    main()