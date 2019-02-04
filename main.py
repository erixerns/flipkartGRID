import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf


# Returns an array that gives the coords of starting point
# of each segment.
def getSegments(chunks):
    # X and Y are constant
    height = 640
    width = 480
    wd = width / chunks
    ht = height / chunks
    arr = []

    for i in range(0, chunks):
        for j in range(0, chunks):
            arr.append((int(i * wd), int(j * ht)))

    return arr


def areaOfBox(l):
    return abs((l[2] - l[0]) * (l[3] - l[1]))


# list1 x1,y1,x2,y2
# list2 xmin,ymin,xmax,ymax


def getArea(list1, list2):
    # topleft
    f1 = list1[0] > list2[0] and list1[1] > list2[1] and list1[0] < list2[2] and list1[1] < list2[3]
    # topright
    f2 = list1[2] > list2[0] and list1[1] > list2[1] and list1[2] < list2[2] and list1[1] < list2[3]
    # bottomleft
    f3 = list1[0] > list2[0] and list1[3] > list2[1] and list1[0] < list2[2] and list1[3] < list2[3]
    # bottomright
    f4 = list1[2] > list2[0] and list1[3] > list2[1] and list1[2] < list2[2] and list1[3] < list2[3]

    if ((f1 or f2 or f3 or f4) == 0):
        return 0

    elif ((f1 and f2 and f3 and f4) == 1):
        return areaOfBox(list1) / ((list1[3] - list1[1]) * (list1[2] - list1[0]))
    else:
        x1 = max(list1[0], list2[0])
        x2 = min(list1[2], list2[2])
        y1 = max(list1[1], list2[1])
        y2 = min(list1[3], list2[3])
        return areaOfBox([x1, y1, x2, y2]) / ((list1[3] - list1[1]) * (list1[2] - list1[0]))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


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
        img = plt.imread("images/" + trainingCSV.ix[i - 1, 0])
        ax = fig.add_subplot(rows, columns, i)
        rect = patches.Rectangle((trainingCSV.ix[i - 1, 1], trainingCSV.ix[i - 1, 3]),
                                 trainingCSV.ix[i - 1, 2] - trainingCSV.ix[i - 1, 1],
                                 trainingCSV.ix[i - 1, 4] - trainingCSV.ix[i - 1, 3], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
        plt.imshow(img)
    plt.show()

    # Note: Every image is 480 x 640

    imageSegment = []
    segmentClasses = []


    # Split the images into chunks
    chunks = 5
    rows = chunks
    cols = chunks
    for j in range(2000):
        print("\nImg: ",j," of 2000")
        img = plt.imread("images/" + trainingCSV.ix[j, 0])
        # img.shape

        # plt.imshow(img)
        # plt.show()

        # fig1 = plt.figure(figsize=(8, 8))
        chunkSegment = getSegments(chunks)
        # Segment the images
        for i in range(1, rows * cols + 1):
            print(".",end='')
            x1 = chunkSegment[i - 1][0]
            x2 = chunkSegment[i - 1][0] + int(480 / chunks)
            y1 = chunkSegment[i - 1][1]
            y2 = chunkSegment[i - 1][1] + int(640 / chunks)

            # Segment the image and store it in list
            segment = rgb2gray(img[x1:x2, y1:y2, :])
            segment = segment.astype(np.float16)
            imageSegment.append(segment)

            # Set class for each segment of image
            area = getArea([y1, x1, y2, x2],
                           [trainingCSV.ix[0, 1], trainingCSV.ix[0, 3], trainingCSV.ix[0, 2], trainingCSV.ix[0, 4]])
            if area >= 0.5:
                segmentClasses.append(1)
            else:
                segmentClasses.append(0)

            # fig1.add_subplot(rows, cols, i)
            # plt.imshow(segment)
        # plt.show()


    print("\nTotal Segments: ", len(imageSegment))
    imageSegment = np.asarray(imageSegment)
    # classes = ['nothing', 'object']
    classes = [0, 1]

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
        tf.keras.layers.Flatten(input_shape=(480 / chunks, 640 / chunks)),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(imageSegment, segmentClasses, epochs=20)

    # Run Main


if __name__ == "__main__":
    main()
