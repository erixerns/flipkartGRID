import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



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

    # Every image is 480 x 640





# Run Main
if __name__=="__main__":
    main()