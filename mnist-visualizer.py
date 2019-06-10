import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, shutil
import cv2

print("Tensorflow version " + tf.__version__)


def display_predictions(model_file_path):
    """Creates 200 jpgs with 50 digits and their predictions each"""

    # Get the mnist database
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize the values and reshape them
    x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Load a model and print it out
    model = tf.keras.models.load_model(model_file_path, compile=True)
    model.summary()

    if not os.path.exists("figs"):
        os.mkdir("figs")

    failed = False

    how_many_digits_per_plot = 50
    for i, _ in enumerate(x_test):

        # New enumerator is needed for the subplots to go from 0 to 99 and start again each time
        if i >= how_many_digits_per_plot:
            j = i - int(i / how_many_digits_per_plot) * how_many_digits_per_plot
        else:
            j = i

        # Subplot settings
        plt.subplot(5, 10, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(x_test[i], 2), cmap=plt.cm.binary)

        # Need an 4d array for the prediction of 1 picture
        x_temp = np.stack(x_test[i:(i + 1)], axis=0)

        # Set the label according to the prediction
        predictions = model.predict(x_temp)
        prediction = np.argmax(predictions[0])

        if prediction == y_test[i]:
            plt.title(str(i + 1) + "\nCorrect", color="black", fontsize=7, y=0.9)
            plt.xlabel('P: ' + str(prediction) + " R: " + str(y_test[i]), color="black", fontsize=7)
        else:
            plt.title(str(i + 1) + "\nWrong", color="red", fontsize=7, y=0.9)
            plt.xlabel('P: ' + str(prediction) + " R: " + str(y_test[i]), color="red", fontsize=7)
            failed = True

        # Every 100th iteration save the plot and start another one
        if (j + 1) % how_many_digits_per_plot == 0:
            plt.tight_layout()
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)

            name = "figs/" + str(int((i + 1) / how_many_digits_per_plot))
            if failed:
                plt.savefig(name + ",failed.jpg", dpi=300, optimize=True)
            else:
                plt.savefig(name + ".jpg", dpi=300, optimize=True)

            plt.close()
            print(int((i + 1) / 50), "out of", int(len(x_test) / how_many_digits_per_plot), "done")
            failed = False


def create_movie():
    """Creates a movie from the digits jpgs"""
    print("\nDone predicting. Creating video")
    images = ["figs/" + img for img in os.listdir("figs/")]

    # Sort the jpgs by their number because for some reason 1 is not followed by 2 but by 10 then 100
    images.sort(key=lambda x: int(x.replace("figs/", "").replace(".jpg", "").split(",")[0]))

    height, width, _ = cv2.imread(images[0]).shape
    video = cv2.VideoWriter("mnist model.mp4", -1, fps=15, frameSize=(width, height))
    for image in images:
        temp = image.split(",")
        if len(temp) == 2:  # True if the jpg contains an failed prediction
            for _ in range(2 * 15):  # 2s * 15fps => add the file 30 times to see it for 2s
                video.write(cv2.imread(image))
        else:
            video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()
    print("\nSaved video as mnist model.mp4")


def main(args):
    display_predictions(str(args[1]))
    create_movie()

    # If not specified delete the temporary figs
    try:
        if int(args[2]) == 0:
            try:
                shutil.rmtree("figs")
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
    except IndexError:
        try:
            shutil.rmtree("figs")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


if __name__ == '__main__':
    main(sys.argv)
