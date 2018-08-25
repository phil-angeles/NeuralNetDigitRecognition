# Imports
import numpy as np
import matplotlib.pyplot as plt

# Display the Digit from the image
# If the Label and PredLabel is given display it too
def display_digit(image, label=None, pred_label=None):
    if image.shape == (784,):
        image = image.reshape((28, 28))
    label = np.argmax(label, axis=0)
    if pred_label is None and label is not None:
        plt.title('Label: %d' % (label))
    elif label is not None:
        plt.title('Label: %d, Pred: %d' % (label, pred_label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

# Display the convergence of the error while training
def display_convergence(error):
    plt.plot(error)
    plt.title('Error of the NN')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()