from tensorflow import keras 
from tensorflow.keras.utils import to_categorical

def load_mnist_dataset():
    """
    Load MNIST dataset.
    Return:
        Dict: "training" --> [training_input, training_label]
            "testing" --> [testing_input, testing_label]
    """
    input_shape = (28, 28, 1)
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Do the normalization
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return {"training": [x_train, y_train], "testing": [x_test, y_test]}

def load_cifar10_dataset():
    """
    Load CIFAR-10 dataset
    Return:
        Dict: "training" --> [training_input, training_label]
            "testing" --> [testing_input, testing_label]
    """
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return {"training": [x_train, y_train], "testing": [x_test, y_test]}

def load_imdb_dataset(max_features=20000, maxlen=500):
    """
    Load IMDB dataset.
    Return:
        Dict: "training" --> [training_input, training_label]
            "testing" --> [testing_input, testing_label]
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
        num_words=max_features
    )
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    return {"training": [x_train, y_train], "testing": [x_test, y_test]}
    
