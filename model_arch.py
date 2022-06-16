from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend

class DLModels(metaclass=ABCMeta):
    # Abstract classes for all Deep Learning models
    @abstractmethod
    def build_model(self):
        """
        Compile the model and return the corrsponding model.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluate the model.
        Return:
            Corresponding metrics (like accuracy)
        """
        pass

def choose_optimizer(optimizer: str, learning_rate: float):
    if optimizer == 'SGD':
        opt = keras.optimizers.SGD(learning_rate)
    elif optimizer == 'RMSprop':
        opt = keras.optimizers.RMSprop(learning_rate)
    elif optimizer == 'Adam':
        opt = keras.optimizers.Adam(learning_rate)
    elif optimizer == 'Adadelta':
        opt = keras.optimizers.Adadelta(learning_rate)
    elif optimizer == 'Adagrad':
        opt = keras.optimizers.Adagrad(learning_rate)
    elif optimizer == 'Adamax':
        opt = keras.optimizers.Adamax(learning_rate)
    elif optimizer == 'Nadam':
        opt = keras.optimizers.Nadam(learning_rate)
    elif optimizer == 'Ftrl':
        opt = keras.optimizers.Ftrl(learning_rate)
    return opt

class NaiveCNNHyper(DLModels):
    def __init__(self, params: dict):
        """
        Pass the dict of parameters to init the
        CNN on MNIST dataset.
        """
        super(NaiveCNNHyper, self).__init__()
        self.params = params
        self.filters = params["filters"]
        self.kernel_size = params["kernel_size"]
        self.activation_methods = params["activation"]
        self.drop_out_ratio = params["dropout_ratio"]
        self.pool_type = params["pool_type"]
        self.loss_function = params["loss_function"]
        self.optimizer = params["optimizer"]
        self.learning_rate = params["learning_rate"]
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]
        # Begin build the model
        
        inputs = layers.Input(shape=(28, 28))
        x = layers.Reshape(target_shape=(28, 28, 1))(inputs)
        x = layers.Conv2D(filters=self.filters,
                          kernel_size=self.kernel_size, 
                          activation=self.activation_methods)(x)
        if self.pool_type == 'max':
            x = layers.MaxPooling2D(2)(x)
        elif self.pool_type == 'avg':
            x = layers.AveragePooling2D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(self.drop_out_ratio)(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        self.model = keras.Model(inputs, outputs)
    
    def build_model(self):
        """
        Built the model by given optimizer and loss function
        Return:
            return the compiled models ready for training
        """
        opt = choose_optimizer(self.optimizer, self.learning_rate)
        self.model.compile(optimizer=opt,
                           loss=self.loss_function,
                           metrics=["accuracy"])
        return self.model
    
    def train(self, train_input, train_label):
        self.model.fit(train_input, train_label, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)
    
    def evaluate(self, test_input, test_label):
        result = self.model.evaluate(test_input, test_label, return_dict=True, batch_size=self.batch_size)
        return result["accuracy"]

# ===== Below is the ResNet architecture =====

class ResNetHyper(DLModels):
    def __init__(self, params: dict, input_shape = (32, 32, 3), classes=10):
        super(ResNetHyper, self).__init__()
        self.params = params
        self.kernel_size = params['kernel_size']
        self.filters_1 = params['filters_1']
        self.filters_2 = params['filters_2']
        self.filter_3 = params['filter_3']
        self.filter_4 = params['filter_4']
        self.filters_5 = params['filters_5']
        self.activation = params['activation']
        self.dropout_ratio = params['dropout_ratio']
        self.pooling_1 = params['pooling_1']
        self.pooling_2 = params['pooling_2']
        self.loss_function = params['loss_function']
        self.optimizer = params['optimizer']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        
        conv3_depth = 4
        conv4_depth = 6
        preact = True
        use_bias = True
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        inputs = layers.Input(shape=input_shape)
        x = inputs
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
        x = layers.Conv2D(self.filters_1, self.kernel_size, strides=2, use_bias=use_bias, name='conv1_conv')(x)
        if preact is False:
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
            x = layers.Activation(self.activation, name='conv1_relu')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        if self.pooling_1 == 'avg':
            x = layers.AveragePooling2D(3, strides=2, name='pool1_pool')(x)
        elif self.pooling_1 == 'max':
            x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
        x = stack2(x, self.filters_2, 3, name='conv2', activation_methods=self.activation)
        x = stack2(x, self.filter_3, conv3_depth, name='conv3', activation_methods=self.activation)
        x = stack2(x, self.filter_4, conv4_depth, name='conv4', activation_methods=self.activation)
        x = stack2(x, self.filters_5, 3, stride1=1, name='conv5', activation_methods=self.activation)
        if preact is True:
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
            x = layers.Activation(self.activation, name='post_relu')(x)
        if self.pooling_2 == 'globalavg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif self.pooling_2 == 'globalmax':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)
        
        x = layers.Dropout(self.dropout_ratio)(x)
        outputs = layers.Dense(classes, activation='softmax', name='probs')(x)
        self.model = keras.Model(inputs, outputs, name='ResNet')
    
    def build_model(self):
        opt = choose_optimizer(self.optimizer, self.learning_rate)
        self.model.compile(optimizer=opt, loss=self.loss_function, metrics=['accuracy'])
        return self.model
    
    def train(self, train_input, train_label):
        self.model.fit(train_input, train_label, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)
    
    def evaluate(self, test_input, test_label):
        result = self.model.evaluate(test_input, test_label, return_dict=True, batch_size=self.batch_size)
        return result["accuracy"]
        
                
def stack2(x, filters, blocks, stride1=2, name=None, activation_methods='relu'):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1', activation_methods=activation_methods)
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i), activation_methods=activation_methods)
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks), activation_methods=activation_methods)
    return x



def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None, activation_methods='relu'):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
    preact = layers.Activation(activation_methods, name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation(activation_methods, name=name + '_1_relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation(activation_methods, name=name + '_2_relu')(x)
    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x

# ===== Below is the CNN for NLP architecture =====

class CNNTextHyper(DLModels):
    def __init__(self, params: dict, max_features = 20000):
        """
        Build the CNN for text task.
        parameters
            max_features: Only consider the top max_features words
            
        """
        super(CNNTextHyper, self).__init__()
        self.filters_1 = params['filters_1']
        self.filters_2 = params['filters_2']
        self.kernel_size_1 = params['kernel_size_1']
        self.kernel_size_2 = params['kernel_size_2']
        self.neurons = params['neurons']
        self.embedding_dim = params['embedding_dim']
        self.activation_1 = params['activation_1']
        self.activation_2 = params['activation_2']
        self.activation_3 = params['activation_3']
        self.dropout_ratio_1 = params['dropout_ratio_1']
        self.dropout_ratio_2 = params['dropout_ratio_2']
        self.pool_type = params['pool_type']
        self.loss_function = params['loss_function']
        self.optimizer = params['optimizer']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.learning_rate = params['learning_rate']
        
        # Build the model
        inputs = tf.keras.Input(shape=(None,), dtype="int64")
        x = layers.Embedding(max_features, self.embedding_dim)(inputs)
        x = layers.Dropout(self.dropout_ratio_1)(x)
        x = layers.Conv1D(self.filters_1, self.kernel_size_1, padding="valid", 
                          activation=self.activation_1, strides=3)(x)
        x = layers.Conv1D(self.filters_2, self.kernel_size_2, padding="valid", 
                          activation=self.activation_2, strides=3)(x)    
        if self.pool_type == 'globalmax':
            x = layers.GlobalMaxPooling1D()(x)
        elif self.pool_type == 'globalavg':
            x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(self.neurons, activation=self.activation_3)(x)
        x = layers.Dropout(self.dropout_ratio_2)(x)
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
        self.model = tf.keras.Model(inputs, predictions)

    def build_model(self):
        opt = choose_optimizer(self.optimizer, self.learning_rate)
        self.model.compile(optimizer=opt, loss=self.loss_function, metrics=['accuracy'])
        return self.model
    
    def train(self, train_input, train_label):
        self.model.fit(train_input, train_label, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)
    
    def evaluate(self, test_input, test_label):
        result = self.model.evaluate(test_input, test_label, return_dict=True, batch_size=self.batch_size)
        return result["accuracy"]

# ===== Below is the LSTM for NLP architecture =====
class LSTMTextHyper(DLModels):
    def __init__(self, params: dict, maxlen=200, max_features = 20000):
        """
        Build the LSTM for text task.
        parameters
            max_len: Only consider the first max_len words of each movie review
            max_features: Only consider the top max_features words
        """
        super(LSTMTextHyper, self).__init__()
        self.units_1 = params['units_1']
        self.units_2 = params['units_2']
        self.embedding_dim = params['embedding_dim']
        self.dropout_ratio = params['dropout_ratio']
        self.activation_1 = params['activation_1']
        self.activation_2 = params['activation_2']
        self.loss_function = params['loss_function']
        self.optimizer = params['optimizer']
        self.learning_rate = params['learning_rate']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        
        inputs = keras.Input(shape=(maxlen,), dtype="int32")
        x = layers.Embedding(max_features, self.embedding_dim)(inputs)
        x = layers.LSTM(self.units_1, activation=self.activation_1, return_sequences=True)(x)
        x = layers.LSTM(self.units_2, activation=self.activation_2)(x)
        x = layers.Dropout(self.dropout_ratio)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(inputs, outputs)
        
    def build_model(self):
        opt = choose_optimizer(self.optimizer, self.learning_rate)
        self.model.compile(optimizer=opt, loss=self.loss_function, metrics=['accuracy'])
        return self.model
    
    def train(self, train_input, train_label):
        self.model.fit(train_input, train_label, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)
    
    def evaluate(self, test_input, test_label):
        result = self.model.evaluate(test_input, test_label, return_dict=True, batch_size=self.batch_size)
        return result["accuracy"]
        
        