import keras_tuner
import model_arch

def keras_build_model(hp, model_arch_str: str):
    """
    Return the corresponding keras search space for
    given model architecture:
        naivecnn: NaiveCNNHyper
        resnet: ResNetHyper
        cnntext: CNNTextHyper
        lstmtext: LSTMTextHyper
    """
    return_param_dict = dict()
    if model_arch_str.lower() == "naivecnn":
        return_param_dict["filters"] = hp.Choice('filters', [6, 12, 24, 48, 96, 192], default=12)
        return_param_dict["kernel_size"] = hp.Choice('kernel_size', [3, 5, 7, 9], default=3)
        return_param_dict["activation"] = hp.Choice('activation', ['tanh', 'relu', 'sigmoid'], default='relu')
        return_param_dict["pool_type"] = hp.Choice('pool_type', ['max', 'avg'], default='max')
        return_param_dict["loss_function"] = hp.Choice('loss_function', ['categorical_crossentropy', 'poisson', 'kullback_leibler_divergence'],
                            default='categorical_crossentropy')
        return_param_dict["optimizer"] = hp.Choice('optimizer', 
                                                ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'], 
                                                default='Adam')
        return_param_dict["learning_rate"] = hp.Float('learning_rate', 1e-4, 1e-1, sampling='log', default=1e-3)
        return_param_dict["dropout_ratio"] = hp.Float('dropout_ratio', 0.0, 0.9, sampling='linear', default=0.5)
        return_param_dict["batch_size"] = hp.Choice('batch_size', [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], default=128)
        #return_param_dict["epochs"] = hp.Int('epochs', 1, 100, sampling='linear', default=4)
        return_param_dict["epochs"] = hp.Int('epochs', 1, 6, sampling='linear', default=4)
    elif model_arch_str.lower() == "resnet":
        return_param_dict['kernel_size'] = hp.Choice('kernel_size', [3, 5, 7, 9], default=7)
        return_param_dict['filters_1'] = hp.Choice('filters_1', [16, 32, 64, 128], default=64)
        return_param_dict['filters_2'] = hp.Choice('filters_2', [16, 32, 64, 128], default=64)
        return_param_dict['filters_3'] = hp.Choice('filters_3', [16, 32, 64, 128, 256], default=128)
        return_param_dict['filters_4'] = hp.Choice('filters_4', [16, 32, 64, 128, 256, 512], default=256)
        return_param_dict['filters_5'] = hp.Choice('filters_5', [16, 32, 64, 128, 256, 512, 1024], default=512)
        return_param_dict['activation'] = hp.Choice('activation', ['tanh', 'relu', 'sigmoid'], default='relu')
        return_param_dict['dropout_ratio'] = hp.Float('dropout_ratio', 0.0, 0.9, sampling='linear', default=0.5)
        return_param_dict['pooling_1'] = hp.Choice('pooling_1', ['avg', 'max'], default='max')
        return_param_dict['pooling_2'] = hp.Choice('pooling_2', ['globalavg', 'globalmax'], default='globalavg')
        return_param_dict['loss_function'] = hp.Choice('loss_function', ['categorical_crossentropy', 'poisson', 'kullback_leibler_divergence'], default='categorical_crossentropy')
        return_param_dict['optimizer'] = hp.Choice('optimizer', ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'], default='Adam')
        return_param_dict['learning_rate'] = hp.Float('learning_rate', 1e-4, 1e-1, sampling='log', default=1e-2)
        return_param_dict['batch_size'] = hp.Choice('batch_size', [16, 32, 64, 128, 256, 512], default=32)
        #return_param_dict['epochs']= hp.Int('epochs', 1, 100, sampling='linear', default=25)
        return_param_dict["epochs"] = hp.Int('epochs', 1, 1, sampling='linear', default=4)
    elif model_arch_str.lower() == "cnntext":
        return_param_dict['filters_1'] = hp.Choice('filters_1', [8, 16, 32, 64, 128, 256], default=128)
        return_param_dict['filters_2'] = hp.Choice('filters_2', [8, 16, 32, 64, 128, 256], default=128)
        return_param_dict['kernel_size_1'] = hp.Choice('kernel_size_1', [3, 5, 7, 9], default=7)
        return_param_dict['kernel_size_2'] = hp.Choice('kernel_size_2', [3, 5, 7, 9], default=7)
        return_param_dict['neurons'] = hp.Choice('neurons', [8, 16, 32, 64, 128, 256], default=128)
        return_param_dict['embedding_dim'] = hp.Choice('embedding_dim', [8, 16, 32, 64, 128, 256], default=128)
        return_param_dict['activation_1'] = hp.Choice('activation_1', ['tanh', 'relu', 'sigmoid'], default='relu')
        return_param_dict['activation_2'] = hp.Choice('activation_2', ['tanh', 'relu', 'sigmoid'], default='relu')
        return_param_dict['activation_3'] = hp.Choice('activation_3', ['tanh', 'relu', 'sigmoid'], default='relu')
        return_param_dict['dropout_ratio_1'] = hp.Float('dropout_ratio_1', 0.0, 0.9, sampling='linear', default=0.5)
        return_param_dict['dropout_ratio_2'] = hp.Float('dropout_ratio_2', 0.0, 0.9, sampling='linear', default=0.5) 
        return_param_dict['pool_type'] = hp.Choice('pool_type', ['globalmax', 'globalavg'], default='globalmax')
        return_param_dict['loss_function'] = hp.Choice('loss_function', ['binary_crossentropy', 'poisson', 'kullback_leibler_divergence'], default='binary_crossentropy')
        return_param_dict['optimizer'] = hp.Choice('optimizer', ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'], default='Adam')
        return_param_dict['learning_rate'] = hp.Float('learning_rate', 1e-4, 1e-1, sampling='log', default=1e-3)
        return_param_dict['batch_size'] = hp.Choice('batch_size', [16, 32, 64, 128, 256, 512], default=32)
        #return_param_dict['epochs']= hp.Int('epochs', 1, 100, sampling='linear', default=3)
        return_param_dict["epochs"] = hp.Int('epochs', 1, 1, sampling='linear', default=4)
    elif model_arch_str.lower() == "lstmtext":
        return_param_dict['units_1'] = hp.Choice('units_1', [16, 32, 64, 128, 256], default=64)
        return_param_dict['units_2'] = hp.Choice('units_2', [16, 32, 64, 128, 256], default=64)
        return_param_dict['embedding_dim'] = hp.Choice('embedding_dim', [16, 32, 64, 128, 256], default=128)
        return_param_dict['dropout_ratio'] = hp.Float('dropout_ratio', 0.0, 0.9, sampling='linear', default=0.5)
        return_param_dict['activation_1'] = hp.Choice('activation_1', ['tanh', 'relu', 'sigmoid'], default='tanh')
        return_param_dict['activation_2'] = hp.Choice('activation_2', ['tanh', 'relu', 'sigmoid'], default='tanh')
        return_param_dict['loss_function'] = hp.Choice('loss_function', ['binary_crossentropy', 'poisson', 'kullback_leibler_divergence'], default='binary_crossentropy')
        return_param_dict['optimizer'] = hp.Choice('optimizer', ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'], default='Adam')
        return_param_dict['learning_rate'] = hp.Float('learning_rate', 1e-4, 1e-1, sampling='log', default=1e-3)
        return_param_dict['batch_size'] = hp.Choice('batch_size', [16, 32, 64, 128, 256, 512], default=32)
        #return_param_dict['epochs']= hp.Int('epochs', 1, 100, sampling='linear', default=2)
        return_param_dict["epochs"] = hp.Int('epochs', 1, 1, sampling='linear', default=4)
    else:
        raise NotImplementedError
    return return_param_dict

class MyHyperModel(keras_tuner.HyperModel):
    def __init__(self, *args, **kwargs):
        self.model_arch_str = kwargs["model_arch_str"]
        kwargs.pop("model_arch_str")
        super(MyHyperModel).__init__(*args, **kwargs)
    def build(self, hp):
        search_space = keras_build_model(hp, self.model_arch_str)
        #naivecnn: NaiveCNNHyper
        #resnet: ResNetHyper
        #cnntext: CNNTextHyper
        #lstmtext: LSTMTextHyper
        if self.model_arch_str == "naivecnn":
            model_wrapper = model_arch.NaiveCNNHyper(search_space)
        elif self.model_arch_str == "resnet":
            model_wrapper = model_arch.ResNetHyper(search_space)
        elif self.model_arch_str == "cnntext":
            model_wrapper = model_arch.CNNTextHyper(search_space)
        elif self.model_arch_str == "lstmtext":
            model_wrapper = model_arch.LSTMTextHyper(search_space)
        return model_wrapper.build_model()
    def fit(self, hp, model, x, y, **kwargs):
        epochs = hp["epochs"]
        batch_size = hp["batch_size"]
        return model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)