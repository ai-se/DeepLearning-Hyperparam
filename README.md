# DeepLearning-Hyperparam

## File structure

```
.
├── data_utils.py # Utility for dataset loading
├── demo.ipynb # Demonstration of how the models are trained
├── hyper_param
│   ├── cnntext_hyperparam.txt
│   ├── default_cnntext_hyperparam.txt
│   ├── default_lstm_hyperparam.txt
│   ├── default_mnistcnn_hyperparam.txt
│   ├── default_resnet_hyperparam.txt
│   ├── lstm_hyperparam.txt
│   ├── mnistcnn_hyperparam.txt
│   └── resnet_hyperparam.txt
├── model_arch.py # Definition of Keras deep learning models
└── README.md
```

## Hyperparam Loading

Under the hyper_param directory, there are pre-defined search spaces for every DL model. The hyperparams are stored in JSON format. You can load the file by:
```
with open("hyper_param/mnistcnn_hyperparam.txt", "r") as ifile:
    mnistcnn_hyperparam = json.loads(ifile.read())
```

For each dict that contains the hyperparam, it has secondary indexes:
```
    parameter_dict -> parameter_type: {parameter_name: possible value}
    parameter_type: [categorical, numerical] # Numerical category is defined as [lower, upper, sample_method]
```

Notice: for each model we have a default hyperparam, which should provide acceptable performance for each model.

## Model Loading

We have four models: `class NaiveCNNHyper`, `class ResNetHyper`, `class LSTMTextHyper`, `class CNNTextHyper`. They have the same interfaces for initialization (`function build_model`), training and evaluation.

Please refer to `demo.ipynb` for more information.
