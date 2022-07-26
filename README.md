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
├── keras_tuner_hyper.py
├── tuner_example.ipynb
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


## Keras-tuner

Keras tuner seems to be jupyter-notebook friedly because some of the search summaries are just printed out instead of returning a data structure. 

Please refer to `tuner_example.ipynb` for details. The search classes are implemented under `keras_tuner_hyper.py`.

Generally speaking, one can first init the base model `keras_tuner_hyper.MyHyperModel`, then do the search `tuner.search`, and get the best topK hyper-parameters useing `all_hyper_parameter_list = tuner.get_best_hyperparameters(num_trials=TRIALS)`.

We do not record the trained model when doing the search, becauase Keras-tuner suggests to re-train the model from the given hyperparameters. In default the model saving in Keras-tuner is not implemented. (Refer to [base_tuner](https://github.com/keras-team/keras-tuner/blob/f1a475eb51ce4692a249906ec1a54e368fc7ae2b/keras_tuner/engine/base_tuner.py#L207))

## TOSEM Data

Please download the data at: https://drive.google.com/file/d/1t_Pf9jM41F7hNNJiXz-sAxEt4fkAhDCK/view?usp=sharing and unfold it in the root directory.

We have four folders for each of the model+data experiment. Under each directory, `minst_cnn_D1_fix_filters_metrics` means when applying the search, the `filters` hyperparameter is fixed. All the files here are kept in JSON format recording the corresponding results when applying Keras search.

"""
Dict:
    hyperparameters: the hyperparameters keras-tuner gives.
    original_model: the performance of the original model (accuracy, latency, flop, etc.)
    pruned_model: original model after pruned (Not useful for now).
    quantized_model: original model after quantized (Not useful for now).
    pruned_quantized_model: original model after quantized and pruned (Not useful for now).
"""

Under notebook `TOSEM_data_analysis.ipynb`, we have examples to load the data. 