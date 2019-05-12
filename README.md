# Cifar-100-Classifier
A Cifar-100 classifier with well designed architecture and good OOP design.<br>
This project follows the **best practice tensorflow folder structure of** [Tensorflow Best Practice](https://github.com/MrGemy95/Tensorflow-Project-Template) 


# Table of contents

- [Project structure](#project-structure)
- [Download pretrained models](#Download-pretrained-models)
- [Dependencies](#install-dependencies)
- [Config file](#config-file)
- [How to train](#Model-training)
- [How to test](#Model-testing)
- [How to predict class of images using pretrained models](#Make-predictions-with-pretrained-models)
- [Implementation details](#Implementation-details)
     - [TinyVGG architecture](#TinyVGG-model-arch)
     - [TinyVGG training)(#TinyVGG-training)
     - [TinyVGG testing)(#TinyVGG-testing)



# Project structure
--------------

```
├── Configs            
│   └── config_model.json  - Contains the paths used and config of the models(learning_rate, num_epochs, ...)
│     
├──  base
│   ├── base_model.py   - This file contains the abstract class of all models used.
│   ├── base_train.py   - This file contains the abstract class of the trainer of all models used.
│   └── base_test.py    - This file contains the abstract class of the testers of all models used.
│
├── models              - This folder contains 2 models implemented for cifar-100.
│   ├── tiny_vgg_model.py  - Contains the architecture of TinyVGG model, this model is somehow a tiny version of vgg16.
│   └── simple_model.py    - Contains the architecture of SimpleModel, the model I started with.
│
├── trainer             - This folder contains trainers used which inherit from BaseTrain.
│   ├── tiny_vgg_trainer.py - Contains the trainer class of the TinyVGG model.  
│   └── simple_trainer.py   - Contains the trainer class of the SimpleModel.
|
├── testers             - This folder contains testers used which inherit from BaseTest.
│   ├── tiny_vgg_tester.py - Contains the tester class of the TinyVGG model.  
│   └── simple_tester.py   - Contains the tester class of the SimpleModel.
| 
├──  mains              
│    └── main.py  - responsible for the whole pipeline.
|
│  
├──  data _loader  
│    ├── data_generator.py  - Contains DataGenerator class which handles Cifar-100 dataset.
│    └── preprocessing.py   - Contains helper functions for preprocessing Cifar-100 dataset.
| 
└── utils
     ├── config.py  - Contains utility functions to handle json config file.
     ├── logger.py  - Contains Logger class which handles tensorboard.
     └── utils.py   - Contains utility functions to parse arguments and handle pickle data.     
```


# Download pretrained models:
I have uploaded a pretrained TinyVGG model at [Google Drive](https://drive.google.com/open?id=1LGjmId-4rHdJIcT5Gd7ZKRuaEOHwRg6_)
**Note: edit checkpoint file because it has absolute path and I am going to fix that soon.
# Install dependencies

* Python3.x <br>

* [Tensorflow](https://www.tensorflow.org/install)

* Tensorboard[optional] <br>
* OpenCV
```
pip3 install opencv-contrib-python
```

* Numpy
```
pip3 install numpy
```

* bunch
```
pip3 install bunch
```
* tqdm
```
pip3 install tqdm
```

# Config File
In order to train, pretrain or test the model you need first to edit the config file:
```
{
  "mode":"train",                  - mode:train, test, prediction.
  "model":"TinyVGG",               - model_name to be used, leave it to TinyVGG it has the best accuracy. 
  "num_epochs": 800,               - Numer of epochs to train the model if it is in train mode.
  "learning_rate": 0.0001,         - Learning rate used for training the model.
  "batch_size": 256,               - Batch size for training, validation and testing sets(#TODO: edit single batch_size per mode)
  "val_per_epoch": 1,              - Get validation set acc and loss per val_per_epoch. (Can be ignored).
  "state_size": [32, 32, 3],       - Input shape if in train or test mode(can be ignored in predicitonn mode).
  "val_split_ratio":0.2,           - Ratio to split validation and training set.
  "max_to_keep":1,                 - Maximum number of checkpoints to keep.
  "use_val":false,                 - If set to false, the model is trained on the whole training set.
  "pretrain": true,                - Should be set to true when we pretrain the model.

  "train_data_path":"path_to_training_set",                      - Path to training data.
  "test_data_path":"path_to_test_set",                           - Path to test data.
  "meta_data_path":"path_to_dataset_meta_data",                  - Path to meta-data. 
  "checkpoint_dir":"path_to_store_the_model_checkpoints",        - Path to checkpoints store location/ or loading model.
  "summary_dir":"path_to_store_model_summaries_for_tensorboard"  - Path to summaries store location/.
}
```

# Model training
In order to train, pretrain or test the model you need first to edit the config file that is described at(#Config-File).<br>
To train a TinyVGG model:<br>
set:<br>
```
"mode":"train"
"model":"TinyVGG",
"num_epochs":200,
"learning_rate":0.0001,
"batch_size":256,
"val_split_ratio":0,
"state_size":[32, 32, 3],
"use_val":false,
"pretrain": Set it to true if you want to pretrain the model found at checkpoint_dir. else set it to false.
"train_data_path": set it to path of the training data e.g: "/content/train"
"meta_data_path": path to metadata of the training set, e.g: "/content/cifar-100-python/meta"
"checkpoint_dir": path to store checkpoints, e.g: "/content/saved_models/tiny_vgg_model/checkpoint/"
"summary_dir": path to store the model summaries for tensorboard, e.g: "/content/saved_models/tiny_vgg_model/summary/"
```
Then change directory to the project's folder and run:
python3.6 -m src.mains.main --config path_to_config_file

# Model testing
To test the model on test_set, things are the same as [model_training](#model-training) except:<br>
change the following attributes in config file:<br>
```
"mode":"test",
"test_data_path": set it to the path of test data.
```
Then change directory to the project's folder and run:<br>
python3.6 -m src.mains.main --config path_to_config_file



# Make predictions with pretrained models
To make predictions by using images of any size and any format:<br>
Set the following attributes in the config file:
```
"mode":"prediction",
"model":"TinyVGG",
"checkpoint_dir": set it to the path of the checkpoints of the TinyVgg model.
"meta_data_path": path to metadata of the training set, e.g: "/content/cifar-100-python/meta"
```
***Note: metadata is needed to print class label.
Then change directory to the project's folder and run:<br>
```python3.6 -m src.mains.main --img_path="Path to your image" --config path_to_config_file```

You can use my pretrained model see [Download pretrained models](#Download-pretrained-models) <br>
**Note: edit checkpoint file because it has absolute path and I am going to fix that soon.

#TODO:
Right now for some reasons there is nly one active model, TinyVGG model to use, I will fix the problem soon it is very simple.

# Implementation details

## TinyVGG model arch
<img src="https://github.com/MohamedAli1995/Cifar-100-Classifier/blob/master/src/models/tiny_vgg_graph.png"
     alt="Image not loaded"
     style="float: left; margin-right: 10px;" />

## TinyVGG training
 I trained the TinyVGG model by splitting training_data into train/val for 200 epoch, then train for 200 epoch using all training_data.<br>
 Acheived train accuracy of 53%

## TinyVGG testing
   Acheived testing accuracy of 52%


