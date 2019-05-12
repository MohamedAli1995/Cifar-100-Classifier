# Cifar-100-Classifier
A Cifar-100 classifier with well designed architecture and good OOP design.<br>
This project follows the **best practice tensorflow folder structure of** [Tensorflow Best Practice](https://github.com/MrGemy95/Tensorflow-Project-Template) 


# Table of contents

- [Project structure](#project-structure)
- [Dependencies](#install-dependencies)
- [Config file](#config-file)
- [Training](#training-dependencies)



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
  "checkpoint_dir":"path_to_store_the_model_checkpoints",        - Path to checkpoints store location.
  "summary_dir":"path_to_store_model_summaries_for_tensorboard"  - Path to summaries store location.
}
```
