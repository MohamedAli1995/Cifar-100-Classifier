# Cifar-100-Classifier
A Cifar-100 classifier with well designed architecture and good OOP design.<br>
This project follows the **best practice tensorflow folder structure of** [Tensorflow Best Practice](https://github.com/MrGemy95/Tensorflow-Project-Template) 


# Table of contents

- [Project structure](#project-structure)
- [Dependencies](#install-dependencies)




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
