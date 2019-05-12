# Cifar-100-Classifier
A Cifar-100 classifier with well designed architecture and good OOP design.<br>
This project follows the **best practice tensorflow folder structure of** [Tensorflow Best Practice](https://github.com/moemen95) 


# Table of contents

- [Project structure](#project-structure)





# Project structure
Folder structure
--------------

```
├──  base
│   ├── base_model.py   - this file contains the abstract class of all models used.
│   ├── base_train.py   - this file contains the abstract class of the trainer of all models used.
│   └── base_test.py    - this file contains the abstract class of the testers of all models used.
│
├── models              - this folder contains 2 models implemented for cifar-100.
│   ├── tiny_vgg_model.py  - Contains the architecture of TinyVGG model, this model is somehow a tiny version of vgg16.
│   └── simple_model.py    - Contains the architecture of SimpleModel, the model I started with.
│
├── trainer             - this folder contains trainers used which inherit from BaseTrain.
│   ├── tiny_vgg_trainer.py - Contains the trainer class of the TinyVGG model.  
│   └── simple_trainer.py   - Contains the trainer class of the SimpleModel.
| 
├──  mains              - here's the main(s) of your project (you may need more than one main).
│    └── example_main.py  - here's an example of main that is responsible for the whole pipeline.
|
│  
├──  data _loader  
│    └── data_generator.py  - here's the data_generator that is responsible for all data handling.
│ 
└── utils
     ├── logger.py
     └── any_other_utils_you_need
