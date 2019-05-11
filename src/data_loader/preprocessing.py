import numpy as np

def cifar_100_data_transformation(data):
    x = data['data'].reshape(data['data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8") / 255.0
    y = data['fine_labels']

    return x, y

def one_hot_encoding(labels, num_classes):
    one_hot_encoded = np.zeros((len(labels), num_classes))
    one_hot_encoded[range(len(labels)), labels] = 1
    return one_hot_encoded









