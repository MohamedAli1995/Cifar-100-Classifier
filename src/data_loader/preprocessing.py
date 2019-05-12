import numpy as np
import cv2


def cifar_100_data_transformation(data):
    x = data['data'].reshape(data['data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8") / 255.0
    y = data['fine_labels']

    return x, y


def one_hot_encoding(labels, num_classes):
    one_hot_encoded = np.zeros((len(labels), num_classes))
    one_hot_encoded[range(len(labels)), labels] = 1
    return one_hot_encoded

def preprocess_input_image(img):
    if img.shape[0] != 32 or img.shape[1] != 32:
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)

    rgb_img =  rgb_to_bgr(img)
    img = rgb_img / 255.0
    return img


def rgb_to_bgr(img):
    """Converts RGB to BGR

    Args:
        img: input image of color bytes arrangement R->G->B.

    Returns:
        Same image with color bytes arrangement B->G->R.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
