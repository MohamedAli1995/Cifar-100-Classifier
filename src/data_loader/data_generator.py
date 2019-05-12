import numpy as np
from src.utils.utils import unpickle
from src.data_loader.preprocessing import one_hot_encoding, cifar_100_data_transformation
from random import shuffle


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.all_train_data = None
        self.x_all_train = None
        self.y_all_train = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.meta = None

        self.num_batches_train = None
        self.num_batches_val = None
        self.num_batches_test = None

        self.indx_batch_train = 0
        self.indx_batch_val = 0
        self.indx_batch_test = 0

        self.__load_data()

    def __load_data(self):
        self.all_train_data = unpickle(self.config.train_data_path)
        test_data = unpickle(self.config.test_data_path)
        self.meta = unpickle(self.config.meta_data_path)


        self.x_all_train, self.y_all_train = cifar_100_data_transformation(self.all_train_data)
        self.y_all_train = one_hot_encoding(self.y_all_train, 100)

        x_test, y_test = cifar_100_data_transformation(test_data)
        y_test = one_hot_encoding(y_test, 100)

        self.x_test = x_test
        self.y_test = y_test

        self.__shuffle_all_data()
        self.__split_train_val()


        self.num_batches_train = int(np.ceil(self.x_train.shape[0] / self.config.batch_size))
        self.num_batches_val = int(np.ceil(self.x_val.shape[0] / self.config.batch_size))
        self.num_batches_test = int(np.ceil(self.x_test.shape[0] / self.config.batch_size))

    def prepare_new_epoch_data(self):
        self.indx_batch_train = 0
        self.indx_batch_val = 0
        self.indx_batch_test = 0
        self.__shuffle_train_data()

    def __shuffle_all_data(self):
        indices_list = [i for i in range(self.x_all_train.shape[0])]
        shuffle(indices_list)
        # Next two lines may cause memory error if no sufficient ram.
        self.x_all_train = self.x_all_train[indices_list]
        self.y_all_train = self.y_all_train[indices_list]

    def __split_train_val(self):
        if self.config.use_val:
            split_point = int(self.config.val_split_ratio * self.x_all_train.shape[0])
        else:
            split_point = 0
        self.x_train = self.x_all_train[split_point:self.x_all_train.shape[0]]
        self.y_train = self.y_all_train[split_point:self.y_all_train.shape[0]]
        self.x_val = self.x_all_train[0:split_point]
        self.y_val = self.y_all_train[0:split_point]

    def __shuffle_train_data(self):
        indices_list = [i for i in range(self.x_train.shape[0])]
        shuffle(indices_list)
        # Next two lines may cause memory error if no sufficient ram.
        self.x_train = self.x_train[indices_list]
        self.y_train = self.y_train[indices_list]

    def next_batch(self, batch_type="train"):
        if batch_type == "train":
            x = self.x_train[self.indx_batch_train:self.indx_batch_train + self.config.batch_size]
            y = self.y_train[self.indx_batch_train:self.indx_batch_train + self.config.batch_size]
            self.indx_batch_train = (self.indx_batch_train + self.config.batch_size) % self.x_train.shape[0]
            return x, y

        elif batch_type == "val":
            x = self.x_val[self.indx_batch_val:self.indx_batch_val + self.config.batch_size]
            y = self.y_val[self.indx_batch_val:self.indx_batch_val + self.config.batch_size]
            self.indx_batch_val = (self.indx_batch_val + self.config.batch_size) % self.x_val.shape[0]
            return x, y
        elif batch_type == "test":
            x = self.x_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            y = self.y_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            self.indx_batch_test = (self.indx_batch_test + self.config.batch_size) % self.x_test.shape[0]
            return x, y


    def get_label_name(self, one_hot_encoded_label):
        indx = np.argmax(one_hot_encoded_label)
        return self.meta['fine_label_names'][indx]

