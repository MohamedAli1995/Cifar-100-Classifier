import numpy as np
from src.utils.utils import unpickle
from src.data_loader.preprocessing import one_hot_encoding, cifar_100_data_transformation
from random import shuffle




class DataGenerator:
    """DataGenerator class responsible for dealing with cifar-100 dataset.

    Attributes:
        config: Config object to store data related to training, testing and validation.
        all_train_data: Contains the whole dataset(since the dataset fits in memory).
        x_all_train: Contains  the whole input training-data.
        x_all_train: Contains  the whole target_output labels for training-data.
        x_train: Contains training set inputs.
        y_train: Contains training set target output.
        x_val: Contains validation set inputs.
        y_val: Contains validation set target output.
        meta: Contains meta-data about Cifar-100 dataset(including label names).
    """

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
        self.__load_meta_data()

        if self.config.mode == "train":
            self.__load_train_data()
        elif self.config.mode == "test":
            self.__load_test_data()

    def __load_test_data(self):
        """Private function.
        Reads cifar-100 dataset.
        Transforms cifar-100 dataset  to height x width x number of channels.
        ***Note:
            Dataset is BGR format not RGB!..

        Returns:
        """
        test_data = unpickle(self.config.test_data_path)
        x_test, y_test = cifar_100_data_transformation(test_data)
        y_test = one_hot_encoding(y_test, 100)

        self.x_test = x_test
        self.y_test = y_test
        self.num_batches_test = int(np.ceil(self.x_test.shape[0] / self.config.batch_size))



    def __load_train_data(self):
        """Private function.
        Reads cifar-100 dataset.
        Transforms cifar-100 dataset  to height x width x number of channels.
        Shuffles all data then splits the data to train nd validation sets.
        ***Note:
            Dataset is BGR format not RGB!..

        Returns:
        """
        self.all_train_data = unpickle(self.config.train_data_path)
        self.x_all_train, self.y_all_train = cifar_100_data_transformation(self.all_train_data)
        self.y_all_train = one_hot_encoding(self.y_all_train, 100)

        self.meta = unpickle(self.config.meta_data_path)

        self.__shuffle_all_data()
        self.__split_train_val()

        self.num_batches_train = int(np.ceil(self.x_train.shape[0] / self.config.batch_size))
        self.num_batches_val = int(np.ceil(self.x_val.shape[0] / self.config.batch_size))

    def __load_meta_data(self):
        """Loads metadata only in case of prediction.

        Returns:
        """
        self.meta = unpickle(self.config.meta_data_path)

    def __shuffle_all_data(self):
        """Private function.
        Shuffles the whole training set to avoid patterns recognition by the model(I liked that course:D).
        shuffle function is used instead of sklearn shuffle function in order reduce usage of
        external dependencies.
        ***Note:
            Dataset is BGR format not RGB!..

        Returns:
        """
        indices_list = [i for i in range(self.x_all_train.shape[0])]
        shuffle(indices_list)
        # Next two lines may cause memory error if no sufficient ram.
        self.x_all_train = self.x_all_train[indices_list]
        self.y_all_train = self.y_all_train[indices_list]

    def __split_train_val(self):
        """Private function.
        Splits the training set to train and validation sets using config.val_split_ratio from config file.
        ***Note:
            Dataset is BGR format not RGB!..

        Returns:
        """
        if self.config.use_val:
            split_point = int(self.config.val_split_ratio * self.x_all_train.shape[0])
        else:
            split_point = 0
        self.x_train = self.x_all_train[split_point:self.x_all_train.shape[0]]
        self.y_train = self.y_all_train[split_point:self.y_all_train.shape[0]]
        self.x_val = self.x_all_train[0:split_point]
        self.y_val = self.y_all_train[0:split_point]

    def __shuffle_train_data(self):
        """Private function.
        Shuffles the training data.
        TODO(MohamedAli1995): Remove this function and build a base class to inherit from, this is
        a short-term solution, better build a hierarchical OOP structure.
        ***Note:
            Dataset is BGR format not RGB!..

        Returns:
        """
        indices_list = [i for i in range(self.x_train.shape[0])]
        shuffle(indices_list)

        # Next two lines may cause memory error if no sufficient ram.
        self.x_train = self.x_train[indices_list]
        self.y_train = self.y_train[indices_list]

    def prepare_new_epoch_data(self):
        """Prepares the dataset for a new epoch by setting the indx of the batches to 0 and shuffling
        the training data.

        Returns:
        """
        self.indx_batch_train = 0
        self.indx_batch_val = 0
        self.indx_batch_test = 0
        self.__shuffle_train_data()

    def next_batch(self, batch_type="train"):
        """Moves the indx_batch_... pointer to the next segment of the data.

        Args:
            batch_type: the type of the batch to be returned(train, test, validation).

        Returns:
            The next batch of the data with type of batch_type.
        """
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
        else:
            x = self.x_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            y = self.y_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            self.indx_batch_test = (self.indx_batch_test + self.config.batch_size) % self.x_test.shape[0]
            return x, y

    def get_label_name(self, one_hot_encoded_label):
        """Get the label-name of the one_hot_encoded_label from meta data of the dataset.

        Args:
            one_hot_encoded_label: one hot encoded label.

        Returns:
            The name of the one_hot_encoded_label.
        """
        indx = np.argmax(one_hot_encoded_label)
        return self.meta['fine_label_names'][indx]
