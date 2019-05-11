import tensorflow as tf



class BaseTest:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.config = config
        self.sess = sess
        self.data = data
        self.logger = logger


    def test(self):
        raise NotImplemented


    def test_step(self):
        raise NotImplemented
