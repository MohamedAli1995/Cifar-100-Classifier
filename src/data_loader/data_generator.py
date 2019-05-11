import numpy as np

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.input = np.ones((500, 32, 32, 3))
        self.y = np.ones((500, 100))


    def next_batch(self, batch_size):
        indx = np.random.choice(500, batch_size)
        yield self.input[indx], self.y[indx]