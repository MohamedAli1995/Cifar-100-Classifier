import tensorflow as tf



class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.config = config
        self.sess = sess
        self.data = data
        self.logger = logger
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)


    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.data.prepare_new_epoch_data()
            self.train_epoch()
            if self.config.use_val and (cur_epoch % self.config.val_per_epoch == 0 or cur_epoch == self.config.num_epochs):
                self.validate_epoch()

            self.sess.run(self.model.increment_cur_epoch_tensor)



    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplemented

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """

        raise NotImplementedError

    def validate_epoch(self):
        raise NotImplemented

    def validate_step(self):
        raise NotImplemented