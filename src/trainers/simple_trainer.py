from src.base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class SimpleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(SimpleTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.data.num_batches_train))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        print("accuracy: ", acc)
        print("loss: ", loss)
        print(self.model.cur_epoch_tensor.eval(self.sess))
        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {
            'loss': loss,
            'acc': acc
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = self.data.next_batch(batch_type="train")
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True,
                     self.model.hold_prob: 0.5}

        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
