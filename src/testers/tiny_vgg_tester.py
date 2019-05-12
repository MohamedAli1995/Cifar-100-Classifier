import tensorflow as tf
from src.base.base_test import BaseTest
from tqdm import tqdm
import numpy as np
import cv2
from src.data_loader.preprocessing import preprocess_input_image
class TinyVGGTester(BaseTest):
    def __init__(self, sess, model, data, config, logger):
        super().__init__(sess, model, data, config, logger)


    def test(self):
        loop = tqdm(range(self.data.num_batches_test))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.test_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        print("test_accuracy: ",
              acc * 100, "% train_loss: ", loss)

    def test_step(self):
        batch_x, batch_y = self.data.next_batch(batch_type="test")
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: False,
                     self.model.hold_prob: 1.0}

        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def predict_image(self, img_path):
        """Predicts the class of an input image.

        Args:
            img_path:

        Returns:
            prediction of the input image.
        """
        img = cv2.imread(img_path)
        img = preprocess_input_image(img)
        img = np.asarray(img)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        feed_dict = {self.model.x: img, self.model.is_training: False,
                     self.model.hold_prob: 1.0}
        pred = self.sess.run(self.model.pred, feed_dict=feed_dict)
        return pred[0]