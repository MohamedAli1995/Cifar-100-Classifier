import tensorflow as tf
import numpy as np
from src.data_loader.data_generator import DataGenerator
from src.models.simple_model import SimpleModel
from src.trainers.simple_trainer import SimpleTrainer
from src.utils.config import processing_config
# from src.utils.utils import get_args
from src.utils.dirs import create_dirs
from src.utils.logger import Logger
from src.testers.simple_tester import SimpleTester
def main():
    # try:
    #     args = get_args()
    #     config = processing_config(args.config)
    #
    # except:
    #     print("Missing or invalid arguments")
    #     exit(0)

    config = processing_config("/media/syrix/programms/projects/Cifar-100-Classifier/configs/simple_model.json")
    create_dirs([config.summary_dir, config.checkpoint_dir])
    sess = tf.Session()

    data = DataGenerator(config)
    print(" train data size: ", data.x_train.shape[0],
          " val data size: ", data.x_val.shape[0])

    print(" test data size: ", data.x_test.shape[0])
    # batch_x, batch_y = data.next_batch(config.batch_size, batch_type="train")
    # import cv2
    # cv2.imshow(data.get_label_name(batch_y[10]), batch_x[10])
    # cv2.waitKey(0)
    # test_x, test_y = data.next_batch(config.batch_size, batch_type="test")
    #

    model = SimpleModel(config)

    logger = Logger(sess, config)

    model.load(sess)
    if config.mode == "train":
        trainer = SimpleTrainer(sess, model, data, config, logger)
        trainer.train()
    else:
        tester = SimpleTester(sess, model, data, config, logger)
        tester.test()

if __name__ == '__main__':
    main()