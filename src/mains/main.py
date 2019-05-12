import tensorflow as tf
import numpy as np
from src.data_loader.data_generator import DataGenerator
from src.models.simple_model import SimpleModel
from src.trainers.simple_trainer import SimpleTrainer
from src.utils.config import processing_config
from src.utils.utils import get_args
from src.utils.dirs import create_dirs
from src.utils.logger import Logger
from src.testers.simple_tester import SimpleTester
from src.trainers.tiny_vgg_trainer import TinyVGGTrainer
from src.models.tiny_vgg_model import TinyVGG
from src.testers.tiny_vgg_tester import TinyVGGTester


def main():
    try:
        args = get_args()
        print(args.config)
        config = processing_config(args.config)
    except:
        print("Missing or invalid arguments")
        exit(0)


    sess = tf.Session()
    model = TinyVGG(config)
    model.load(sess)
    data = DataGenerator(config)

    if config.mode == "prediction":
        tester = TinyVGGTester(sess, model, None, config, None)
        prediction = tester.predict_image(args.img_path)
        print("the input image is of class: ", data.get_label_name(prediction))
        return

    create_dirs([config.summary_dir, config.checkpoint_dir])




    logger = Logger(sess, config)

    if config.mode == "train":
        print(" train data size: ", data.x_train.shape[0],
              " val data size: ", data.x_val.shape[0])
        trainer = TinyVGGTrainer(sess, model, data, config, logger)
        trainer.train()
    elif config.mode == "test":
        print(" test data size: ", data.x_test.shape[0])
        tester = TinyVGGTester(sess, model, data, config, logger)
        tester.test()
    else:
        print(" Mode: ", config.mode," is not supported")

if __name__ == '__main__':
    main()
