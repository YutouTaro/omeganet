# Copyright 2020 Fabio Tosi, Filippo Aleotti, Pierluigi Zama Ramirez, Matteo Poggi,
# Samuele Salti, Luigi Di Stefano, Stefano Mattoccia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate semantic artifacts for KITTI
"""
import os
import tensorflow as tf
import cv2
from tqdm import tqdm
from testers import general_tester
from helpers import utilities


class Tester(general_tester.GeneralTester):
    def prepare(self):
        """Create output folders
        """
        dest = os.path.join(self.params.output_path, "semantic")
        utilities.create_dir(dest)

    def test(self, network, dataloader, is_training):
        """Generate semantic artifacts.
            It saves semantic artifacts in the
            self.params.output_path/semantic folder.
            :param network: network to test
            :param dataloader: dataloader for this test
            :param is_training: training_flag for Batchnorm
        """

        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        self.prepare()
        var_list = network.get_network_params()
        saver = tf.train.Saver(var_list=var_list)

        init_op = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )
        sess.run(init_op)

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        saver.restore(sess, self.params.checkpoint_path)

        print(" [*] Load model: SUCCESS")

        prediction_semantic = tf.image.resize_images(
            network.semantic_logits, [dataloader.image_h, dataloader.image_w]
        )
        ops = [tf.argmax(prediction_semantic[0], -1)]

        with tqdm(total=self.num_test_samples) as pbar:
            for step in range(self.num_test_samples):
                outputs = sess.run(ops, feed_dict={is_training: False})
                name = self.get_file_name(step)
                semantic_map = outputs[0].squeeze()
                dest = os.path.join(self.params.output_path, "semantic", name + ".png")
                cv2.imwrite(dest, semantic_map)
                pbar.update(1)

        coordinator.request_stop()
        coordinator.join(threads)

    def get_file_name(self, step):
        """ Get name of nth line of test file
            :param step: current step
            :return name: name suited for KITTI (eg 000000_10)
        """
        name = str(step).zfill(6) + "_10"
        return name

    def train(self, network, dataloader, is_training):
        """Generate semantic network training.
            It saves semantic artifacts in the
            self.params.output_path/semantic folder.
            :param network: network to test
            :param dataloader: dataloader for this test
            :param is_training: training_flag for Batchnorm
        """

        # dimention of the training data
        # get number of samples
        num_train_samples = 0
        with open(dataloader.filenames_file) as fin:
            lines = fin.readlines()
            for filename in lines:
                filename = filename.rstrip()
                # print(filename)
                # print(os.path.join(dataloader.datapath, filename))
                if os.path.isfile(os.path.join(dataloader.datapath, filename)):
                    # print('+1')
                    num_train_samples += 1
                # break
        print("num_train_samples = {}".format(num_train_samples))

        num_features = network.classes

        #parameters
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # learning rate policy
        decay_steps = int(num_train_samples / self.params.batchSize * self.params.epochs_per_decay)
        learning_rate = tf.train.exponential_decay(
            self.params.learning_rate,
            global_step,
            decay_steps,
            self.params.learning_rate_decay_factor,
            staircase=True,
            name='exponential_decay_learning_rate'
        )

        # config = tf.ConfigProto(allow_soft_placement=True)
        # sess = tf.Session(config=config)
        #
        # self.prepare()
        # var_list = network.get_network_params()
        # saver = tf.train.Saver(var_list=var_list)
        #
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # sess.run(init_op)