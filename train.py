import os
from datetime import datetime, timezone, timedelta
import tensorflow as tf
from networks import general_network, complete_network
from dataloaders.general_dataloader import dataloader_parameters
from testers import factory as tester_factory
from dataloaders import factory as dataloader_factory
from helpers import utilities

# disable future warnings and info messages for this demo
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

now = datetime.now(timezone(timedelta(hours=8)))
timestr = now.strftime("%y%m%d-%H%M%S")

from options.option_train import TrainOptions
opt = TrainOptions()
opt = opt.parse()

if opt.gpu_ids == -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def configure_parameters():
    """Prepare configurations for Network, Dataloader and Tester
        :return network_params: configuration for Network
        :return dataloader_params: configuration for Dataloader
        :return testing_params: configuration for Tester
    """
    network_params = general_network.network_parameters(
        height=opt.height,
        width=opt.width,
        load_only_baseline=opt.load_only_baseline,
        tau=opt.tau,
    )

    dataloader_params = dataloader_parameters(
        height=opt.height, width=opt.width, task=opt.task
    )

    testing_params = tester_factory.tester_parameters(
        output_path=opt.dest,
        checkpoint_path=opt.ckpt,
        width=opt.width,
        height=opt.height,
        filenames_file=opt.filenames_file,
        datapath=opt.datapath,
    )

    return network_params, dataloader_params, testing_params

def configure_network(network_params, dataloader_params):
    """Build the Dataloader, then build the Network.
        :param network_params: configuration for Network
        :param dataloader_params: configuration for Dataloader
        :return network: built Network
        :return dataloader: built Dataloader
        :return training_flag: bool placeholder. For Batchnorm

    """
    training_flag = tf.placeholder(tf.bool)
    dataloader = dataloader_factory.get_dataloader(opt.task)(
        datapath=opt.datapath,
        filenames_file=opt.filenames_file,
        params=dataloader_params,
    )
    batch = dataloader.get_next_batch()
    network = complete_network.OmegaNet(
        batch, is_training=training_flag, params=network_params
    )

    network.build()
    return network, dataloader, training_flag

def main(_):
    """Create the Dataloader, the Network and the Trainer.
        Then, run the Trainer.
        :raise ValueError: if model does not exist
    """
    model_exists = utilities.check_model_exists(opt.ckpt)
    if not model_exists:
        raise ValueError("Model not found")
    network_params, dataloader_params, testing_params = configure_parameters()
    network, dataloader, training_flag = configure_network(network_params, dataloader_params)
    print("=======training_flag: {}".format(training_flag))

if __name__ == "__main__":
    tf.app.run()