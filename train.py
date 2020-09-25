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
trainopt = TrainOptions()
args = trainopt.parse()

if args.gpu_ids == -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def configure_parameters():
    """Prepare configurations for Network, Dataloader and Tester
        :return network_params: configuration for Network
        :return dataloader_params: configuration for Dataloader
        :return testing_params: configuration for Tester
    """
    network_params = general_network.network_parameters(
        height=args.height,
        width=args.width,
        load_only_baseline=args.load_only_baseline,
        tau=args.tau,
    )

    dataloader_params = dataloader_parameters(
        height=args.height, width=args.width, task=args.task
    )

    testing_params = tester_factory.tester_parameters(
        output_path=args.dest,
        checkpoint_path=args.ckpt,
        width=args.width,
        height=args.height,
        filenames_file=args.filenames_file,
        datapath=args.datapath,
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
    dataloader = dataloader_factory.get_dataloader(args.task)(
        datapath=args.datapath,
        filenames_file=args.filenames_file,
        params=dataloader_params,
    )
    batch = dataloader.get_next_batch()
    network = complete_network.OmegaNet(
        batch, is_training=training_flag, params=network_params
    )

    network.build()
    return network, dataloader, training_flag

def main(_):
    """Create the Dataloader, the Network and the Tester.
        Then, run the Tester.
        :raise ValueError: if model does not exist
    """
    model_exists = utilities.check_model_exists(args.ckpt)
    if not model_exists:
        raise ValueError("Model not found")
    network_params, dataloader_params, testing_params = configure_parameters()
    print("=======dataloader_params: {}".format(dataloader_params))
    network, dataloader, training_flag = configure_network(
        network_params, dataloader_params
    )
    print("=======training_flag: {}".format(training_flag))

    # tester = tester_factory.get_tester(args.task)(testing_params)
    # tester.test(network, dataloader, training_flag)


if __name__ == "__main__":
    tf.app.run()