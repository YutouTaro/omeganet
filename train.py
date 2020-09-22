import os
from datetime import datetime, timezone, timedelta
import tensorflow as tf
import numpy as np
# import sys
# import select
# from IPython import embed
# from tensorflow.python.client import timeline

# import imagenet_input as data_input
# import resnet
import argparse

now = datetime.now(timezone(timedelta(hours=8)))
timestr = now.strftime("%y%m%d-%H%M%S")

from options.option_train import TrainOptions
opt = TrainOptions()
opt = opt.parse()