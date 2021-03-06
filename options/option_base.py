import argparse
import os
from datetime import datetime, timezone, timedelta

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--task',           type=str, default='depth', help='task to test', choices=['depth', 'semantic', 'flow', 'mask'])
        self.parser.add_argument('--datapath',       type=str, required=True, help='path to image folder')
        self.parser.add_argument('--ckpt',           type=str, required=True, help='path to checkpoint, folder of model weights')
        # self.parser.add_argument('--filelist_test' , type=str, help='path to file list of test  image names') # TODO default
        self.parser.add_argument('--dest',           type=str, help='path to the folder for result output') # TODO default
        self.parser.add_argument('--batchSize',      type=int, default=75, help='input batch size')
        self.parser.add_argument("--height",         type=int, help="height of resized image", default=192)
        self.parser.add_argument("--width",          type=int, help="width of resized image", default=640)
        self.parser.add_argument('--input_nc',       type=int, default=3, help='number of input image channels')
        # self.parser.add_argument('--output_nc', type=int, default=7, help='# of output image channels')
        self.parser.add_argument("--load_only_baseline", action="store_true", help="if set, load only Baseline (CameraNet+DSNet). Otherwise, full OmegaNet will be loaded")
        self.parser.add_argument("--tau",            type=float, help="tau threshold in the paper. For motion segmentation at testing time", default=0.5)
        # self.parser.add_argument('--lstm_hidden_size', type=int, default=256, help='hidden size of the LSTM layer in PoseLSTM')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name',    type=str, default='',  help='name of the experiment, it will be used to name the saving folder')
        # self.parser.add_argument('--dataset_mode', type=str, default='unaligned_posenet', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        # self.parser.add_argument('--model', type=str, default='posenet', help='chooses which model to use. [posenet | poselstm]')
        # self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        # self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        # self.parser.add_argument('--display_winsize', type=int, default=224,  help='display window size')
        # self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        # self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        # self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        # self.parser.add_argument('--resize_or_crop', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        # self.parser.add_argument('--no_flip', action='store_true', default=True, help='if specified, do not flip the images for data augmentation')
        # self.parser.add_argument('--seed', type=int, default=0, help='initial random seed for deterministic results')
        # self.parser.add_argument('--beta', type=float, default=500, help='beta factor used in posenet.')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        self.opt.phase = self.phase

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        ## set gpu ids
        # if len(self.opt.gpu_ids) > 0:
            ## torch.cuda.set_device(self.opt.gpu_ids[0])
            #TODO change to tensorflow
        now = datetime.now(timezone(timedelta(hours=8)))
        timestr = now.strftime("%y%m%d-%H%M%S")
        if len(self.opt.name) == 0:
            self.opt.name = timestr
        else:
            self.opt.name += '-' + timestr
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.dest, self.opt.name)
        os.makedirs(expr_dir)
        # util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_'+self.opt.phase+'.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
