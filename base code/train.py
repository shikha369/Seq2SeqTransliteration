# -*- coding: utf-8 -*-


import transliterate


import argparse
import numpy as np


FLAGS = argparse.ArgumentParser(
        conflict_handler='resolve',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
FLAGS.add_argument('-init', '--init',type=int,
                 help='Enter 1 for Xavier and 2 for random uniform')
                 
FLAGS.add_argument('-lr','--lr' ,type = float, help = 'Learning rate.')

FLAGS.add_argument('-batch_size',type = int,
                           help =  'Batch size to use during training.')
FLAGS.add_argument('-cell_size', type = int,help = 'Size of each model layer.')

FLAGS.add_argument('-data_dir', help = 'Data directory')

FLAGS.add_argument('-checkpoint', type = int,
                            help = 'How many training steps to do per checkpoint.')
FLAGS.add_argument('-evaluate', type = int,
                            help = 'Set to 1 for evaluating( transliterate_file )')
FLAGS.add_argument('-pat', type = int,
                            help = 'patience parameter to use during training.')
FLAGS.add_argument('-max_steps',type = int,
                            help = 'max steps to use during training.')
FLAGS.add_argument('-drop_out',type = float,
                            help = 'drop out to use during training.')
FLAGS.add_argument('-bi_lstm',type = int,
                            help = 'Set to 1 for bilstm .')
FLAGS.add_argument('-attent',type = int,
                            help = 'Set to 1 for attention .')
        
        

if __name__ == '__main__':
    args = FLAGS.parse_args()
    transliterate.main_called(args)
