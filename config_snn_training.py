'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,4"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


from config import config
conf = config.flags


#conf.train_epoch = 100
#conf.num_train_data = 10000

conf.nn_mode = 'SNN'

conf.n_reset_type = 'reset_by_sub'
#conf.n_reset_type = 'reset_to_zero'


conf.vth_rand_static = False

conf.vrest = 0.0
conf.vrest_rand_static = False

conf.adaptive_vth = False
conf.adaptive_vth_scale = 1.1

