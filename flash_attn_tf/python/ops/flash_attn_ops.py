from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

flash_attn_ops_fwd = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_flash_attn_ops_fwd.so'))
flash_attn_ops_bwd = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_flash_attn_ops_bwd.so'))

flash_mha_fwd = flash_attn_ops_fwd.flash_mha_fwd
flash_mha_bwd = flash_attn_ops_bwd.flash_mha_bwd
