import glob
import os
import sys
import tensorflow as tf
from flash_attn_tf import flash_mha
import time

import numpy as np

if __name__ == '__main__':
    q = np.random.normal(size=[2, 4096, 4, 32]).astype(np.float16)
    k = np.random.normal(size=[2, 4096, 4, 32]).astype(np.float16)
    v = np.random.normal(size=[2, 4096, 4, 32]).astype(np.float16)