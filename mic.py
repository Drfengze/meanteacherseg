# MIC model
# Author: Huaize Feng
# Date: 2023-03-21
# This is a MIC model for satellite image segmentation, 
# which is based on U-net model and semi-supervised learning.

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

