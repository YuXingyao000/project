# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-11-13 10:51:33
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-02 17:02:16
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9;9.0'
os.environ['FORCE_CUDA'] = '1'

setup(name='gridding',
      version='2.1.0',
      ext_modules=[
          CUDAExtension('gridding', ['gridding_cuda.cpp', 'gridding.cu', 'gridding_reverse.cu']),
      ],
      cmdclass={'build_ext': BuildExtension})
