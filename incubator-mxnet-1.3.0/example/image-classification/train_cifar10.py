# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

def download_cifar10():
    data_dir="data"
    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"))
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fnames[0])
    return fnames

def set_cifar_aug(aug):
    aug.set_defaults(rgb_mean='125.307,122.961,113.8575', rgb_std='51.5865,50.847,51.255')
    aug.set_defaults(random_mirror=1, pad=4, fill_value=0, random_crop=1)
    aug.set_defaults(min_random_size=32, max_random_size=32)

if __name__ == '__main__':
    # download data
    (train_fname, val_fname) = download_cifar10()

    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    # uncomment to set standard cifar augmentations
    # set_cifar_aug(parser)
    parser.set_defaults(
        # network
        network        = 'resnet',
        num_layers     = 110,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 10,
        num_examples  = 50000,
        image_shape    = '3,28,28',
        pad_size       = 4,
        # train
        batch_size     = 128,
        num_epochs     = 300,
        lr             = .05,
        lr_step_epochs = '200,250',
    )
    # 获得所有的参数
    # parse_args() 的返回值是一个命名空间，包含传递给命令的参数。该对象将参数保存其属性，
    # 因此如果你的参数 dest 是 "myoption"，那么你就可以args.myoption 来访问该值。
    args = parser.parse_args()

    # load network
    from importlib import import_module
    # 根据参数中的网络名称导入需要的网网络文件，这里是resnet.py，在symbols文件夹中
    net = import_module('symbols.'+args.network)
    # *args 是用来发送一个非键值对的可变数量的参数列表给一个函数.
    # **kwargs 允许你将不定长度的键值对, 作为参数传递给一个函数。 如果你想要在一个函数里处理带名字的参数, 你应该使用**kwargs。
    # vars返回args的属性和属性值的字典对象，加入**之后作为get_symbol的**kwarg参数传入
    # 根据网络的结构返回了一个resnet实例，作为symbol
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)
