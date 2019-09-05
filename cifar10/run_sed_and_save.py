import os
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import models

from se.alg import smart_net


model_names = ['VGG', 'DPN', 'DPN26', 'DPN92', 'LeNet',
               'PreActBlock', 'SENet', 'SENet18',
               'SepConv', 'PNASNet', 'PNASNetA', 'PNASNetB',
               'DenseNet', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161',
               'Inception', 'GoogLeNet',
               'ShuffleBlock', 'ShuffleNet', 'ShuffleNetG2', 'ShuffleNetG3', 'ShuffleNetV2',
               'ResNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
               'ResNeXt', 'ResNeXt29_2x64d', 'ResNeXt29_4x64d', 'ResNeXt29_8x64d', 'ResNeXt29_32x4d',
               'PreActResNet', 'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152',
               'MobileNet', 'MobileNetV2', 'EfficientNet', 'EfficientNetB0',
               'SEMaskResNet', 'SEMaskResNet18', 'SEMaskResNet34', 'SEMaskResNet50', 'SEMaskResNet101', 'SEMaskResNet152']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
""" Start """
parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet34', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: ResNet34)')
# parser.add_argument('--dest-arch', '-da', metavar='DEST-ARCH', default='SEMaskResNet34', choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names) + ' (default: ResNet34)')
parser.add_argument('--resume', '-r', type=str, help='resume from specified model')
# parser.add_argument('--dest-dir', '-dd', type=str, help='Destination directory.')
""" SED arguments """
parser.add_argument('--iternum', type=int, default=30, help='Maximum number of iterations in SED.')
parser.add_argument('--threshold', type=float, help='Threshold for element wise sparsifying.')
parser.add_argument('--threshold_decay', type=float, default=0.5, help='Decay ratio of threshold in SED')
parser.add_argument('--scale', action='store_true', help='Whether scare rows before Ce quantization.')
parser.add_argument('--tol', type=float, default=1e-10, help='Tolerance to quit the algorithm.')
parser.add_argument('--rcond', type=float, default=1e-10, help='rcond param in NumPy lsq solver.')
parser.add_argument('--threshold_row', action='store_true', help='Whether to sparsify in a row-wise way.')
parser.add_argument('--init_method', type=str, default='trivial', help='Initialization method in SED.')
""" End """
args = parser.parse_args()

assert args.resume is not None
dirname = os.path.dirname(args.resume)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
print('==> Building model..')
net = models.__dict__[args.arch]()
# net = models.__dict__[args.arch](**qargs)
# net = VGG('VGG19')
# net = ResNet34()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
# se_net = models.__dict__[args.dest_arch](threshold=args.threshold)
# se_net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    # se_net = torch.nn.DataParallel(se_net)
    cudnn.benchmark = True

print('==> Resuming from checkpoint {}..'.format(args.resume))
# assert os.path.isdir(exp_dir), 'Error: no checkpoint directory `{}` found!'.format(exp_dir)
checkpoint = torch.load(args.resume)
state_dict = checkpoint['net']
net.load_state_dict(state_dict)
best_acc = checkpoint['acc']
epoch = checkpoint['epoch']

decompose_opts = dict(
    decompose_iternum = args.iternum,
    decompose_threshold = args.threshold,
    decompose_decay = args.threshold_decay,
    decompose_scale = args.scale,
    decompose_tol = args.tol,
    decompose_rcond = args.rcond,
    threshold_row = args.threshold_row,
    init_method = args.init_method
)
""" Perfrom SED """
smart_net(net, **decompose_opts)

save_path = os.path.join(dirname, 'ckpt-sed.pth.tar')
torch.save({
    'net': net.state_dict(),
    'acc': best_acc,
    'epoch': epoch,
}, save_path)
print('Decomposed model is saved into {}'.format(save_path))

