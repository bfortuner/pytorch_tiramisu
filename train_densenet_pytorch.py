import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math
import time
import string
import random

import shutil

import setproctitle

from densenet_pytorch import DenseNet
import make_graph

DATA_PATH='data/'
CIFAR10_PATH=DATA_PATH+'cifar10/'
RESULTS_PATH='results/'
WEIGHTS_PATH='models/'

def get_rand_str(n):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

def save_weights(model, epoch, loss, err, sessionName):
    weights_fname = sessionName+'-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch+1,
            'loss':loss,
            'error': err,
            'sessionName': sessionName,
            'state_dict': model.state_dict()
        }, weights_fpath )
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.pth')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights from session {} (lastEpoch {}, loss {}, error {})"
          .format(weights['sessionName'], startEpoch-1, weights['loss'], 
                  weights['error']))
    return startEpoch
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nClasses', type=int, default=10) #CIFAR
    parser.add_argument('--reduction', type=float, default=1.0) #no reduction
    parser.add_argument('--bottleneck', type=bool, default=False)
    parser.add_argument('--growthRate', type=int, default=12)     
    parser.add_argument('--modelDepth', type=int, default=40)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=2)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save', type=str, default=RESULTS_PATH)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--existingWeights', type=str, default=None)
    parser.add_argument('--sessionName', type=str, default=get_rand_str(5)) 
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

#    if os.path.exists(args.save):
#        shutil.rmtree(args.save)
#    os.makedirs(args.save, exist_ok=True)

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    trainLoader = DataLoader(
        dset.CIFAR10(root=CIFAR10_PATH, train=True, download=True,
                     transform=trainTransform),
        batch_size=args.batchSize, shuffle=True, **kwargs)
    testLoader = DataLoader(
        dset.CIFAR10(root=CIFAR10_PATH, train=False, download=True,
                     transform=testTransform),
        batch_size=args.batchSize, shuffle=False, **kwargs)

    net = DenseNet(growthRate=args.growthRate, depth=args.modelDepth, reduction=args.reduction,
                            bottleneck=args.bottleneck, nClasses=args.nClasses)

    if args.existingWeights:
        print ("Loading existing weights: %s" % args.existingWeights)
        startEpoch = load_weights(net, args.existingWeights)
        endEpoch = startEpoch + args.nEpochs
        print ('Resume training at epoch: {}'.format(startEpoch))
        if os.path.exists(args.save+'train.csv'): #assume test.csv exists
            print("Found existing train.csv")
            append_write = 'a' # append if already exists
        else:
            print("Creating new train.csv")
            append_write = 'w' # make a new file if not
        trainF = open(os.path.join(args.save, 'train.csv'), append_write)
        testF = open(os.path.join(args.save, 'test.csv'), append_write)
    else:
        print ("Training new model from scratch")
        startEpoch = 1
        endEpoch = args.nEpochs
        trainF = open(os.path.join(args.save, 'train.csv'), 'w')
        testF = open(os.path.join(args.save, 'test.csv'), 'w')
        
        
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)


    print("Training....")
    for epoch in range(startEpoch, endEpoch+1):
        since = time.time()
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test(args, epoch, net, testLoader, optimizer, testF)
        time_elapsed = time.time() - since  
        print('Time {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        if epoch != 1:
            os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
#        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
#            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
#            loss.data[0], err))
        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()
    save_weights(net, epoch, loss.data[0], err, args.sessionName)
    print('Epoch {:d}: Train - Loss: {:.6f}\tError: {:.6f}'.format(epoch, loss.data[0], err))

        
def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('Test - Loss: {:.4f}, Error: {}/{} ({:.0f}%)'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
