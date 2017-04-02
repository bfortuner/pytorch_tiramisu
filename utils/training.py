import argparse
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
import os
import sys
import math
import time
import string
import random
import shutil
import utils.make_graph as make_graph

DATA_PATH='data/'
RESULTS_PATH='results/'
WEIGHTS_PATH='models/'

def get_rand_str(n):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

def save_weights(model, epoch, loss, err, sessionName, isBest=False):
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
    if isBest:
        shutil.copyfile(weights_fpath, WEIGHTS_PATH+'best.pth')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights from session {} (lastEpoch {}, loss {}, error {})"
          .format(weights['sessionName'], startEpoch-1, weights['loss'],
                  weights['error']))
    return startEpoch

def train(epoch, net, trainLoader, optimizer, trainF, sessionName=get_rand_str(5)):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
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
    save_weights(net, epoch, loss.data[0], err, sessionName)
    print('Epoch {:d}: Train - Loss: {:.6f}\tError: {:.6f}'.format(epoch, loss.data[0], err))

def test(epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda())
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
