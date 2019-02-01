import os
import sys
import math
import string
import random
import shutil

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from . import imgs as img_utils

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'


def save_weights(model, epoch, loss, err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    incorrect = preds.ne(targets).cpu().sum()
    err = incorrect/n_pixels
    return round(err,5)

def train(model, trn_loader, optimizer, criterion):
    model.train()
    trn_loss = 0
    trn_error = 0
    for idx, (inputs, targets) in enumerate(trn_loader):
        inputs = inputs.cuda(non_blocking = True)
        targets = targets.cuda(non_blocking = True)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
       
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        
        _, _, trn_acc_curr = numpy_metrics(output.data.cpu().numpy(), targets.data.cpu().numpy())
        trn_error += (1 - trn_acc_curr)

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error

def test(model, test_loader, criterion, num_classes = 11):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_error = 0
        I_tot = np.zeros(num_classes)
        U_tot = np.zeros(num_classes)
       
        for data, target in test_loader:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(data)


            loss = criterion(output, target)

            test_loss += loss

            # compute current mIOU statistic
            I, U, acc = numpy_metrics(output.cpu().numpy(), target.cpu().numpy(), n_classes=num_classes, void_labels=[num_classes])
            I_tot += I
            U_tot += U
            test_error += (1 - acc)

        test_loss /= len(test_loader)
        test_error /= len(test_loader)
        m_jacc = np.mean(I_tot / U_tot)

        return test_loss.item(), test_error, m_jacc

def numpy_metrics(y_pred, y_true, n_classes = 11, void_labels=[11]):
    """
    Similar to theano_metrics to metrics but instead y_pred and y_true are now numpy arrays
    from: https://github.com/SimJeg/FC-DenseNet/blob/master/metrics.py
    void label is 11 by default
    """

    # Put y_pred and y_true under the same shape
    y_pred = np.argmax(y_pred, axis=1)

    # We use not_void in case the prediction falls in the void class of the groundtruth
    not_void = ~ np.any([y_true == label for label in void_labels], axis=0)

    I = np.zeros(n_classes)
    U = np.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = y_true == i
        y_pred_i = y_pred == i

        I[i] = np.sum(y_true_i & y_pred_i)
        U[i] = np.sum((y_true_i | y_pred_i) & not_void)

    accuracy = np.sum(I) / np.sum(not_void)
    return I, U, accuracy

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions

def view_sample_predictions(model, loader, n):
    inputs, label = next(iter(loader))
    data = inputs.cuda()
    #label = label.cuda()
    output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)

    # normalize inputs if necessary
    img_means = inputs.mean((1,2)).unsqueeze(1).unsqueeze(2)
    img_std = torch.tensor([inputs[:,i,:,:].std() for i in range(3)]).unsqueeze(1).unsqueeze(2)

    inputs = (inputs - img_means) / img_std

    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        img_utils.view_annotated(label[i])
        img_utils.view_annotated(pred[i])

def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)
    
def masked_ce_loss(y_pred, y_true, void_class = 11., weight=None, reduce = True):
    """
    y_pred: output predictded probabilities
    Y_true: true labels
    void_class: background/void class label
    weight: if weights are use
    reduce: if reduction is appleid to final loss

    masked version of crossentropy loss
    ported over from the masked version in
    https://github.com/SimJeg/FC-DenseNet/blob/master/metrics.py
    """

    el = torch.ones_like(y_true) * void_class
    mask = torch.ne(y_true, el).long()

    y_true_tmp = y_true * mask

    loss = F.cross_entropy(y_pred, y_true_tmp, weight=weight, reduction='none')
    loss = mask.float() * loss

    if reduce:
        return loss.sum()/mask.sum()
    else:
        return loss, mask

def schedule(epoch, lr_init, epochs):
    """
    piecewise learning rate schedule
    borrowed from https://github.com/timgaripov/swa
    """
    t = (epoch) / (epochs)
    lr_ratio = 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor