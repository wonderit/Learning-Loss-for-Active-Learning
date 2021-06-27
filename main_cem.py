#!/usr/bin/env python
# coding: utf-8

# Active Learning Procedure in PyTorch.
#
# Reference:
# [Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
# '''

# Install Neptune

# In[1]:


#!pip install -q neptune-client==0.9.9 numpy==1.19.2 torch==1.8.1 torchvision==0.9.1 folium==0.2.1


# Install libraries

# In[2]:


# Python
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
import torch.nn.functional as F

# Utils
from tqdm import tqdm

# Custom
import models.resnetcem as resnet
import models.lossnetcem as lossnet
from data.sampler import SubsetSequentialSampler

import neptune.new as neptune
from cem import CEMDataset

from sklearn.metrics import r2_score, mean_squared_error

# Create Neptune Run

# In[3]:


run = neptune.init(project='wonderit/maxwellfdfd-ll4al',
                   tags='cem',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2ZmY3ZjczOC0wYWM2LTQzZGItOTNkZi02Y2Y3ZjkxMDZhZTgifQ==')


# In[4]:


# Params
PARAMS = {
    'num_train': 27000,
    'num_val': 0,
    'batch_size': 128,
    'subset_size': 10000,
    'k': 200,
    'margin': 0.01,
    'lpl_lambda': 1.0,
    'trials': 3,
    'cycles': 10,
    'epoch': 200,
    'lr': 0.1,
    'milestones': [160],
    'epoch_l': 120,
    'sgd_momentum': 0.9,
    'weight_decay': 5e-4,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'kd_type': 'soft_target',
    'is_kd': False,
    'T': 4,
    'kd_lambda': 0.1,
    'is_ua': False,
    'ua_lambda': 0.1,
    'ua_activation': 'sigmoid', # linear, relu
    'beta': 1.0,
    'ua_type': 'prediction_loss',  # prediction_loss, cross_entropy
    're-init-backbone': False,
    're-init-module': False,
    'is_tbr': False,
    'tbr_lambda': 0.5,
    'is_random': False,
    'is_tor': False,
    'tor_lambda': 0.1,
    'tor_zscore': 2.0
}


# In[5]:


run['config/hyperparameters'] = PARAMS


# In[ ]:





# In[6]:


# Seed
random_seed = 425
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

##
# Data
data_dir = './maxwellfdfd'

cem_train = CEMDataset('./maxwellfdfd', train=True)
cem_unlabeled = CEMDataset('./maxwellfdfd', train=True)
cem_test = CEMDataset('./maxwellfdfd', train=False)

dataset_size = {'train': len(cem_train), 'test': len(cem_test)}

checkpoint_dir = os.path.join('./cem', 'train', 'weights')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# In[7]:


run["config/dataset/path"] = data_dir
run["config/dataset/size"] = dataset_size


# In[ ]:


##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


def SoftTarget(out_s, out_t):
    loss = F.kl_div(F.log_softmax(out_s/PARAMS['T'], dim=1),
                    F.softmax(out_t/PARAMS['T'], dim=1)
                    ,reduction='batchmean') * PARAMS['T'] * PARAMS['T']
    return loss

def UncertaintyAttentionLoss(t_output, t_pred_loss):

    if PARAMS['ua_type'] == 'prediction_loss':
        t_ua = t_pred_loss.mean()
    else:
        teacher_probs = F.softmax(t_output)
        t_ua = torch.distributions.Categorical(teacher_probs).entropy().mean()

    if PARAMS['ua_activation'] == 'sigmoid':
        t_ua_norm = F.sigmoid(PARAMS['beta'] * t_ua)
    elif PARAMS['ua_activation'] == 'relu':
        t_ua_norm = F.relu(t_ua)
    elif PARAMS['ua_activation'] == 'tanh':
        t_ua_norm = F.tanh(t_ua)

    return t_ua_norm, t_ua


def ua_loss(outputs, labels, t_ua, beta, ua_lambda):
    ua_attention = 1/(1 + torch.exp(- beta * t_ua))
    log_softmax = torch.nn.LogSoftmax(dim=1)
    x_log = log_softmax(outputs)
    loss = (-x_log[range(labels.shape[0]), labels] * (1. + ua_lambda * ua_attention)).mean()
    return loss


def TeacherBoundedLoss(out_s, out_t, labels):
    mse_t = torch.abs(out_t - labels)
    mse_s = torch.abs(out_s - labels)
    flag = (mse_s - mse_t) > 0
    loss = (flag * mse_s).mean()
    return loss


def TeacherOutlierRejection(out_s, out_t, labels):
    mse_t = torch.abs(labels - out_t)
    z_flag_1 = ((mse_t - mse_t.mean()) / mse_t.std()) > PARAMS['tor_zscore']
    z_flag_0 = ((mse_t - mse_t.mean()) / mse_t.std()) <= PARAMS['tor_zscore']
    loss = (z_flag_1 * torch.sqrt(torch.abs(out_s - out_t) + 1e-7) + z_flag_0 * (out_s - labels) ** 2).mean()
    return loss

##
# Train Utils
iters = 0

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, cycle, trial):
    models['backbone'].train()
    models['module'].train()

    # load teacher model
    if PARAMS['is_kd'] and cycle > 0:
        prev_cycle = cycle - 1
        teacher_model_path =f'{checkpoint_dir}/teacher_model_trial{trial}_cycle{prev_cycle}.pth'
        models['teacher_backbone'] = resnet.ResNet18(num_classes=24)
        checkpoint = torch.load(teacher_model_path)
        models['teacher_backbone'].load_state_dict(checkpoint['state_dict_backbone'])
        models['teacher_backbone'].to(PARAMS['device'])
        models['teacher_backbone'].eval()
        models['teacher_backbone'].train(mode=False)
        models['have_teacher'] = True
        models['teacher_module'] = lossnet.LossNet()
        models['teacher_module'].load_state_dict(checkpoint['state_dict_module'])
        models['teacher_module'].to(PARAMS['device'])
        models['teacher_module'].eval()
        models['teacher_module'].train(mode=False)

    global iters
    m_backbone_loss = 0
    count = 0
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].to(PARAMS['device'])
        labels = data[1].to(PARAMS['device'])
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs.float())
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            # features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        # loss for multi-output
        target_loss = torch.mean(target_loss, dim=1)
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=PARAMS['margin'])
        loss            = m_backbone_loss + PARAMS['lpl_lambda'] * m_module_loss

        if models.get('have_teacher', False):
            teacher_outputs, teacher_feature = models['teacher_backbone'](inputs.float())
            teacher_pred_loss = models['teacher_module'](teacher_feature)
            teacher_pred_loss = teacher_pred_loss.view(teacher_pred_loss.size(0))

            # kd_loss = SoftTarget(scores, teacher_outputs)
            # run[f'train/trial{trial}/cycle{cycle}/batch/kd_loss({PARAMS["kd_type"]})'].log(kd_loss.item())
            # loss = loss + PARAMS['kd_lambda'] * kd_loss

            if PARAMS['is_tbr']:
                tbr_loss = TeacherBoundedLoss(pred_loss, teacher_pred_loss, target_loss)
                loss = loss + PARAMS['tbr_lambda'] * tbr_loss
                run[f'train/trial{trial}/cycle{cycle}/batch/tbr_loss({PARAMS["tbr_lambda"]})'].log(tbr_loss.item())

            if PARAMS['is_tor']:
                tor_loss = TeacherOutlierRejection(pred_loss, teacher_pred_loss, target_loss)
                loss = loss + PARAMS['tor_lambda'] * tor_loss
                run[f'train/trial{trial}/cycle{cycle}/batch/tor_loss({PARAMS["tor_lambda"]})'].log(tor_loss.item())

            if PARAMS['is_ua']:
                teacher_uncertainty_normalized, teacher_uncertainty = UncertaintyAttentionLoss(teacher_outputs, teacher_pred_loss)
                loss = loss + PARAMS['ua_lambda'] * PARAMS['kd_lambda'] * kd_loss * teacher_uncertainty_normalized
                run[f'train/trial{trial}/cycle{cycle}/batch/t_ua_{PARAMS["ua_type"]}'].log(teacher_uncertainty)
                run[f'train/trial{trial}/cycle{cycle}/batch/t_ua_norm({PARAMS["ua_type"]}-b{PARAMS["beta"]})'].log(teacher_uncertainty_normalized)

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

        run[f'train/trial{trial}/cycle{cycle}/batch/backbone_loss'].log(m_backbone_loss.item())
        run[f'train/trial{trial}/cycle{cycle}/batch/module_loss'].log(m_module_loss.item())
        run[f'train/trial{trial}/cycle{cycle}/batch/total_loss'].log(loss.item())


#
def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    test_r2 = 0
    test_rmse = 0
    with torch.no_grad():
        total = 0
        pred_array = []
        labels_array = []
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.to(PARAMS['device'])
            labels = labels.to(PARAMS['device'])
            scores, _ = models['backbone'](inputs.float())

            pred_array.extend(scores.cpu().numpy().reshape(-1))
            labels_array.extend(labels.cpu().numpy().reshape(-1))
            total += labels.size(0)

        pred_array = np.array(pred_array)
        labels_array = np.array(labels_array)

        pred_array = pred_array.reshape(-1)
        labels_array = labels_array.reshape(-1)

        test_rmse = np.sqrt(mean_squared_error(labels_array, pred_array))
        test_r2 = r2_score(y_true=labels_array, y_pred=pred_array, multioutput='uniform_average')


    return test_rmse, test_r2

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, cycle_number, trial_number):
    print('>> Train a Model.')
    best_acc = 0.

    for epoch in range(num_epochs):

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, cycle_number, trial_number)

        schedulers['backbone'].step()
        schedulers['module'].step()

        # # Save a checkpoint
        # if PARAMS['is_kd'] and epoch % 5 == 4:
        #     acc = test(models, dataloaders, 'test')
        #     if best_acc < acc:
        #         best_acc = acc
        #         torch.save({
        #             'epoch': epoch + 1,
        #             'state_dict_backbone': models['backbone'].state_dict(),
        #             'state_dict_module': models['module'].state_dict()
        #         },
        #         f'{checkpoint_dir}/teacher_model_cycle{cycle_number}.pth')
        #     print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))

    print('>> Finished.')

#
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).to(PARAMS['device'])

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.to(PARAMS['device'])
            # labels = labels.to(device)

            scores, features = models['backbone'](inputs.float())
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu()

vis = None
##
# Main
if __name__ == '__main__':
    # vis = visdom.Visdom(server='http://localhost', port=9000)
    # plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

    for trial in range(PARAMS['trials']):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        indices = list(range(PARAMS['num_train']))
        random.shuffle(indices)
        labeled_set = indices[:PARAMS['k']]
        unlabeled_set = indices[PARAMS['k']:]

        train_loader = DataLoader(cem_train, batch_size=PARAMS['batch_size'],
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader  = DataLoader(cem_test, batch_size=PARAMS['batch_size'])
        dataloaders  = {'train': train_loader, 'test': test_loader}

        # Model
        resnet18    = resnet.ResNet18(num_classes=24).to(PARAMS['device'])
        loss_module = lossnet.LossNet().to(PARAMS['device'])
        models      = {'backbone': resnet18, 'module': loss_module}

        # Add Teacher for KD
        if PARAMS['is_kd']:
            models['teacher_backbone'] = None
            models['teacher_module'] = None
            models['have_teacher'] = False

        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(PARAMS['cycles']):
            # Re init model
            if PARAMS['re-init-backbone'] and cycle > 0:
                resnet18    = resnet.ResNet18(num_classes=24).to(PARAMS['device'])
                models['backbone'] = resnet18

            if PARAMS['re-init-module'] and cycle > 0:
                loss_module = lossnet.LossNet().to(PARAMS['device'])
                models['module'] = loss_module

            # Loss, criterion and scheduler (re)initialization
            # criterion      = nn.CrossEntropyLoss(reduction='none')
            criterion = nn.L1Loss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=PARAMS['lr'],
                                    momentum=PARAMS['sgd_momentum'], weight_decay=PARAMS['weight_decay'])
            optim_module   = optim.SGD(models['module'].parameters(), lr=PARAMS['lr'],
                                    momentum=PARAMS['sgd_momentum'], weight_decay=PARAMS['weight_decay'])
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=PARAMS['milestones'])
            sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=PARAMS['milestones'])

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, PARAMS['epoch'], PARAMS['epoch_l'], cycle, trial)
            rmse, r2 = test(models, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test rmse {}, r2 {}'.format(trial+1, PARAMS['trials'], cycle+1, PARAMS['cycles'], len(labeled_set), rmse, r2))

            # Log acc
            run[f'test/trial_{trial}/rmse'].log(rmse)
            run[f'test/trial_{trial}/r2'].log(r2)
            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:PARAMS['subset_size']]

            if PARAMS['is_random']:
                # Update Random
                labeled_set += list(torch.tensor(subset)[-PARAMS['k']:].numpy())
                unlabeled_set = list(torch.tensor(subset)[:-PARAMS['k']].numpy()) + unlabeled_set[
                                                                                    PARAMS['subset_size']:]
            else:
                # Create unlabeled dataloader for the unlabeled subset
                unlabeled_loader = DataLoader(cem_unlabeled, batch_size=PARAMS['batch_size'],
                                              sampler=SubsetSequentialSampler(subset),
                                              # more convenient if we maintain the order of subset
                                              pin_memory=True)

                # Measure uncertainty of each data points in the subset
                uncertainty = get_uncertainty(models, unlabeled_loader)

                # Index in ascending order
                arg = np.argsort(uncertainty)

                # Update the labeled dataset and the unlabeled dataset, respectively
                labeled_set += list(torch.tensor(subset)[arg][-PARAMS['k']:].numpy())
                unlabeled_set = list(torch.tensor(subset)[arg][:-PARAMS['k']].numpy()) + unlabeled_set[
                                                                                         PARAMS['subset_size']:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(cem_train, batch_size=PARAMS['batch_size'],
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)
            # Save a checkpoint
            torch.save({
                        'trial': trial + 1,
                        'state_dict_backbone': models['backbone'].state_dict(),
                        'state_dict_module': models['module'].state_dict()
                    },
                    f'{checkpoint_dir}/teacher_model_trial{trial}_cycle{cycle}.pth')


# In[ ]:


run.stop()

