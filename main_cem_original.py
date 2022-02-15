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

# Utils
from tqdm import tqdm

# Custom
import models.convcem as convnet
import models.lossconvcem as lossnet
from data.sampler import SubsetSequentialSampler

import neptune.new as neptune
from cem import CEMDataset

from sklearn.metrics import r2_score, mean_squared_error

# Create Neptune Run

run = neptune.init(project='wonderit/maxwellfdfd-ll4al',
                   tags=['margin0.1', 'sub20000', 're-init', 'original_convnet', 'll', '20x40', 'new-tbr'],
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2ZmY3ZjczOC0wYWM2LTQzZGItOTNkZi02Y2Y3ZjkxMDZhZTgifQ==')

# Params
PARAMS = {
    'num_train': 500,
    'num_val': 0,
    'batch_size': 128,
    'subset_size': 200,
    'k': 200,
    'margin': 0.1,
    'lpl_lambda': 1.0,
    'lpl_l1_lambda': 0,
    'trials': 3,
    'cycles': 10,
    'epoch': 200,
    'lr': 0.1,
    'milestones': [160],
    'epoch_l': 120,
    'sgd_momentum': 0.9,
    'weight_decay': 5e-4,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'T': 4,
    'is_random': False,
    're-init-backbone': True,
    're-init-module': True,
    'is_kd': True,
    'is_tbr': True,
    'tbr_lambda': 0.9,
    'is_tor': True,
    'tor_lambda': 0.9,
    'tor_zscore': 2.0,
    'server': 2,
}


run['config/hyperparameters'] = PARAMS

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
data_dir = './data'

cem_train = CEMDataset(data_dir, train=True)
cem_unlabeled = CEMDataset(data_dir, train=True)
cem_test = CEMDataset(data_dir, train=False)

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


def TeacherBoundedLoss(out_s, out_t, labels):
    l1_t = torch.abs(out_t - labels)
    l1_s = torch.abs(out_s - labels)
    l1_t_s = torch.abs(out_t - out_s)
    flag = (l1_s - l1_t) > 0
    # TBR edited
    loss = (flag * l1_t_s).mean()
    return loss


def TeacherOutlierRejection(out_s, out_t, labels):
    l1_t = torch.abs(labels - out_t)
    l1_s = torch.abs(out_s - labels)
    z_flag_1 = ((l1_t - l1_t.mean()) / l1_t.std()) > PARAMS['tor_zscore']
    z_flag_0 = ((l1_t - l1_t.mean()) / l1_t.std()) <= PARAMS['tor_zscore']
    loss = (z_flag_1 * torch.sqrt(torch.abs(out_s - out_t) + 1e-7) + z_flag_0 * l1_s).mean()
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
        teacher_model_path =f'{checkpoint_dir}/teacher_model_trial{trial}_cycle{prev_cycle}_server{PARAMS["server"]}.pth'
        models['teacher_backbone'] = convnet.ConvNet(num_classes=24)
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
            features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        # loss for multi-output
        target_loss = torch.mean(target_loss, dim=1)
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=PARAMS['margin'])
        m_module_l1_loss = nn.L1Loss()(pred_loss, target_loss)
        loss            = m_backbone_loss + PARAMS['lpl_lambda'] * m_module_loss + PARAMS['lpl_l1_lambda'] * m_module_l1_loss

        if models.get('have_teacher', False):
            teacher_outputs, teacher_feature = models['teacher_backbone'](inputs.float())
            teacher_pred_loss = models['teacher_module'](teacher_feature)
            teacher_pred_loss = teacher_pred_loss.view(teacher_pred_loss.size(0))

            if PARAMS['is_tbr']:
                tbr_loss = TeacherBoundedLoss(pred_loss, teacher_pred_loss, target_loss)
                loss = loss + PARAMS['tbr_lambda'] * tbr_loss
                run[f'train/trial{trial}/cycle{cycle}/batch/tbr_loss({PARAMS["tbr_lambda"]})'].log(tbr_loss.item())

            if PARAMS['is_tor']:
                tor_loss = TeacherOutlierRejection(pred_loss, teacher_pred_loss, target_loss)
                loss = loss + PARAMS['tor_lambda'] * tor_loss
                run[f'train/trial{trial}/cycle{cycle}/batch/tor_loss({PARAMS["tor_lambda"]})'].log(tor_loss.item())

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

        run[f'train/trial{trial}/cycle{cycle}/batch/backbone_loss'].log(m_backbone_loss.item())
        run[f'train/trial{trial}/cycle{cycle}/batch/module_loss'].log(m_module_loss.item())
        run[f'train/trial{trial}/cycle{cycle}/batch/module_l1_loss'].log(m_module_l1_loss.item())
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
        cem_convnet    = convnet.ConvNet(num_classes=24).to(PARAMS['device'])
        loss_module = lossnet.LossNet().to(PARAMS['device'])
        models      = {'backbone': cem_convnet, 'module': loss_module}

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
                cem_convnet    = convnet.ConvNet(num_classes=24).to(PARAMS['device'])
                models['backbone'] = cem_convnet

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
                    f'{checkpoint_dir}/teacher_model_trial{trial}_cycle{cycle}_server{PARAMS["server"]}.pth')


run.stop()

