#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 23:47:47 2022
@author: qiang
"""

import pandas as pd
from datetime import datetime
import numpy as np
import torch
import os
from glob import glob
import itertools
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse
from NN_networks import  SlipDetectGlobalGru as Net
from copy import copy
import math

def modify_robot_data(robot_data,infos, trans_crit=0.02, rot_crit=2):
    if infos['move_type'] == 'trans':
        if infos['obj_name'] == 'COFFEE':
            z_coor = np.array(robot_data)
        else:
            z_coor = np.array(robot_data)[...,2]
        z_coor_after_offset = bias_data(z_coor)
        weighted_list = []
        for idx in range(len(z_coor_after_offset)):
            weighted_list.append(abs(np.array(z_coor_after_offset[idx-50:idx+50]).mean(axis=0))/trans_crit)
    
    elif infos['move_type'] == 'rot':
        robot_vec_data = np.array(copy(robot_data))
        rotation_vecs = robot_vec_data[...,-3:]
        rotation_vecs_degree = []
        for row in rotation_vecs:
            itm = []
            for i in row:
                itm.append(math.degrees(i))
            rotation_vecs_degree.append(itm)
        rotation_vecs_degree = np.array(rotation_vecs_degree)
        angle_offset_arraylist = rotation_vecs_degree[0:500]
        angle_offset = angle_offset_arraylist.mean(axis=0)
        angle_difference = []
        for deg in rotation_vecs_degree:
            angle_difference.append(np.linalg.norm((deg-angle_offset)))
        angle_after_offset = bias_data(angle_difference)
        weighted_list = []
        for idx in range(len(angle_after_offset)):
            weighted_list.append(abs(np.array(angle_after_offset[idx-50:idx+50]).mean(axis=0))/rot_crit)
    
    elif infos['move_type'] == 'trans+rot':
        z_coor = np.array(robot_data)[...,2]
        z_coor_after_offset = bias_data(z_coor)
        weighted_list_z = []
        for idx in range(len(z_coor_after_offset)):
            weighted_list_z.append(abs(np.array(z_coor_after_offset[idx-50:idx+50]).mean(axis=0))/trans_crit)
        
        robot_vec_data = np.array(copy(robot_data))
        rotation_vecs = robot_vec_data[...,-3:]
        rotation_vecs_degree = []
        for row in rotation_vecs:
            itm = []
            for i in row:
                itm.append(math.degrees(i))
            rotation_vecs_degree.append(itm)
        rotation_vecs_degree = np.array(rotation_vecs_degree)
        angle_offset_arraylist = rotation_vecs_degree[0:500]
        angle_offset = angle_offset_arraylist.mean(axis=0)
        angle_difference = []
        for deg in rotation_vecs_degree:
            angle_difference.append(np.linalg.norm((deg-angle_offset)))
        angle_after_offset = bias_data(angle_difference)
        weighted_list_angle = []
        for idx in range(len(angle_after_offset)):
            weighted_list_angle.append(abs(np.array(angle_after_offset[idx-50:idx+50]).mean(axis=0))/rot_crit)
        
        weighted_list = []
        for idx, i in enumerate(weighted_list_angle):
            weighted_list.append(max(i, weighted_list_z[idx]))
    
    else:
        raise RuntimeError('Non support type')
    
    return weighted_list
    

def get_danger_zone_end(data, threshold, infos):
    robot_data = data['robot_data']
    t = data['raw_t']
    if infos['move_type'] == 'trans':
        if infos['obj_name'] == 'COFFEE':
            z_coor = np.array(robot_data)
    else:
        z_coor = np.array(robot_data)[...,2]
    
    if infos['move_type'] == 'trans':
        data_after_offset = bias_data(z_coor)
        for idx in range(100, len(data_after_offset)):
            k = abs(np.array(data_after_offset[idx-50:idx+50]).mean(axis=0))
            if k >= threshold[0]:
                t_zone_final = t[idx]
                break
    
    if infos['move_type'] == 'rot':
        robot_vec_data = np.array(copy(robot_data))
        rotation_vecs = robot_vec_data[...,-3:]
        rotation_vecs_degree = []
        for row in rotation_vecs:
            itm = []
            for i in row:
                itm.append(math.degrees(i))
            rotation_vecs_degree.append(itm)
        rotation_vecs_degree = np.array(rotation_vecs_degree)
        angle_offset_arraylist = rotation_vecs_degree[0:500]
        angle_offset = angle_offset_arraylist.mean(axis=0)
        angle_difference = []
        for deg in rotation_vecs_degree:
            angle_difference.append(np.linalg.norm((deg-angle_offset)))
        angle_difference_after_offset = bias_data(angle_difference)
        for idx in range(100, len(angle_difference_after_offset)):
            k = abs(np.array(angle_difference_after_offset[idx-50:idx+50]).mean(axis=0))
            if k >= threshold[1]:
                t_zone_final = t[idx]
                break
    
    t_zone_final_1 = t[-1]
    t_zone_final_2 = t[-1]
    if infos['move_type'] == 'trans+rot':
        data_after_offset = bias_data(z_coor)
        for idx in range(100, len(data_after_offset)):
            k = abs(np.array(data_after_offset[idx-50:idx+50]).mean(axis=0))
            if k >= threshold[0]:
                t_zone_final_1 = t[idx]
                break
        robot_vec_data = np.array(copy(robot_data))
        rotation_vecs = robot_vec_data[...,-3:]
        rotation_vecs_degree = []
        for row in rotation_vecs:
            itm = []
            for i in row:
                itm.append(math.degrees(i))
            rotation_vecs_degree.append(itm)
        rotation_vecs_degree = np.array(rotation_vecs_degree)
        angle_offset_arraylist = rotation_vecs_degree[0:500]
        angle_offset = angle_offset_arraylist.mean(axis=0)
        angle_difference = []
        for deg in rotation_vecs_degree:
            angle_difference.append(np.linalg.norm((deg-angle_offset)))
        angle_difference_after_offset = bias_data(angle_difference)
        for idx in range(100, len(angle_difference_after_offset)):
            k = abs(np.array(angle_difference_after_offset[idx-50:idx+50]).mean(axis=0))
            if k >= threshold[1]:
                t_zone_final_2 = t[idx]
                break
        t_zone_final = min(t_zone_final_1,t_zone_final_2)
    return t_zone_final

def get_online_data_infos(file):
    data_infos = {}
    file_items = file[1:].split('/')
    data_infos['obj_name'] = file_items[-5]
    data_infos['move_type'] = file_items[-4]
    data_infos['state'] = file_items[-3]
    data_infos['f'] = int(file_items[-2].split('-')[0][2:])
    data_infos['v'] = float(file_items[-2].split('-')[1][2:])
    data_infos['a'] = float(file_items[-2].split('-')[2][2:])
    return data_infos
    
def get_default_eval_args(model_path, number):
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gpu', default=1)
    parser.add_argument('--eval', default=1)
    parser.add_argument('--eval_all', default=0)
    parser.add_argument('--save-csv', default=0)
    parser.add_argument('--save-img', default=1)
    parser.add_argument('--save-path', default=1)
    args = parser.parse_args()
    args.model_load_path = model_path
    args.model_number = number
    args = reload_args(args)
    args.exp_name = 'eval'
    args = args_handler(args)
    return args
    
def load_model(args):
    print(f'The are/is {args.model_num} model(s) in the ensemble model')
    models = []
    for i in range(args.model_num):
        m = Net(args).to(args.device)
        m.load_state_dict(torch.load(os.path.join(args.model_load_path, f"ckpt_{args.model_num}_model_{i}.pth")))
        models.append(m)
    return models

def bias_data(data, bias_start=100, bias_end=800):
    offset_list = []
    offset_arraylist = data[bias_start:bias_end]
    for i in offset_arraylist:
        if isinstance(i, list):
            offset_list.append(i)
        else:
            offset_list.append(i.tolist())
    offset = np.array(offset_list).mean(axis=0)
    biased_data = []
    for sample in data:
        biased_data.append(sample - offset)
    return biased_data

def data_cutter(data, robot_data, seq_start=1000, offset_start=100, offset_end=1000, move_x=False):
    rob_offset_list = []
    rob_offset_arraylist = robot_data[offset_start:offset_end]
    for i in rob_offset_arraylist:
        rob_offset_list.append(i)
    offset = np.array(rob_offset_list).mean(axis=0)
    for j in range(100,len(robot_data)):
        if abs(np.array(robot_data[j-20:j+20]).mean(axis=0) - offset) >= 0.015:
            break
    if move_x:
        return data[seq_start-move_x:j-move_x]
    else:
        return data[seq_start:j]

def detected_points_handler(sequence, interval=0.25):
    if not sequence.any():
        return None
    else:
        starts = [sequence[0]]
        ends = []
        s_old = sequence[0]
        for s in sequence:
            if s - s_old >= interval:
                ends.append(s_old)
                starts.append(s)
            s_old = s
        ends.append(sequence[-1])
        
        starts_and_ends = []
        for idx, start in enumerate(starts):
            starts_and_ends.append([start, ends[idx]])

        return starts_and_ends

def reload_args(args):
    reload = np.load(f'{args.model_load_path}/reload_info.npy',
                     allow_pickle=True).item()
    args.mode = reload['mode']
    args.categories = reload['categories']
    args.model_num = reload['model_num']
    args.stack_samples = reload['stack_samples']
    args.model_size = reload['model_size']
    args.hidden_dim = reload['hidden_dim']
    return args

def args_handler(args):
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
    args.device = device

    if args.mode == 'direct':
        args.input_dim = 18 * args.stack_samples
        args.categories = 2
    else:
        raise RuntimeError('Error')

    if args.eval:
        if not args.save_path:
            args.save_path = os.path.join(
                args.model_load_path, args.exp_name + "_" + datetime.now().strftime("%m-%d-%Y %H:%M:%S"))
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            if args.eval_all:
                args.eval_items = glob('datasets/fullset/*/*')
            else:
                args.eval_items = glob('datasets/test/*/*')
        print('Exporting the model and log to the folder:', args.save_path)
    else:
        if not args.save_path:
            proj_root_path = os.path.split(os.path.realpath(__file__))[0]
            save_root = f'{proj_root_path}/save/trained_models'
            args.save_path = os.path.join(
                save_root, args.exp_name + "_" + datetime.now().strftime("%m-%d-%Y %H:%M:%S"))
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
        print('Exporting the model and log to the folder:', args.save_path)

    try:
        if args.dataset_path:
            args.trainset_path = f'{args.dataset_path}/dataset_train.npy'
            args.testset_path = f'{args.dataset_path}/dataset_test.npy'
        else:
            args.trainset_path = 'datasets/merge/dataset_train.npy'
            args.testset_path = 'datasets/merge/dataset_test.npy'
    except AttributeError:
        print('Note: you are not inputing the dataset path, using the default path or you are evaluating the model')

    return args


def save_config(args, model):
    argsDict = deepcopy(args).__dict__
    with open(f'{args.save_path}/config.txt', 'w') as f:
        f.writelines('------------------ Args start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- Args end -------------------')
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.writelines(
            '------------------ Model architecture start ------------------' + '\n')
        f.write(str(model))
        f.writelines(
            '------------------ Model architecture end ------------------' + '\n')

    reload = {'mode': args.mode,
              'exp_name': args.exp_name,
              'categories': args.categories,
              'model_num': args.model_num,
              'stack_samples': args.stack_samples,
              'model_size': args.model_size,
              'hidden_dim':args.hidden_dim
              }
    np.save(f'{args.save_path}/reload_info.npy', reload)


class loss_log():
    def __init__(self,
                 ):
        self.title = ['epoch', 'loss_train']
        self.data = []

    def update(self, log, path):
        self.data.append(log)
        dataframe = pd.DataFrame(data=self.data, columns=self.title)
        dataframe.to_csv(path, index=False, sep=',')


class slip_log():
    def __init__(self,
                 ):
        self.title = ['step',
                      'label_non_cont', 'pred_non_cont',
                      'label_cont', 'pred_cont',
                      'label_slip', 'pred_slip',
                      'label_stop', 'pred_stop']
        self.data = []

    def update(self, log):
        self.data.append(log)

    def save(self, path):
        dataframe = pd.DataFrame(data=self.data, columns=self.title)
        dataframe.to_csv(path, index=False, sep=',')

    def clear(self):
        self.data = []


def print_log(epoch, t_train, t_test, l_train, l_test):
    print('+'*40)
    print('|  ', datetime.now(), "        |")
    print('|  ', 'Epoch: {}'.format(epoch), "                          |")
    print('|  ', 'Train loss: %.6f' % l_train, "              |")
    print('|  ', 'Test loss: %.6f' % l_test, "               |")
    print('+'*40)
    print('\n')


class eval_log():
    def __init__(self,
                 ):
        self.title = ['step', 'dx', 'dy', 'dz', 'label_non_cont', 'pred_non_cont', 'label_cont',
                      'pred_cont', 'threshold_cont', 'label_slip', 'pred_slip', 'threshold_slip']
        self.data = []

    def update(self, log):
        self.data.append(log)

    def save(self, path):
        dataframe = pd.DataFrame(data=self.data, columns=self.title)
        dataframe.to_csv(path, index=False, sep=',')

    def clear(self):
        self.data = []

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
        
    plt.figure(dpi=600)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes, rotation=0)
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.savefig(save_path, bbox_inches = 'tight')