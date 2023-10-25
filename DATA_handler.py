#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 22:44:04 2022

@author: qiang
"""
import pandas as pd
import random
import math
from copy import copy
import constants as CONSTANTS
import numpy as np
import scipy.signal as signal


def drop_samples(df, labels, drop_rate):
    assert df.shape[0] == labels.shape[0], "Error"
    drop_idx = []
    for i in range(labels.shape[0]):
        if np.random.uniform(0,1) <= drop_rate:
            drop_idx.append(i)
    cutted_df = df.drop(drop_idx, axis=0)
    cutted_labels = np.delete(labels, drop_idx, axis=0)
    assert cutted_df.shape[0] == cutted_labels.shape[0], "Error"
    return cutted_df, cutted_labels


def reshape_sequence_raw(file_path):
    data = pd.read_csv(file_path, header=None)
    df = pd.DataFrame(data)
    df.columns = CONSTANTS.headers
    if file_path[-8:-4] == 'stop':
        v = float(file_path.split("-")[1][2:])
        if v in CONSTANTS.stop_v_2100:
            stop_start = 2100
            stop_end = 2100 + CONSTANTS.stop_len
        else:
            stop_start = 2050
            stop_end = 2050 + CONSTANTS.stop_len
        slip_start_points = []
        for pillar_idx in range(0, 9):
            try:
                slip_start_points.append(
                    copy(df[CONSTANTS.pillars[pillar_idx][1]]).tolist().index(1))
                # print(
                #     f"Slippage happened in the stop case: {file_path} - pillar{pillar_idx}")
                pass
            except ValueError:
                pass
        cutted_df = df[:stop_end]
        labels = np.zeros(stop_end)
        if slip_start_points:
            incipient_slip_start = min(slip_start_points)
            labels[incipient_slip_start:stop_start] = 1
        labels[stop_start:] = 0
    else:
        slip_start_points = []
        for pillar_idx in range(0, 9):
            try:
                slip_start_points.append(
                    copy(df[CONSTANTS.pillars[pillar_idx][1]]).tolist().index(1))
            except ValueError:
                print(
                    f"No slippage in the slip case: {file_path} - pillar{pillar_idx}")
                pass
        incipient_slip_start = min(slip_start_points)
        incipient_slip_end = max(slip_start_points)
        
        cutted_df = df[:4000]
        labels = np.zeros(cutted_df.shape[0])
        labels[incipient_slip_start:incipient_slip_end] = 1
        labels[incipient_slip_end:] = 0
        
    cutted_df = cutted_df[CONSTANTS.drop_begin:]
    cutted_df = median_filter_and_velocity(cutted_df)
    assert cutted_df.shape[0] == labels.shape[0], "Error"
    return cutted_df, labels


def reshape_sequence(df, stop, drop_sample=True):
    if stop != None:
        slip_start_points = []
        for pillar_idx in range(1, 9):
            try:
                slip_start_points.append(
                    copy(df[CONSTANTS.pillars[pillar_idx][1]]).tolist().index(1))
                # print("Slippage happened in the stop case")
            except ValueError:
                pass
        stop_start = stop
        stop_end = stop + CONSTANTS.stop_len
        cutted_df = df[:stop_end]
        labels = np.zeros(stop_end)
        if slip_start_points:
            incipient_slip_start = min(slip_start_points)
            labels[incipient_slip_start:stop_start] = 1
        labels[stop_start:] = 0
    else:
        slip_start_points = []
        for pillar_idx in range(1, 9):
            try:
                slip_start_points.append(
                    copy(df[CONSTANTS.pillars[pillar_idx][1]]).tolist().index(1))
            except ValueError:
                # print("No slippage in the slip case")
                pass
        incipient_slip_start = min(slip_start_points)
        incipient_slip_end = max(slip_start_points)
        
        cutted_df = df[:4000]
        labels = np.zeros(cutted_df.shape[0])
        labels[incipient_slip_start:incipient_slip_end] = 1
        labels[incipient_slip_end:] = 0
        
    cutted_df = cutted_df[CONSTANTS.drop_begin:]
    cutted_df = median_filter_and_velocity(cutted_df)
    assert cutted_df.shape[0] == labels.shape[0], "Error"
    return cutted_df, labels

def sample_scaler(samples, scale):
    samples_temp = []
    for sample in samples:
        FX = copy(sample['FX']) * scale
        FY = copy(sample['FY']) * scale
        SLIP = copy(sample['SLIP'])
        samples_temp.append({"FX":FX,
                             "FY":FY,
                             "SLIP":SLIP})
    return samples_temp


class pillar_data_sampler():
    def __init__(
        self,
        train_set,
        test_set,
        pillar_num,
        zero_pillar_ratio=0.1,
        slip_scale = [1, 1],
        stop_scale = [1, 1],
        max_zero_num=3,
        gussian_loc=0.0,
        gussian_scale=0.001,
        slip_ratio = None,
    ):
        self.train_set = train_set
        self.test_set = test_set
        self.zero_pillar_ratio = zero_pillar_ratio
        self.max_zero_num = max_zero_num
        self.gussian_loc = gussian_loc
        self.gussian_scale = gussian_scale
        self.pillar_num = pillar_num
        self.slip_scale = slip_scale
        self.stop_scale = stop_scale
        
        total_slip_num = len(self.train_set['slip_data']) + len(self.test_set['slip_data'])
        total_stop_2100_num =  len(self.train_set['stop_data_2100']) + len(self.test_set['stop_data_2100'])
        total_stop_2050_num =  len(self.train_set['stop_data_2050']) + len(self.test_set['stop_data_2050'])
        total_stop_num = total_stop_2100_num + total_stop_2050_num
        
        if not slip_ratio:
            self.slip_ratio = total_slip_num / (total_slip_num + total_stop_num)
        else:
            self.slip_ratio = slip_ratio
        
        self.stop_2100_ratio = total_stop_2100_num / (total_stop_2100_num + total_stop_2050_num)
        
        print(f'total slip num: {total_slip_num}')
        print(f'total stop 2100 num: {total_stop_2100_num}')
        print(f'total stop 2050 num: {total_stop_2050_num}')
        
        self.sampled_slip_num = 0
        self.sampled_stop_num = 0
    
    def sample(self, mode, in_raw_dataset_style=True):
        zero_pillar_amount = 0
        if mode == 'train':
            if np.random.uniform(0,1) <= self.slip_ratio:
                self.sampled_slip_num += 1
                samples = copy(random.sample(self.train_set['slip_data'], self.pillar_num))
                samples = sample_scaler(samples, np.random.uniform(self.slip_scale[0],self.slip_scale[1]))
                stop = None
            else:
                if np.random.uniform(0,1) <= self.stop_2100_ratio:
                    stop = 2100
                    self.sampled_stop_num += 1
                    samples = copy(random.sample(self.train_set['stop_data_2100'], self.pillar_num))
                    samples = sample_scaler(samples, np.random.uniform(self.stop_scale[0],self.stop_scale[1]))
                else:
                    stop = 2050
                    self.sampled_stop_num += 1
                    samples = copy(random.sample(self.train_set['stop_data_2050'], self.pillar_num))
                    samples = sample_scaler(samples, np.random.uniform(self.stop_scale[0],self.stop_scale[1]))
        elif mode == 'test':
            if np.random.uniform(0,1) <= self.slip_ratio:
                self.sampled_slip_num += 1
                samples = copy(random.sample(self.test_set['slip_data'], self.pillar_num))
                samples = sample_scaler(samples, np.random.uniform(self.slip_scale[0],self.slip_scale[1]))
                stop = None
            else:
                if np.random.uniform(0,1) <= self.stop_2100_ratio:
                    stop = 2100
                    self.sampled_stop_num += 1
                    samples = copy(random.sample(self.test_set['stop_data_2100'], self.pillar_num))
                    samples = sample_scaler(samples, np.random.uniform(self.stop_scale[0],self.stop_scale[1]))
                else:
                    stop = 2050
                    self.sampled_stop_num += 1
                    samples = copy(random.sample(self.test_set['stop_data_2050'], self.pillar_num))
                    samples = sample_scaler(samples, np.random.uniform(self.stop_scale[0],self.stop_scale[1]))
        
        if np.random.uniform(0,1) <= self.zero_pillar_ratio:
            zero_pillar_amount = random.randint(1, 3)
            # print(f'{zero_pillar_amount} num of pillars are set to 0')
            zero_sample_random_nums = random.sample(range(9), zero_pillar_amount)
            for zero_pillar_num in zero_sample_random_nums:
                samples[zero_pillar_num]['FX'] = np.random.normal(loc=self.gussian_loc, 
                                                                  scale=self.gussian_scale,
                                                                  size=samples[zero_pillar_num]['FX'].shape)
                samples[zero_pillar_num]['FY'] = np.random.normal(loc=self.gussian_loc, 
                                                                  scale=self.gussian_scale,
                                                                  size=samples[zero_pillar_num]['FY'].shape)
                samples[zero_pillar_num]['SLIP'] = np.zeros(shape=samples[zero_pillar_num]['SLIP'].shape)

        if not in_raw_dataset_style:
            return samples, stop, zero_pillar_amount
        else:
            re_samples = {}
            for i, sample in enumerate(samples):
                re_samples[f'S0_P{i}_slip'] = sample['SLIP']
                re_samples[f'S0_P{i}_FX'] = sample['FX']
                re_samples[f'S0_P{i}_FY'] = sample['FY']
                # print(sample['SLIP'].shape)
                # print(sample['FX'].shape)
                # print(sample['FY'].shape)
            df = pd.DataFrame.from_dict(re_samples)
            return df, stop, zero_pillar_amount
        

def median_filter_and_velocity(data):
    df = copy(data)
    for pillar_idx in range(0,9):
        forcex = df[f'S0_P{pillar_idx}_FX']
        forcey = df[f'S0_P{pillar_idx}_FY']
        x_filter = signal.medfilt(forcex,CONSTANTS.window_size)
        y_filter = signal.medfilt(forcey,CONSTANTS.window_size)
        df[f'S0_P{pillar_idx}_FILTERED_FX'] = x_filter
        df[f'S0_P{pillar_idx}_FILTERED_FY'] = y_filter
        forcex = x_filter
        forcey = y_filter
        vx = []
        vy = []
        for idx, f in enumerate(forcex):
            if idx == 0:
                vx.append(0.0)
                f_old = f
            else:
                vx.append((f - f_old) / CONSTANTS.t_interval)
                f_old = f
        del idx, f
        for idx, f in enumerate(forcey):
            if idx == 0:
                vy.append(0.0)
                f_old = f
            else:
                vy.append((f - f_old) / CONSTANTS.t_interval)
                f_old = f
        del idx, f
        vx = np.array(vx)
        vy = np.array(vy)
        df[f'S0_P{pillar_idx}_VX'] = vx
        df[f'S0_P{pillar_idx}_VY'] = vy
    return df


def one_hot_label_2cats(labels):
    one_hot = []
    for label in labels:
        if label == 0:
            one_hot.append([1, 0])
        elif label == 1:
            one_hot.append([0, 1])
        else:
            raise RuntimeError('Error')
    return np.array(one_hot)


def one_hot_label_3cats(labels):
    one_hot = []
    for label in labels:
        if label == 0:
            one_hot.append([1, 0, 0])
        elif label == 1:
            one_hot.append([0, 1, 0])
        elif label == 2:
            one_hot.append([0, 0, 1])
        else:
            raise RuntimeError('Error')
    return np.array(one_hot)


def split_seqs(total_len, sub_len):
    post_subs_num = math.floor(total_len / sub_len)
    subs = []
    pointer = 0
    for i in range(post_subs_num):
        subs.append([pointer, pointer+sub_len])
        pointer += sub_len
    if total_len % sub_len != 0:
        subs.append([-sub_len-1, -1])
    return subs


def rotational_augment(raw, angle):
    angle = math.radians(angle)
    rotate_matrix = np.asarray([[math.cos(angle), -math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])
    slip_feature_new = copy(raw)
    for idx in range(raw.shape[0]):
        slip_feature_new[idx, :2] = np.dot(rotate_matrix, raw[idx, :2])
    return slip_feature_new


def random_seq(total_len, sub_len):
    start_pos_end = total_len - sub_len
    start_pos = np.random.randint(0, start_pos_end)
    end_pos = start_pos + sub_len
    return start_pos, end_pos


def remove_list_from_list(raw_list, list_to_remove):
    for item in list_to_remove:
        raw_list.remove(item)
    return raw_list


def randomSplit(M, minV, maxV):
    N = round(M / ((minV + maxV)/2))
    res = []
    while N > 0:
        l = max(minV, M - (N-1)*maxV)
        r = min(maxV, M - (N-1)*minV)
        num = random.randint(l, r)
        N -= 1
        M -= num
        res.append(num)
    return res
