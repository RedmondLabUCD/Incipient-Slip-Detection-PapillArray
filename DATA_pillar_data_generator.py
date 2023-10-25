#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:35:44 2022

@author: qiang

Generate pillar data for data augmentation
"""
import argparse
from glob import glob
import os
import constants as CONSTANTS
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


def run(args):
    if not args.data_load_path:
        args.data_load_path = 'datasets'
    
    if args.fullset:
        train_items = glob(f"{args.data_load_path}/fullset/*")
        train_and_test = [train_items]
    else:
        train_items = glob(f"{args.data_load_path}/train/*")
        test_items = glob(f"{args.data_load_path}/test/*")
        train_and_test = [train_items, test_items]
    
    pillar_path = f"{args.data_load_path}/pillar_data"
    
    if not os.path.exists(pillar_path):
        os.makedirs(pillar_path)
    
    pillar_data_train = []
    stop_data_train_2100 = []
    stop_data_train_2050 = []
    slip_data_train = []
    
    
    pillar_data_test = []
    stop_data_test_2100 = []
    stop_data_test_2050 = []
    slip_data_test = []
    
    t_2100_num = 0
    t_2050_num = 0
    for idx1, moves in enumerate(train_and_test):
        if idx1 == 0:
            mode = 'train'
        elif idx1 == 1:
            mode = 'test'
        else:
            raise RuntimeError('Error')
    
        for idx2, move in enumerate(moves):
            move_items = glob(f"{move}/*")
            for idx3, file_path in enumerate(move_items):
                data = pd.read_csv(file_path, header=None)
                df = pd.DataFrame(data)
                df.columns = CONSTANTS.headers
                print(
                    f"Train or test: {idx1}/{len(train_and_test)}, Move: {idx2}/{len(moves)}, File: {idx3}/{len(move_items)}")
                
                for idx4 in range(9):
                    sub_feature = {}
                    sub_feature['FX'] = df[f'S0_P{idx4}_FX'][:CONSTANTS.rest_point]
                    sub_feature['FY'] = df[f'S0_P{idx4}_FY'][:CONSTANTS.rest_point]
                    sub_feature['SLIP'] = df[f'S0_P{idx4}_slip'][:CONSTANTS.rest_point]
                    
                    if file_path[-8:-4] == 'stop':
                        v = float(file_path.split("-")[1][2:])
                        if mode == 'train':
                            if v in CONSTANTS.stop_v_2100:
                                stop_data_train_2100.append(sub_feature)
                                t_2100_num += 1
                            else:
                                stop_data_train_2050.append(sub_feature)
                                t_2050_num += 1
                        else:
                            if v in CONSTANTS.stop_v_2100:
                                stop_data_test_2100.append(sub_feature)
                                t_2100_num += 1
                            else:
                                stop_data_test_2050.append(sub_feature)
                                t_2050_num += 1
                    
                    else:
                        if mode == 'train':
                            slip_data_train.append(sub_feature)
                        else:
                            slip_data_test.append(sub_feature)
    
    pillar_data_train = {'slip_data':slip_data_train,
                         'stop_data_2100':stop_data_train_2100,
                         'stop_data_2050':stop_data_train_2050,}
    pillar_data_test = {'slip_data':slip_data_test,
                         'stop_data_2100':stop_data_test_2100,
                         'stop_data_2050':stop_data_test_2050,}
    
    np.save(f"{pillar_path}/pillar_data_train.npy", pillar_data_train)
    np.save(f"{pillar_path}/pillar_data_test.npy", pillar_data_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-load-path', default=None)
    parser.add_argument('--fullset', default=True)
    args = parser.parse_args()
    run(args)