#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:35:44 2022
@author: qiang
"""

import argparse
from glob import glob
import os
import constants as CONSTANTS
import numpy as np
import warnings
import DATA_handler as dataset_handler
import random
warnings.filterwarnings("ignore")


def run(args):
    """
    The main function for augmenting the dataset, the augmentation include rotational transformation,
    mixing pillars positions, mixing sequence.
    For using, you MUST create the pillar data by DATA_pillars_data_generator.py firstly. 
    You MUST put the raw data under the datasets directory.
    """
    
    feature_start_num = 4 
    feature_end_num = 6 
    
    if not args.data_load_path:
        args.data_load_path = 'datasets'
    
    if args.fullset:
        train_items = glob(f"{args.data_load_path}/fullset/*")
        train_and_test = [train_items]
    else:
        train_items = glob(f"{args.data_load_path}/train/*")
        test_items = glob(f"{args.data_load_path}/test/*")
        train_and_test = [train_items, test_items]
    
    train_pillar_set = np.load(f'{args.data_load_path}/pillar_data/pillar_data_train.npy', allow_pickle=True).item()
    test_pillar_set = np.load(f'{args.data_load_path}/pillar_data/pillar_data_test.npy', allow_pickle=True).item()
    
    sampler = dataset_handler.pillar_data_sampler(train_pillar_set,
                                                  test_pillar_set,
                                                  pillar_num=9,
                                                  zero_pillar_ratio = 0.2,
                                                  max_zero_num = 3,
                                                  gussian_loc = 0.0,
                                                  gussian_scale = 0.001)
    
    merge_path = f"{args.data_load_path}/merge"
    
    if not os.path.exists(merge_path):
        os.makedirs(merge_path)
    
    for idx1, moves in enumerate(train_and_test):
        merged_dataset = []
        if idx1 == 0:
            mode = 'train'
        elif idx1 == 1:
            mode = 'test'
        else:
            raise RuntimeError('Error')
    
        for idx2, move in enumerate(moves):
            move_items = glob(f"{move}/*")
            for idx3, file_path in enumerate(move_items):
                print(
                    f"Train or test: {idx1}/{len(train_and_test)}, Move: {idx2}/{len(moves)}, File: {idx3}/{len(move_items)}")
                df, labels = dataset_handler.reshape_sequence_raw(file_path)
                subs = dataset_handler.split_seqs(df.shape[0], args.seq_len)
                for [sub_start, sub_end] in subs:
                    for repeat in range(args.angle_repeats):
                        if repeat == 0:
                            aug_angle = 0
                        else:
                            aug_angle = random.uniform(0, 360)
                        
                        feature0 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[0][feature_start_num:feature_end_num]])
                        feature1 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[1][feature_start_num:feature_end_num]])
                        feature2 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[2][feature_start_num:feature_end_num]])
                        feature3 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[3][feature_start_num:feature_end_num]])
                        feature4 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[4][feature_start_num:feature_end_num]])
                        feature5 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[5][feature_start_num:feature_end_num]])
                        feature6 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[6][feature_start_num:feature_end_num]])
                        feature7 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[7][feature_start_num:feature_end_num]])
                        feature8 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[8][feature_start_num:feature_end_num]])
                        
                        one_hot_labels = dataset_handler.one_hot_label_2cats(labels)[sub_start:sub_end]
    
                        feature0_augments = dataset_handler.rotational_augment(feature0, aug_angle)
                        feature1_augments = dataset_handler.rotational_augment(feature1, aug_angle)
                        feature2_augments = dataset_handler.rotational_augment(feature2, aug_angle)
                        feature3_augments = dataset_handler.rotational_augment(feature3, aug_angle)
                        feature4_augments = dataset_handler.rotational_augment(feature4, aug_angle)
                        feature5_augments = dataset_handler.rotational_augment(feature5, aug_angle)
                        feature6_augments = dataset_handler.rotational_augment(feature6, aug_angle)
                        feature7_augments = dataset_handler.rotational_augment(feature7, aug_angle)
                        feature8_augments = dataset_handler.rotational_augment(feature8, aug_angle)
    
                        data_seq = np.concatenate((feature0_augments, feature1_augments, feature2_augments,
                                                    feature3_augments, feature4_augments, feature5_augments,
                                                    feature6_augments, feature7_augments, feature8_augments,
                                                    one_hot_labels), axis=1)
    
                        merged_dataset.append(data_seq)
                        
            pillar_repeats = args.pillars_to_raw * len(move_items)
            
        if pillar_repeats:
            print('Advanced augmenting')
            for i in range(pillar_repeats):
                df, stop, zero_pillar_amount = sampler.sample(mode=mode,
                                          in_raw_dataset_style=True)
                df, labels = dataset_handler.reshape_sequence(df, stop)
                if args.drop_sample:
                    drop_rate = random.sample(CONSTANTS.drop_rates, 1)[0]
                    df, labels = dataset_handler.drop_samples(df, labels, drop_rate)
                subs = dataset_handler.split_seqs(df.shape[0], args.seq_len)
                
                if i % 10 == 0:
                    print(
                        f"Mode: {mode}, Idx: {i}/{pillar_repeats}")
                
                for [sub_start, sub_end] in subs:
                    for repeat in range(args.angle_repeats):
                        if repeat == 0:
                            aug_angle = 0
                        else:
                            aug_angle = random.uniform(0, 360)
                        
                        feature0 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[0][feature_start_num:feature_end_num]])
                        feature1 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[1][feature_start_num:feature_end_num]])
                        feature2 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[2][feature_start_num:feature_end_num]])
                        feature3 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[3][feature_start_num:feature_end_num]])
                        feature4 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[4][feature_start_num:feature_end_num]])
                        feature5 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[5][feature_start_num:feature_end_num]])
                        feature6 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[6][feature_start_num:feature_end_num]])
                        feature7 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[7][feature_start_num:feature_end_num]])
                        feature8 = np.array(df[sub_start:sub_end][CONSTANTS.pillars[8][feature_start_num:feature_end_num]])
                        
                        one_hot_labels = dataset_handler.one_hot_label_2cats(labels)[sub_start:sub_end]
                        
                        feature0_augments = dataset_handler.rotational_augment(feature0, aug_angle)
                        feature1_augments = dataset_handler.rotational_augment(feature1, aug_angle)
                        feature2_augments = dataset_handler.rotational_augment(feature2, aug_angle)
                        feature3_augments = dataset_handler.rotational_augment(feature3, aug_angle)
                        feature4_augments = dataset_handler.rotational_augment(feature4, aug_angle)
                        feature5_augments = dataset_handler.rotational_augment(feature5, aug_angle)
                        feature6_augments = dataset_handler.rotational_augment(feature6, aug_angle)
                        feature7_augments = dataset_handler.rotational_augment(feature7, aug_angle)
                        feature8_augments = dataset_handler.rotational_augment(feature8, aug_angle)
        
                        data_seq = np.concatenate((feature0_augments, feature1_augments, feature2_augments,
                                                    feature3_augments, feature4_augments, feature5_augments,
                                                    feature6_augments, feature7_augments, feature8_augments,
                                                    one_hot_labels), axis=1)
        
                        merged_dataset.append(data_seq)
    
        merged_dataset = np.array(merged_dataset)
        np.save(f"{merge_path}/dataset_{mode}.npy", merged_dataset)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', default=360)
    parser.add_argument('--angle-repeats', default=8)
    parser.add_argument('--data-load-path', default=None)
    parser.add_argument('--pillars-to-raw', default=1)
    parser.add_argument('--drop-sample', default=True)
    parser.add_argument('--fullset', default=True)
    args = parser.parse_args()
    run(args)