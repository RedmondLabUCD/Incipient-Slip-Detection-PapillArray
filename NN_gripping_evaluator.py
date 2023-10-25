#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 20:07:21 2023

@author: qiang
"""

from glob import glob
import os
import torch
import numpy as np
from copy import copy
import utils
import argparse
import constants as CONSTANTS
import matplotlib.pyplot as plt
import warnings
import time
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import seaborn
warnings.filterwarnings("ignore")

def run(pre_args):
    online_data_path = pre_args.online_data_path
    model_path = pre_args.model_path
    model_num = pre_args.model_num
    
    objs = ['COFFEE','CHIPS','PAPER_BOX','WOOD','PLASTIC_BOTTEL','MI']
    TH = 0.5 * 100
    
    trans_crit = 0.002
    rot_crit = 2 
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    TOTLE = 0
    args = utils.get_default_eval_args(model_path, model_num)
    models = utils.load_model(args)
    
    correct_path = []
    t_slips = []
    t_slips_trans = []
    t_slips_rot = []
    t_slips_trans_rot = []
    
    for obj in objs:
        obj_path = os.path.join(online_data_path, obj)
        slip_paths = glob(f'{obj_path}/*/slip/f=*/sensor_data.npy')
        stop_paths = glob(f'{obj_path}/*/stop/f=*/sensor_data.npy')
        
        paths = slip_paths + stop_paths
    
        for idx, path in enumerate(paths):
            t_total = 0
            count = 0
            infos = utils.get_online_data_infos(path)
            online_data = np.load(path, allow_pickle=True).item()
            t_raw = np.array(online_data['raw_t'])
    
            raw_data = np.array(utils.bias_data(online_data['raw_data_8']))
            robot_data_raw = utils.modify_robot_data(
                np.array(online_data['robot_data']), infos, trans_crit, rot_crit)
            filtered_data = np.array(utils.bias_data(
                online_data['filtered_data_8']))
    
            df = pd.DataFrame(filtered_data)
            df.columns = CONSTANTS.online_raw
    
            feature1 = np.array(df[CONSTANTS.pillars[1][-2:]])
            feature2 = np.array(df[CONSTANTS.pillars[2][-2:]])
            feature3 = np.array(df[CONSTANTS.pillars[3][-2:]])
            feature4 = np.array(df[CONSTANTS.pillars[4][-2:]])
            feature5 = np.array(df[CONSTANTS.pillars[5][-2:]])
            feature6 = np.array(df[CONSTANTS.pillars[6][-2:]])
            feature7 = np.array(df[CONSTANTS.pillars[7][-2:]])
            feature8 = np.array(df[CONSTANTS.pillars[8][-2:]])
    
            features = np.concatenate((feature1, feature2,
                                       feature3, feature4, feature5,
                                       feature6, feature7, feature8), axis=1)
    
            features_len = features.shape[0] - \
                (features.shape[0] % args.stack_samples)
            features = features[:features_len]
            raw_data = raw_data[:features_len]
            t_raw = t_raw[:features_len]
            robot_data_raw = robot_data_raw[:features_len]
    
            t = []
            robot_data = []
            for i in range(features_len):
                if i % args.stack_samples == 0:
                    t.append(online_data["raw_t"][i])
                    robot_data.append(robot_data_raw[i])
    
            reshaped_features = copy(features).reshape(
                int(features_len/args.stack_samples), int(16 * args.stack_samples))
            global_slip = []
            with torch.no_grad():
                for idx_model, model in enumerate(models):
                    model.eval()
                    h = torch.zeros(1, args.hidden_dim).to(args.device)
                    sub_slip = []
                    for idx_row, row in enumerate(reshaped_features):
                        tensor_row = torch.tensor([copy(row)]).to(
                            torch.float32).to(args.device)
                        t0 = time.time()
                        pred, h = model(tensor_row, h)
                        t_total += time.time() - t0
                        count += 1
                        sub_slip.append(pred[0, 1].detach().cpu().numpy())
                    global_slip.append(sub_slip)
                global_slip = np.array(global_slip).mean(axis=0) * 100
            print(idx)
            
            t_slip = None
            r_slip = None
            for idx2, prob in enumerate(global_slip):
                if prob >= TH:
                    t_slip = t[idx2]
                    r_slip = robot_data[idx2+2]
                    break
            
            TOTLE += 1
            if infos['state'] == 'slip':
                if r_slip <= 1.0:
                    TP += 1
                    correct_path.append(path)
                    
                    if infos['move_type'] == 'trans':
                        t_slips_trans.append(r_slip)
                    if infos['move_type'] == 'rot':
                        t_slips_rot.append(r_slip)
                    if infos['move_type'] == 'trans+rot':
                        t_slips_trans_rot.append(r_slip)
                    t_slips.append(t_slips)
                    
                else:
                    FN += 1
                    print("FN: ", path)
            if infos['state'] == 'stop':
                if r_slip != None:
                    FP += 1
                    print("FP: ", path)
                else:
                    TN += 1
    
    
    a = np.load('online_results.npy',allow_pickle=True).item()
    t_slips_trans=a['t_slips_trans']
    t_slips_rot=a['t_slips_rot']
    t_slips_trans_rot=a['t_slips_trans_rot']
    
    
    global_bins = 12
    title_font_size = 22
    font_colorbar = 15
    nbins_colorbar = 5
    fraction_colorbar = 0.055
    pad_colorbar = 0.015
    label_font_size = 18
    label_pad = 0
    x_label_pad = 5
    box_color='silver'
    gradient_cmap = 'Blues'
    plt.rcParams['font.size'] = font_colorbar
    
    fig, axs = plt.subplots(nrows=2, 
                          ncols=3, 
                          sharex=True, 
                          gridspec_kw={"height_ratios": (.15, .85)},
                          figsize=(24,6),
                          dpi=600)
    
    ax_box1 = axs[0, 0]
    ax_box2 = axs[0, 1]
    ax_box3 = axs[0, 2]
    ax_hist1 = axs[1, 0]
    ax_hist2 = axs[1, 1]
    ax_hist3 = axs[1, 2]
    
    seaborn.boxplot(np.array(t_slips_trans), ax=ax_box1,color=box_color)
    ax_box1.title.set_text("Translation sequences")
    ax_box1.title.set_fontsize(title_font_size)
    ax_box1.set_facecolor('lavender')
    
    seaborn.histplot(np.array(t_slips_trans), 
                 ax=ax_hist1,
                 kde=True, 
                 bins=global_bins,
                 color='r',
                 stat="count",)
    
    ax_hist1.set_ylabel('Count', fontsize=label_font_size, labelpad=label_pad)
    ax_hist1.set_xlabel('Normalized displacement moved by robot, $D_{norm}$', fontsize=label_font_size, labelpad=x_label_pad)
    ax_hist1.set_xlim(0, 1.0)
    xlim = ax_hist1.get_xlim()
    ylim = ax_hist1.get_ylim()
    x = np.linspace(xlim[0], 1, 200)
    polygon = ax_hist1.fill_between(x, ylim[0], ylim[1], lw=0, color='none')
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    cg = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = ax_hist1.imshow(cg, cmap=gradient_cmap, aspect='auto', alpha=0.7, extent=[
                          verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    
    mappable = plt.cm.ScalarMappable(cmap=gradient_cmap)
    mappable.set_array([0.0,0.2,0.4,0.6,0.8,1.0])
    cb1 = fig.colorbar(mappable, 
                       ax=ax_hist1,
                       fraction=fraction_colorbar, 
                       pad=pad_colorbar,
                       alpha=0.7,
                       location='right')
    
    seaborn.boxplot(np.array(t_slips_rot), ax=ax_box2,color=box_color)
    ax_box2.title.set_text("Rotation sequences")
    ax_box2.title.set_fontsize(title_font_size)
    ax_box2.set_facecolor('lavender')
    seaborn.histplot(np.array(t_slips_rot), 
                 ax=ax_hist2,
                 kde=True, 
                 bins=global_bins,
                 color='r',
                 stat="count",)
    ax_hist2.set_ylabel('Count', fontsize=label_font_size, labelpad=label_pad)
    ax_hist2.set_xlabel('Normalized displacement moved by robot, $D_{norm}$', fontsize=label_font_size, labelpad=x_label_pad)
    ax_hist2.set_xlim(0, 1.0)
    xlim = ax_hist2.get_xlim()
    ylim = ax_hist2.get_ylim()
    x = np.linspace(xlim[0], 1, 200)
    polygon = ax_hist2.fill_between(x, ylim[0], ylim[1], lw=0, color='none')
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    cg = np.linspace(0, 1, 256).reshape(1, -1)
    gradient =ax_hist2.imshow(cg, cmap=gradient_cmap, aspect='auto', alpha=0.7, extent=[
                          verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    
    mappable = plt.cm.ScalarMappable(cmap=gradient_cmap)
    mappable.set_array([0.0,0.2,0.4,0.6,0.8,1.0])
    cb2 = fig.colorbar(mappable, 
                       ax=ax_hist2,
                       fraction=fraction_colorbar, 
                       pad=pad_colorbar,
                       alpha=0.7)
    tick_locator2 = ticker.MaxNLocator(nbins=nbins_colorbar)
    cb2.locator = tick_locator2
    cb2.update_ticks()
    ax_hist2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    seaborn.boxplot(np.array(t_slips_trans_rot), ax=ax_box3,color=box_color)
    ax_box3.title.set_text("Translation + Rotation sequences")
    ax_box3.title.set_fontsize(title_font_size)
    ax_box3.set_facecolor('lavender')
    seaborn.histplot(np.array(t_slips_trans_rot), 
                 ax=ax_hist3,
                 kde=True, 
                 bins=global_bins,
                 color='r',
                 stat="count",
                 )
    ax_hist3.set_ylabel('Count', fontsize=label_font_size, labelpad=label_pad)
    ax_hist3.set_xlabel('Normalized displacement moved by robot, $D_{norm}$', fontsize=label_font_size, labelpad=x_label_pad)
    ax_hist3.set_xlim(0, 1.0)
    xlim = ax_hist3.get_xlim()
    ylim = ax_hist3.get_ylim()
    x = np.linspace(xlim[0], 1, 200)
    polygon = ax_hist3.fill_between(x, ylim[0], ylim[1], lw=0, color='none')
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    cg = np.linspace(0, 1, 256).reshape(1, -1)
    gradient =ax_hist3.imshow(cg, cmap=gradient_cmap, aspect='auto', alpha=0.7, extent=[
                          verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    
    mappable = plt.cm.ScalarMappable(cmap=gradient_cmap)
    mappable.set_array([0.0,0.2,0.4,0.6,0.8,1.0])
    cb2 = fig.colorbar(mappable, 
                       ax=ax_hist3,
                       fraction=fraction_colorbar, 
                       pad=pad_colorbar,
                       alpha=0.7)
    tick_locator2 = ticker.MaxNLocator(nbins=nbins_colorbar)
    cb2.locator = tick_locator2
    cb2.update_ticks()
    ax_hist3.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.subplots_adjust(hspace=0.08)
    
    pos1 = ax_box1.get_position()
    pos_new1 = [pos1.x0, pos1.y0, 0.2115, pos1.height]
    ax_box1.set_position(pos_new1)
    
    pos2 = ax_box2.get_position()
    pos_new2 = [pos2.x0, pos2.y0, 0.2115, pos2.height]
    ax_box2.set_position(pos_new2)
    
    pos3 = ax_box3.get_position()
    pos_new3 = [pos3.x0, pos3.y0, 0.2115, pos3.height]
    ax_box3.set_position(pos_new3)
    
    plt.savefig('gripping-results.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='<trained model to your local path>')
    parser.add_argument('--model_num', default=16)
    parser.add_argument('--online-data-path ', default='./gripping-data')
    args = parser.parse_args()
    args = utils.args_handler(args)
    run(args)