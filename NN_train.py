#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:40:38 2022

@author: qiang
"""
from NN_networks import  SlipDetectGlobalGru as Net
from NN_torch_dataloader import loader
import torch
import torch.nn as nn
import numpy as np
import utils
import argparse


def train(args):
    models = []
    for _ in range(args.model_num):
        models.append(Net(args).to(args.device))
    utils.save_config(args, models[0])
    loss_csv = utils.loss_log()
    slip_csv = utils.slip_log()
    train_dataset = loader(args)
    
    # Loss function
    if args.loss_func == 'mse':
        criterion = nn.MSELoss().to(args.device)
    elif args.loss_func == 'ce':
        criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.loss_func == 'bce':
        criterion = nn.BCELoss().to(args.device)
    else:
        raise RuntimeError('Error')
    
    # Optimizor
    optims = []
    if args.optim == 'sgd':
        for i in range(args.model_num):
            optims.append(torch.optim.SGD(models[i].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay))
    elif args.optim == 'adam':
        for i in range(args.model_num):
            optims.append(torch.optim.Adam(models[i].parameters(), lr=args.lr, weight_decay=args.weight_decay))
    else:
        raise RuntimeError('Error')
    
    # LR scheduler
    len_train = len(train_dataset)
    # exp_gamma = 0.98
    # exp_gamma_ratio = 0.98
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    step_size = len_train / args.train_batch_size
    scheduler = torch.optim.lr_scheduler.StepLR(optims[0], step_size=step_size, gamma=0.98)
    
    
    np.set_printoptions(suppress=True)
    global_count = 0
    print("Start training")
    for epoch in range(args.epochs):
    #↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Training area ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        for idx_model, model in enumerate(models):
            model.train()
            l_train = 0.0
            count_train = 0
            for idx,[features,labels] in enumerate(train_dataset):
                if np.random.uniform(0,1) < args.sample_rate:
                    if torch.cuda.is_available():
                        
                        features = torch.transpose(features.cuda(non_blocking=True).to(torch.float32),1,0)
                        labels = torch.transpose(labels.cuda(non_blocking=True).to(torch.float32),1,0)
    
                    h = torch.zeros(features.shape[1],args.hidden_dim).to(args.device)
                    loss = 0.0
                    for idx1 in range(features.shape[0]):
                        pred,h = model(features[idx1,...],h)
                        sub_loss = criterion(pred,labels[idx1])
                        loss += sub_loss
                        l_train += sub_loss.item()
                        count_train += 1
                        
                        optims[idx_model].zero_grad()
                        # loss.backward()
                        sub_loss.backward()
                        optims[idx_model].step()
                        h = h.detach()
        
                    print(f'Epoch: {epoch}/{args.epochs}  ModelNum: {idx_model}   Train: {idx}/{len(train_dataset)}   Loss: {l_train/count_train}')
        print('---'*20)
    
        log = [epoch]
        log.append(l_train/count_train)
        loss_csv.update(log, f'{args.save_path}/log.csv')
        if epoch % args.save_freq == 0:
            for idx_model, model in enumerate(models):
                torch.save(model.state_dict(),f'{args.save_path}/ckpt_{epoch}_model_{idx_model}.pth')
    #↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    for idx_model, model in enumerate(models):
        torch.save(model.state_dict(),f'{args.save_path}/ckpt_final_model_{idx_model}.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--mode', default="direct")
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--use-scheduler', default=False)
    parser.add_argument('--loss-func', default='ce')
    parser.add_argument('--optim', default='sgd')
    parser.add_argument('--stack-samples', default=40)
    parser.add_argument('--hidden-dim', default=128)
    parser.add_argument('--model-size', default='normal')
    parser.add_argument('--model-num', default=5)
    parser.add_argument('--sample-rate', default=0.9)
    parser.add_argument('--momentum', default=0.95)
    parser.add_argument('--weight-decay', default=1e-2)
    parser.add_argument('--train-batch-size', default=1024)
    parser.add_argument('--test-batch-size', default=1024)
    parser.add_argument('--shuffle', default=1)
    parser.add_argument('--use-gpu', default=1)
    parser.add_argument('--eval', default=0)
    parser.add_argument('--save-freq', default=1)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--dataset-path', default=None)
    args = parser.parse_args()
    args.exp_name = f'{args.mode}_{args.loss_func}_{args.optim}'
    args = utils.args_handler(args)
    train(args)
