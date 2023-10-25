#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:58:49 2022

@author: qiang
"""
import torch.nn as nn
import torch

class SlipDetectGlobalGru(nn.Module):
    def __init__(self, 
                 args,
                 bias=True):

        super(SlipDetectGlobalGru, self).__init__()

        self.projector = nn.Sequential(nn.Linear(args.input_dim, 1024, bias=bias),
                                           nn.BatchNorm1d(1024),
                                           nn.ReLU(),
                                           nn.Linear(1024, args.hidden_dim, bias=bias),
                                           nn.BatchNorm1d(args.hidden_dim),
                                           nn.ReLU(),
                                           )
        
        self.rnn = nn.GRUCell(input_size = args.hidden_dim,
                              hidden_size = args.hidden_dim, 
                              bias = bias)
            
        self.predictor = nn.Sequential(nn.Linear(args.hidden_dim, 256, bias=bias),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(),
                                       nn.Linear(256, 128, bias=bias),
                                       nn.BatchNorm1d(128),
                                       nn.ReLU(),
                                       nn.Linear(128, args.categories, bias=bias),
                                       nn.BatchNorm1d(args.categories),
                                       nn.Softmax(dim=1),
                                       )
    def forward(self,inp, h):
        x = self.projector(inp)
        h = self.rnn(x, h)
        x = self.predictor(h)
        return x,h
    
class SlipDetectGlobalLstm(nn.Module):
    def __init__(self, 
                 args,
                 bias=True):

        super(SlipDetectGlobalLstm, self).__init__()

        self.projector = nn.Sequential(nn.Linear(args.input_dim, 1024, bias=bias),
                                           nn.BatchNorm1d(1024),
                                           nn.ReLU(),
                                           nn.Linear(1024, args.hidden_dim, bias=bias),
                                           nn.BatchNorm1d(args.hidden_dim),
                                           nn.ReLU(),
                                           )
        
        self.rnn = nn.LSTMCell(input_size = args.hidden_dim,
                              hidden_size = args.hidden_dim, 
                              bias = bias)
            
        self.predictor = nn.Sequential(nn.Linear(args.hidden_dim*2, 256, bias=bias),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(),
                                       nn.Linear(256, 128, bias=bias),
                                       nn.BatchNorm1d(128),
                                       nn.ReLU(),
                                       nn.Linear(128, args.categories, bias=bias),
                                       nn.BatchNorm1d(args.categories),
                                       nn.Softmax(dim=1),
                                       )
    def forward(self,inp, h,c):
        x = self.projector(inp)
        h, c = self.rnn(x, (h,c))
        x = torch.cat((h,c),dim=1)
        x = self.predictor(x)
        return x,h,c
