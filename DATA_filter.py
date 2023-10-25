#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 17:07:36 2022

@author: qiang
"""
import numpy as np
from copy import copy

class OnlineMedianFilter():
    def __init__(self,
                  window_size,
                  ):
        self.window_size = window_size
        self.reset_filter()
    
    def reset_filter(self):
        self.window_buffer = []
        self.first_flag = True
    
    def filter_and_update(self, sample):
        if len(self.window_buffer) < self.window_size:
            self.window_buffer.append(sample)
            
        else:
            self.window_buffer[:-1] = self.window_buffer[1:]
            self.window_buffer[-1] = sample

        return np.median(copy(self.window_buffer), axis=0)