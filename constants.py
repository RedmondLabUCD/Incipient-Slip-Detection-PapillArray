#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 23:47:47 2022
@author: qiang
"""

stop_v_2100 = [0.5, 1.0, 2.0]
rest_point = 4000
stop_len = 800
contact_fz_th = 0.15
drop_rates = [0, 0.1, 0.2, 0.3, 0.4]

# Filter
window_size = 21
freq = 1000
t_interval = 1 / freq
drop_begin = 0

MIX_Z = True
RE_ORDER = True
ZERO_SEQUENCE = True

headers = ['time', 
           'S0_P0_contact', 'S0_P0_slip', 'S0_P0_DX', 'S0_P0_DY', 'S0_P0_DZ', 'S0_P0_FX', 'S0_P0_FY', 'S0_P0_FZ', 
           'S0_P1_contact', 'S0_P1_slip', 'S0_P1_DX', 'S0_P1_DY', 'S0_P1_DZ', 'S0_P1_FX', 'S0_P1_FY', 'S0_P1_FZ', 
           'S0_P2_contact', 'S0_P2_slip', 'S0_P2_DX', 'S0_P2_DY', 'S0_P2_DZ', 'S0_P2_FX', 'S0_P2_FY', 'S0_P2_FZ', 
           'S0_P3_contact', 'S0_P3_slip', 'S0_P3_DX', 'S0_P3_DY', 'S0_P3_DZ', 'S0_P3_FX', 'S0_P3_FY', 'S0_P3_FZ',
           'S0_P4_contact', 'S0_P4_slip', 'S0_P4_DX', 'S0_P4_DY', 'S0_P4_DZ', 'S0_P4_FX', 'S0_P4_FY', 'S0_P4_FZ', 
           'S0_P5_contact', 'S0_P5_slip', 'S0_P5_DX', 'S0_P5_DY', 'S0_P5_DZ', 'S0_P5_FX', 'S0_P5_FY', 'S0_P5_FZ', 
           'S0_P6_contact', 'S0_P6_slip', 'S0_P6_DX', 'S0_P6_DY', 'S0_P6_DZ', 'S0_P6_FX', 'S0_P6_FY', 'S0_P6_FZ', 
           'S0_P7_contact', 'S0_P7_slip', 'S0_P7_DX', 'S0_P7_DY', 'S0_P7_DZ', 'S0_P7_FX', 'S0_P7_FY', 'S0_P7_FZ', 
           'S0_P8_contact', 'S0_P8_slip', 'S0_P8_DX', 'S0_P8_DY', 'S0_P8_DZ', 'S0_P8_FX', 'S0_P8_FY', 'S0_P8_FZ'
           ]

pillar0 = ['S0_P0_contact', 'S0_P0_slip', 'S0_P0_FX', 'S0_P0_FY', 'S0_P0_FILTERED_FX', 'S0_P0_FILTERED_FY']
pillar1 = ['S0_P1_contact', 'S0_P1_slip', 'S0_P1_FX', 'S0_P1_FY', 'S0_P1_FILTERED_FX', 'S0_P1_FILTERED_FY']
pillar2 = ['S0_P2_contact', 'S0_P2_slip', 'S0_P2_FX', 'S0_P2_FY', 'S0_P2_FILTERED_FX', 'S0_P2_FILTERED_FY']
pillar3 = ['S0_P3_contact', 'S0_P3_slip', 'S0_P3_FX', 'S0_P3_FY', 'S0_P3_FILTERED_FX', 'S0_P3_FILTERED_FY']
pillar4 = ['S0_P4_contact', 'S0_P4_slip', 'S0_P4_FX', 'S0_P4_FY', 'S0_P4_FILTERED_FX', 'S0_P4_FILTERED_FY']
pillar5 = ['S0_P5_contact', 'S0_P5_slip', 'S0_P5_FX', 'S0_P5_FY', 'S0_P5_FILTERED_FX', 'S0_P5_FILTERED_FY']
pillar6 = ['S0_P6_contact', 'S0_P6_slip', 'S0_P6_FX', 'S0_P6_FY', 'S0_P6_FILTERED_FX', 'S0_P6_FILTERED_FY']
pillar7 = ['S0_P7_contact', 'S0_P7_slip', 'S0_P7_FX', 'S0_P7_FY', 'S0_P7_FILTERED_FX', 'S0_P7_FILTERED_FY']
pillar8 = ['S0_P8_contact', 'S0_P8_slip', 'S0_P8_FX', 'S0_P8_FY', 'S0_P8_FILTERED_FX', 'S0_P8_FILTERED_FY']

pillars = [pillar0,pillar1,pillar2,pillar3,pillar4,pillar5,pillar6,pillar7,pillar8]

online_raw = [
                # 'S0_P0_FILTERED_FX', 'S0_P0_FILTERED_FY',
              'S0_P1_FILTERED_FX', 'S0_P1_FILTERED_FY',
              'S0_P2_FILTERED_FX', 'S0_P2_FILTERED_FY',
              'S0_P3_FILTERED_FX', 'S0_P3_FILTERED_FY',
              'S0_P4_FILTERED_FX', 'S0_P4_FILTERED_FY',
              'S0_P5_FILTERED_FX', 'S0_P5_FILTERED_FY',
              'S0_P6_FILTERED_FX', 'S0_P6_FILTERED_FY',
              'S0_P7_FILTERED_FX', 'S0_P7_FILTERED_FY',
              'S0_P8_FILTERED_FX', 'S0_P8_FILTERED_FY',]