#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:59:14 2020

@author: veronikasamborska
"""


import sys
import scipy.io
import data_import_ephys as dimp
import numpy as np

ephys_path = '/Users/veronikasamborska/Desktop/neurons'
beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = dimp.import_code(ephys_path,beh_path)

experiment_aligned_PFC = dimp.all_sessions_aligment(PFC, all_sessions)
experiment_aligned_HP = dimp.all_sessions_aligment(HP, all_sessions)

#data_PFC = dimp.create_mat(experiment_aligned_PFC,'PFC_new_files')
#data_HP = dimp.create_mat(experiment_aligned_HP, 'HP_new_files')
