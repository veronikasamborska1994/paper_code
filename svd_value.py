#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:15:34 2020

@author: veronikasamborska
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:37:07 2020

@author: veronikasamborska
"""

from sklearn.linear_model import LogisticRegression
from collections import OrderedDict
#import regression_function as reg_f
from sklearn.model_selection import StratifiedKFold
import palettable

import matplotlib.pyplot as plt
import numpy as np
# import sys 
# import statsmodels.api as sm
# from statsmodels.stats.anova import AnovaRM
# import pandas as pd
# from palettable import wesanderson as wes
from scipy import stats 
import random
import seaborn as sns
from itertools import combinations 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}
import scipy
     

def load():
    
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')

def task_ind(task, a_pokes, b_pokes):
    
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task));
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid


def rew_prev_behaviour(data,n, perm = True):
   
    if perm:
        dm = data[0]
    else:
        dm = data['DM'][0]    
    results_array = []
    
    for  s, sess in enumerate(dm):
            
         DM = dm[s] 
           
         choices = DM[:,1]
         
         reward = DM[:,2]
         
         previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1]
         
         previous_choices = scipy.linalg.toeplitz(choices-0.5, np.zeros((1,n)))[n-1:-1]
         
         interactions = scipy.linalg.toeplitz((((choices-0.5)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1]
         

         choices_current = choices[n:]
         ones = np.ones(len(interactions)).reshape(len(interactions),1)
         
         X = np.hstack([previous_rewards,previous_choices,interactions,ones])
         
         model = LogisticRegression()
         results = model.fit(X,choices_current)
         results_array.append(results.coef_[0])

    average = np.mean(results_array,0)
        
    return average

def regression_code_session(data, design_matrix): 
    
    tc = np.identity(design_matrix.shape[1])
    
    pdes = np.linalg.pinv(design_matrix)
    tc_pdes = np.matmul(tc,pdes)
    pdes_tc = np.matmul(np.transpose(pdes),np.transpose(tc))
    
    prevar = np.diag(np.matmul(tc_pdes, pdes_tc))
    
    R = np.identity(design_matrix.shape[0]) - np.matmul(design_matrix, pdes)
    tR = np.trace(R)
    
    pe = np.matmul(pdes,data)
    cope = np.matmul(tc,pe)
    
    res = data - np.matmul(design_matrix,pe)
    sigsq = np.sum((res*res)/tR, axis = 0)
    sigsq = np.reshape(sigsq,(1,res.shape[1]))
    prevar = np.reshape(prevar,(tc.shape[0],1))
    varcope = prevar*sigsq
        
    return cope,varcope

def value_reg_svd(data, n = 10, plot_a = False, plot_b = False,  first_half = 1, a ='PFC',perm =False, t = 0):
  
   # dm = data['DM'][0]
   # firing = data['Data'][0]

    average = rew_prev_behaviour(data, n = n, perm = perm )
    if a == 'PFC' or a =='HP':
        all_subjects = data['DM'][0]
        all_firing = data['Data'][0]
    else:
        all_subjects = data[0]
        all_firing = data[1]

    C_1 = []; C_2 = []; C_3 = []
    
    
    
    for  s, sess in enumerate(all_subjects):
        
       
        DM = all_subjects[s]
        block = DM[:,4]
        block_df = np.diff(block)
        ind_block = np.where(block_df != 0)[0]
        
        if len(ind_block) >10:
    
            firing_rates = all_firing[s] #[:ind_block[11]]
            n_trials, n_neurons, n_timepoints = firing_rates.shape
            choices = DM[:,1]
            reward = DM[:,2] #[:ind_block[11]]  
            state = DM[:,0]
            task =  DM[:,5] #[:ind_block[11]]
           
            a_pokes = DM[:,6] #[:ind_block[11]]
            b_pokes = DM[:,7] #[:ind_block[11]]
            
            taskid = task_ind(task, a_pokes, b_pokes)
          
            if t == 0:
                
                task_1 = np.where(task == 1)[0]
                task_2 = np.where(task == 2)[0]
                task_3 = np.where(task == 3)[0]
            else:
                task_1 = np.where(taskid == 1)[0]
                task_2 = np.where(taskid == 2)[0]
                task_3 = np.where(taskid == 3)[0]
                
          
            reward_current = reward
            choices_current = choices - 0.5
    
           
            rewards_1 = reward_current[task_1]
            choices_1 = choices_current[task_1]
            
            previous_rewards_1 = scipy.linalg.toeplitz(rewards_1, np.zeros((1,n)))[n-1:-1]         
            previous_choices_1 = scipy.linalg.toeplitz(0.5-choices_1, np.zeros((1,n)))[n-1:-1]       
            interactions_1 = scipy.linalg.toeplitz((((0.5-choices_1)*(rewards_1-0.5))*2),np.zeros((1,n)))[n-1:-1]
             
    
            ones = np.ones(len(interactions_1)).reshape(len(interactions_1),1)
             
            X_1 = np.hstack([previous_rewards_1,previous_choices_1,interactions_1,ones])
            value_1 =np.matmul(X_1, average)
          
            rewards_1 = rewards_1[n:]
            choices_1 = choices_1[n:]
           
            
            ones_1 = np.ones(len(choices_1))
            trials_1 = len(choices_1)
            
            state_1 = state[task_1]
            state_1 = state_1[n:]
            
            state_2 = state[task_2]
            state_2 = state_2[n:]
            
            state_3 = state[task_2]
            state_3 = state_3[n:]
            
            
            
          
            firing_rates_1 = firing_rates[task_1][n:]
            
            a_1 = np.where(choices_1 == 0.5)[0]
            b_1 = np.where(choices_1 == -0.5)[0]
            
            if plot_a == True:
                
                mean = np.mean(value_1[a_1])
               
                value_high = np.where(value_1 < mean)
                low_high = np.where(value_1 > mean)
               
                ind_st_b = np.intersect1d(value_high,a_1)
                ind_st_a = np.intersect1d(low_high,a_1)
               
                ind_1_b = ind_st_b[:int(len(ind_st_b)/2)]
                ind_2_b =  ind_st_b[int(len(ind_st_b)/2):]
               
                ind_1_a = ind_st_a[:int(len(ind_st_a)/2)]
                ind_2_a =  ind_st_a[int(len(ind_st_a)/2):]
                
                
                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))
               
               
                if first_half == 1:
                    
                    rewards_1 = rewards_1[ind_1] 
                    choices_1 = choices_1[ind_1]    
                    value_1 = value_1[ind_1]
                    ones_1  = ones_1[ind_1]
                    firing_rates_1 = firing_rates_1[ind_1]
                    
                elif first_half == 2:
                  
                    rewards_1 = rewards_1[ind_2] 
                    choices_1 = choices_1[ind_2]    
                    value_1 = value_1[ind_2]
                    ones_1  = ones_1[ind_2]
                    firing_rates_1 = firing_rates_1[ind_2]
                    
             
                   
                
              
            elif plot_b == True:
                
                mean = np.mean(value_1[b_1])

                value_high = np.where(value_1 < mean)
                low_high = np.where(value_1 >= mean)
               
                ind_st_b = np.intersect1d(value_high,b_1)
                ind_st_a = np.intersect1d(low_high,b_1)

                ind_1_b = ind_st_b[:int(len(ind_st_b)/2)]
                ind_2_b =  ind_st_b[int(len(ind_st_b)/2):]
               
                ind_1_a = ind_st_a[:int(len(ind_st_a)/2)]
                ind_2_a =  ind_st_a[int(len(ind_st_a)/2):]
                 

                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))
              
                if first_half == 1:
                    rewards_1 = rewards_1[ind_1] 
                    choices_1 = choices_1[ind_1]    
                    value_1 = value_1[ind_1]
                    ones_1  = ones_1[ind_1]
                    firing_rates_1 = firing_rates_1[ind_1]
                    
                elif first_half == 2:
                  
                    rewards_1 = rewards_1[ind_2] 
                    choices_1 = choices_1[ind_2]    
                    value_1 = value_1[ind_2]
                    ones_1  = ones_1[ind_2]
                    firing_rates_1 = firing_rates_1[ind_2]
              
                
                      
                
                
            predictors_all = OrderedDict([
                                        ('Reward', rewards_1),
                                        ('Value',value_1),                                      
                                        ('ones', ones_1)])
            
            X_1 = np.vstack(predictors_all.values()).T[:trials_1,:].astype(float)
            
            n_predictors = X_1.shape[1]
            y_1 = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            tstats,x = regression_code_session(y_1, X_1)
    
            C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
            
            
            rewards_2 = reward_current[task_2]
            choices_2 = choices_current[task_2]
            
            previous_rewards_2 = scipy.linalg.toeplitz(rewards_2, np.zeros((1,n)))[n-1:-1]      
            previous_choices_2 = scipy.linalg.toeplitz(0.5-choices_2, np.zeros((1,n)))[n-1:-1]        
            interactions_2 = scipy.linalg.toeplitz((((0.5-choices_2)*(rewards_2-0.5))*2),np.zeros((1,n)))[n-1:-1]
             
    
            ones = np.ones(len(interactions_2)).reshape(len(interactions_2),1)
             
            X_2 = np.hstack([previous_rewards_2,previous_choices_2,interactions_2,ones])
            value_2 =np.matmul(X_2, average)
            
            rewards_2 = rewards_2[n:]
            choices_2 = choices_2[n:]
            
            ones_2 = np.ones(len(choices_2))
            trials_2 = len(choices_2)
    
            firing_rates_2 = firing_rates[task_2][n:]              
     
            a_2 = np.where(choices_2 == 0.5)[0]
            b_2 = np.where(choices_2 == -0.5)[0]
            
            if plot_a == True:
                
                mean = np.mean(value_2[a_2])
    
                value_high = np.where(value_2 < mean)
                low_high = np.where(value_2 >= mean)
               
                ind_st_b = np.intersect1d(value_high,a_2)
                ind_st_a = np.intersect1d(low_high,a_2)
                 
                ind_1_b = ind_st_b[:int(len(ind_st_b)/2)]
                ind_2_b =  ind_st_b[int(len(ind_st_b)/2):]
               
                ind_1_a = ind_st_a[:int(len(ind_st_a)/2)]
                ind_2_a =  ind_st_a[int(len(ind_st_a)/2):]
               

                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))
             
                if first_half == 1:
                    rewards_2 = rewards_2[ind_1] 
                    choices_2 = choices_2[ind_1]    
                    value_2 = value_2[ind_1]
                    ones_2  = ones_2[ind_1]
                    firing_rates_2 = firing_rates_2[ind_1]
                    
                elif first_half == 2:
                    rewards_2 = rewards_2[ind_2] 
                    choices_2 = choices_2[ind_2]    
                    value_2 = value_2[ind_2]
                    ones_2  = ones_2[ind_2]
                    firing_rates_2 = firing_rates_2[ind_2]
                     
             
              
            elif plot_b == True:
                
                mean = np.mean(value_2[b_2])

                value_high = np.where(value_2 < mean)
                low_high = np.where(value_2 >= mean)
               
                ind_st_b = np.intersect1d(value_high,b_2)
                ind_st_a = np.intersect1d(low_high,b_2)
               
                
                ind_1_b = ind_st_b[:int(len(ind_st_b)/2)]
                ind_2_b =  ind_st_b[int(len(ind_st_b)/2):]
               
                ind_1_a = ind_st_a[:int(len(ind_st_a)/2)]
                ind_2_a =  ind_st_a[int(len(ind_st_a)/2):]
              
                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))
               
                if first_half == 1:
                    rewards_2 = rewards_2[ind_1] 
                    choices_2 = choices_2[ind_1]    
                    value_2 = value_2[ind_1]
                    ones_2  = ones_2[ind_1]
                    firing_rates_2 = firing_rates_2[ind_1]
                    
                elif first_half == 2:
                    rewards_2 = rewards_2[ind_2] 
                    choices_2 = choices_2[ind_2]    
                    value_2 = value_2[ind_2]
                    ones_2  = ones_2[ind_2]
                    firing_rates_2 = firing_rates_2[ind_2]
               
            
             
           
                        
            predictors_all = OrderedDict([
                                        ('Reward', rewards_2),
                                        ('Value',value_2),                                   
                                        ('ones', ones_2)])
            
            X_2 = np.vstack(predictors_all.values()).T[:trials_2,:].astype(float)
            
            n_predictors = X_2.shape[1]
            y_2 = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            tstats,x = regression_code_session(y_2, X_2)

            C_2.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
      
        
            
            rewards_3 = reward_current[task_3]
            choices_3 = choices_current[task_3]
            
            previous_rewards_3 = scipy.linalg.toeplitz(rewards_3, np.zeros((1,n)))[n-1:-1]
             
            previous_choices_3 = scipy.linalg.toeplitz(0.5-choices_3, np.zeros((1,n)))[n-1:-1]
             
            interactions_3 = scipy.linalg.toeplitz((((0.5-choices_3)*(rewards_3-0.5))*2),np.zeros((1,n)))[n-1:-1]
             
    
            ones = np.ones(len(interactions_3)).reshape(len(interactions_3),1)
             
            X_3 = np.hstack([previous_rewards_3,previous_choices_3,interactions_3,ones])
            value_3 =np.matmul(X_3, average)
       
            rewards_3 = rewards_3[n:]
            choices_3 = choices_3[n:]
       
             
            ones_3 = np.ones(len(choices_3))
            trials_3 = len(choices_3)
    
            firing_rates_3 = firing_rates[task_3][n:]
            
        
            a_3 = np.where(choices_3 == 0.5)[0]
            b_3 = np.where(choices_3 == -0.5)[0]
            if plot_a == True:
                mean = np.mean(value_3[a_3])

                value_high = np.where(value_3 < mean)
                low_high = np.where(value_3 >= mean)
               
                ind_st_b = np.intersect1d(value_high,a_3)
                ind_st_a = np.intersect1d(low_high,a_3)
            
                
                ind_1_b = ind_st_b[:int(len(ind_st_b)/2)]
                ind_2_b =  ind_st_b[int(len(ind_st_b)/2):]
               
                ind_1_a = ind_st_a[:int(len(ind_st_a)/2)]
                ind_2_a =  ind_st_a[int(len(ind_st_a)/2):]
                
                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))
                
            
                if first_half == 1:
                    rewards_3 = rewards_3[ind_1] 
                    choices_3 = choices_3[ind_1]    
                    value_3 = value_3[ind_1]
                    ones_3  = ones_3[ind_1]
                    firing_rates_3 = firing_rates_3[ind_1]   
                    
                    
                    
                elif first_half == 2:
                  
                    rewards_3 = rewards_3[ind_2] 
                    choices_3 = choices_3[ind_2]    
                    value_3 = value_3[ind_2]
                    ones_3  = ones_3[ind_2]
                    firing_rates_3 = firing_rates_3[ind_2]   
   
              
            elif plot_b == True:
                
                mean = np.mean(value_3[b_3])

                value_high = np.where(value_3 < mean)
                low_high = np.where(value_3 >= mean)
               
                ind_st_b = np.intersect1d(value_high,b_3)
                ind_st_a = np.intersect1d(low_high,b_3)
                 
                ind_1_b = ind_st_b[:int(len(ind_st_b)/2)]
                ind_2_b =  ind_st_b[int(len(ind_st_b)/2):]
               
                ind_1_a = ind_st_a[:int(len(ind_st_a)/2)]
                ind_2_a =  ind_st_a[int(len(ind_st_a)/2):]
              
                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))
               
                if first_half == 1:
                    rewards_3 = rewards_3[ind_1] 
                    choices_3 = choices_3[ind_1]    
                    value_3 = value_3[ind_1]
                    ones_3  = ones_3[ind_1]
                    firing_rates_3 = firing_rates_3[ind_1]   
                    
                    
                    
                elif first_half == 2:
                  
                    rewards_3 = rewards_3[ind_2] 
                    choices_3 = choices_3[ind_2]    
                    value_3 = value_3[ind_2]
                    ones_3  = ones_3[ind_2]
                    firing_rates_3 = firing_rates_3[ind_2]   

             
                   
         
  
            predictors_all = OrderedDict([
                                        ('Rew', rewards_3),
                                        ('Value',value_3),                                      
                                        ('ones', ones_3)])
            
            X_3 = np.vstack(predictors_all.values()).T[:trials_3,:].astype(float)
            y_3 = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            tstats,x = regression_code_session(y_3, X_3)
    
            C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
           
    C_1 = np.concatenate(C_1,1)
    C_2 = np.concatenate(C_2,1)
    C_3 = np.concatenate(C_3,1)
    
      
   

      
    return C_1,C_2,C_3


def plot():
    real_vs_shuffle(PFC,HP, n = 11,perm_n = 5000, a_b = 0,   c = 'grey')
    real_vs_shuffle(PFC,HP, n = 11, perm_n = 5000, a_b = 1,  c = 'grey')
    real_vs_shuffle(PFC,HP, n = 11, perm_n = 5000, a_b = 2,  c = 'grey')

def real_vs_shuffle(PFC,HP, n = 11, perm_n = 2, a_b = 2,   c = 'grey'):
  
    u_v_area_shuffle, u_area_shuffle, v_area_shuffle  = svd_on_coefs(PFC,HP, n = n,task = 0, perm_n = perm_n,a_b = a_b)
    u_v_area_shuffle_1, u_area_shuffle_1, v_area_shuffle_1  = svd_on_coefs(PFC,HP, n = n, task = 1, perm_n = perm_n, a_b = a_b)
    u_v_area_shuffle_2, u_area_shuffle_2,  v_area_shuffle_2  = svd_on_coefs(PFC,HP, n = n, task = 2, perm_n = perm_n,a_b = a_b)
    u_v_area_shuffle_3, u_area_shuffle_3, v_area_shuffle_3 = svd_on_coefs(PFC,HP, n = n, task = 3, perm_n = perm_n,a_b = a_b)
   
    #all tasks
    diff_uv  = []
    diff_v = []
    diff_u = []
    for i,ii in enumerate(u_v_area_shuffle):
        diff_uv.append(u_v_area_shuffle[i][0]- u_v_area_shuffle[i][1])
        diff_v.append(v_area_shuffle[i][0]- v_area_shuffle[i][1])
        diff_u.append(u_area_shuffle[i][0]- u_area_shuffle[i][1])
    uv_95 = np.percentile(diff_uv,95)
    v95 = np.percentile(diff_v,95)
    u95 = np.percentile(diff_u,95)
 
    u_v_hp, u_hp, v_hp, within_u_hp,between_u_hp,within_v_hp,between_v_hp,within_uv_hp,between_uv_hp = real_diff(HP, n = n, a = 'HP', task = 0,a_b = a_b)
    u_v_pfc, u_pfc, v_pfc,within_u_pfc,between_u_pfc,within_v_pfc,between_v_pfc,within_uv_pfc,between_uv_pfc = real_diff(PFC, n = n, a = 'PFC', task = 0,a_b = a_b)
    
    real_uv = u_v_hp-u_v_pfc
    real_u = u_hp-u_pfc
    real_v = v_hp-v_pfc

    # task 1 2 
    diff_uv_1  = []
    diff_v_1 = []
    diff_u_1= []
    for i,ii in enumerate(u_v_area_shuffle_1):
        diff_uv_1.append(u_v_area_shuffle_1[i][0]- u_v_area_shuffle_1[i][1])
        diff_v_1.append(v_area_shuffle_1[i][0]- v_area_shuffle_1[i][1])
        diff_u_1.append(u_area_shuffle_1[i][0]- u_area_shuffle_1[i][1])
        
    uv_95_1 = np.percentile(diff_uv_1,95)
    v95_1 = np.percentile(diff_v_1,95)
    u95_1 = np.percentile(diff_u_1,95)
 
    u_v_hp_1, u_hp_1, v_hp_1,  within_u_hp_1, between_u_hp_1, within_v_hp_1, between_v_hp_1, within_uv_hp_1, between_uv_hp_1 = real_diff(HP, n = n, a = 'HP', task = 1,a_b = a_b)
    u_v_pfc_1, u_pfc_1, v_pfc_1, within_u_pfc_1,between_u_pfc_1,within_v_pfc_1, between_v_pfc_1, within_uv_pfc_1, between_uv_pfc_1 = real_diff(PFC, n = n, a = 'PFC', task = 1,a_b = a_b)
    
    real_uv_1 = u_v_hp_1-u_v_pfc_1
    real_u_1 = u_hp_1-u_pfc_1
    real_v_1 = v_hp_1-v_pfc_1


  # task 1 3 
    diff_uv_2  = []
    diff_v_2 = []
    diff_u_2 = []
    for i,ii in enumerate(u_v_area_shuffle_2):
        diff_uv_2.append(u_v_area_shuffle_2[i][0]- u_v_area_shuffle_2[i][1])
        diff_v_2.append(v_area_shuffle_2[i][0]- v_area_shuffle_2[i][1])
        diff_u_2.append(u_area_shuffle_2[i][0]- u_area_shuffle_2[i][1])
    uv_95_2 = np.percentile(diff_uv_2,95)
    v95_2 = np.percentile(diff_v_2,95)
    u95_2 = np.percentile(diff_u_2,95)
 
    u_v_hp_2, u_hp_2, v_hp_2,  within_u_hp_2, between_u_hp_2, within_v_hp_2 ,between_v_hp_2, within_uv_hp_2, between_uv_hp_2 = real_diff(HP, n = n, a = 'HP', task = 2, a_b = a_b)
    u_v_pfc_2, u_pfc_2, v_pfc_2, within_u_pfc_2, between_u_pfc_2, within_v_pfc_2, between_v_pfc_2, within_uv_pfc_2, between_uv_pfc_2  = real_diff(PFC, n = n, a = 'PFC', task = 2,a_b = a_b)
    
    real_uv_2 = u_v_hp_2-u_v_pfc_2
    real_u_2 = u_hp_2-u_pfc_2
    real_v_2 = v_hp_2-v_pfc_2

    diff_uv_3  = []
    diff_v_3 = []
    diff_u_3 = []
    for i,ii in enumerate(u_v_area_shuffle_3):
        diff_uv_3.append(u_v_area_shuffle_3[i][0]- u_v_area_shuffle_3[i][1])
        diff_v_3.append(v_area_shuffle_3[i][0]- v_area_shuffle_3[i][1])
        diff_u_3.append(u_area_shuffle_3[i][0]- u_area_shuffle_3[i][1])
        
    uv_95_3 = np.percentile(diff_uv_3,95)
    v95_3 = np.percentile(diff_v_3,95)
    u95_3 = np.percentile(diff_u_3,95)
 
    u_v_hp_3, u_hp_3, v_hp_3,  within_u_hp_3, between_u_hp_3, within_v_hp_3, between_v_hp_3, within_uv_hp_3, between_uv_hp_3 = real_diff(HP, n = n, a = 'HP', task = 3,a_b = a_b)
    u_v_pfc_3, u_pfc_3, v_pfc_3,  within_u_pfc_3, between_u_pfc_3, within_v_pfc_3, between_v_pfc_3, within_uv_pfc_3, between_uv_pfc_3  = real_diff(PFC, n = n, a = 'PFC', task = 3,a_b = a_b)
    
    real_uv_3 = u_v_hp_3-u_v_pfc_3
    real_u_3 = u_hp_3-u_pfc_3
    real_v_3 = v_hp_3-v_pfc_3
    
    within_pfc_v = [within_v_pfc_3,within_v_pfc_2,within_v_pfc_1,within_v_pfc]
    within_hp_v = [within_v_hp_3,within_v_hp_2,within_v_hp_1,within_v_hp]
    between_pfc_v = [between_v_pfc_3,between_v_pfc_2,between_v_pfc_1,between_v_pfc]
    between_hp_v = [between_v_hp_3,between_v_hp_2,between_v_hp_1,between_v_hp]
    
    within_pfc_u = [within_u_pfc_3,within_u_pfc_2,within_u_pfc_1,within_u_pfc]
    within_hp_u = [within_u_hp_3,within_u_hp_2,within_u_hp_1,within_u_hp]
    between_pfc_u = [between_u_pfc_3,between_u_pfc_2,between_u_pfc_1,between_u_pfc]
    between_hp_u = [between_u_hp_3,between_u_hp_2,between_u_hp_1,between_u_hp]
   
    
    within_pfc = [within_uv_pfc_3,within_uv_pfc_2,within_uv_pfc_1,within_uv_pfc]
    within_hp = [within_uv_hp_3,within_uv_hp_2,within_uv_hp_1,within_uv_hp]
    between_pfc = [between_uv_pfc_3,between_uv_pfc_2,between_uv_pfc_1,between_uv_pfc]
    between_hp = [between_uv_hp_3,between_uv_hp_2,between_uv_hp_1,between_uv_hp]
   
    plt.figure(figsize = (4,10))
    l = 0
    for ii,i in enumerate(within_hp):
        l+=1
        plt.subplot(4,1,l)
        plt.plot(i, label = 'Within HP', color='black')
        plt.plot(within_pfc[ii], label = 'Within PFC', color = 'green')
        
        plt.plot(between_hp[ii], label = 'Between HP', color='black',linestyle = '--')
        plt.plot(between_pfc[ii], label = 'Between PFC', color = 'green', linestyle = '--')
        sns.despine()
        
    plt.figure(figsize = (4,10))
    l = 0
    for ii,i in enumerate(within_hp_v):
        l+=1
        plt.subplot(4,1,l)
        plt.plot(i, label = 'Within HP', color='black')
        plt.plot(within_pfc_v[ii], label = 'Within PFC', color = 'green')
        
        plt.plot(between_hp_v[ii], label = 'Between HP', color='black',linestyle = '--')
        plt.plot(between_pfc_v[ii], label = 'Between PFC', color = 'green', linestyle = '--')
        sns.despine()

    plt.figure(figsize = (4,10))
    l = 0
    for ii,i in enumerate(within_hp_u):
        l+=1
        plt.subplot(4,1,l)
        plt.plot(i, label = 'Within HP', color='black')
        plt.plot(within_pfc_u[ii], label = 'Within PFC', color = 'green')
        
        plt.plot(between_hp_u[ii], label = 'Between HP', color='black',linestyle = '--')
        plt.plot(between_pfc_u[ii], label = 'Between PFC', color = 'green', linestyle = '--')
        sns.despine()
   
        
    plt.figure(figsize = (4,5))
    plt.subplot(4,1,1)
    plt.hist(diff_uv, color = 'grey')
    plt.vlines(real_uv,ymin = 0, ymax = max(np.histogram(diff_uv)[0]))
    plt.vlines(uv_95,ymin = 0, ymax = max(np.histogram(diff_uv)[0]), color = 'red')
  
    plt.subplot(4,1,2)
    plt.hist(diff_uv_1, color = 'grey')
    plt.vlines(real_uv_1,ymin = 0, ymax = max(np.histogram(diff_uv_1)[0]))
    plt.vlines(uv_95_1,ymin = 0, ymax = max(np.histogram(diff_uv)[0]), color = 'red')

    plt.subplot(4,1,3)
    plt.hist(diff_uv_2, color = 'grey')
    plt.vlines(real_uv_2,ymin = 0, ymax = max(np.histogram(diff_uv_2)[0]))
    plt.vlines(uv_95_2,ymin = 0, ymax = max(np.histogram(diff_uv)[0]), color = 'red')

    plt.subplot(4,1,4)
    plt.hist(diff_uv_3, color = 'grey')
    plt.vlines(real_uv_3,ymin = 0, ymax = max(np.histogram(diff_uv_3)[0]))
    plt.vlines(uv_95_3,ymin = 0, ymax = max(np.histogram(diff_uv)[0]), color = 'red')
 
    sns.despine()
  
    
    plt.figure(figsize = (4,5))
    plt.subplot(4,1,1)
    plt.hist(diff_v, color = 'grey')
    plt.vlines(real_v,ymin = 0, ymax = max(np.histogram(diff_v)[0]))
    plt.vlines(v95,ymin = 0, ymax = max(np.histogram(diff_v)[0]), color = 'red')
  
    plt.subplot(4,1,2)
    plt.hist(diff_v_1, color = 'grey')
    plt.vlines(real_v_1,ymin = 0, ymax = max(np.histogram(diff_v_1)[0]))
    plt.vlines(v95_1,ymin = 0, ymax = max(np.histogram(diff_v)[0]), color = 'red')

    plt.subplot(4,1,3)
    plt.hist(diff_v_2, color = 'grey')
    plt.vlines(real_v_2,ymin = 0, ymax = max(np.histogram(diff_v_2)[0]))
    plt.vlines(v95_2,ymin = 0, ymax = max(np.histogram(diff_v)[0]), color = 'red')

    plt.subplot(4,1,4)
    plt.hist(diff_v_3, color = 'grey')
    plt.vlines(real_v_3,ymin = 0, ymax = max(np.histogram(diff_v_3)[0]))
    plt.vlines(v95_3,ymin = 0, ymax = max(np.histogram(diff_v)[0]), color = 'red')
 
    sns.despine()
  
  
    
    plt.figure(figsize = (4,5))
    plt.subplot(4,1,1)
    plt.hist(diff_u, color = 'grey')
    plt.vlines(real_u,ymin = 0, ymax = max(np.histogram(diff_u)[0]))
    plt.vlines(u95,ymin = 0, ymax = max(np.histogram(diff_u)[0]), color = 'red')
  
    plt.subplot(4,1,2)
    plt.hist(diff_u_1, color = 'grey')
    plt.vlines(real_u_1,ymin = 0, ymax = max(np.histogram(diff_u_1)[0]))
    plt.vlines(u95_1,ymin = 0, ymax = max(np.histogram(diff_u)[0]), color = 'red')

    plt.subplot(4,1,3)
    plt.hist(diff_u_2, color = 'grey')
    plt.vlines(real_u_2,ymin = 0, ymax = max(np.histogram(diff_u_2)[0]))
    plt.vlines(u95_2,ymin = 0, ymax = max(np.histogram(diff_u)[0]), color = 'red')

    plt.subplot(4,1,4)
    plt.hist(diff_u_3, color = 'grey')
    plt.vlines(real_u_3,ymin = 0, ymax = max(np.histogram(diff_u_3)[0]))
    plt.vlines(u95_3,ymin = 0, ymax = max(np.histogram(diff_u)[0]), color = 'red')
 
    sns.despine()
  
def svd_on_coefs(PFC,HP, n = 11, task = 0, perm_n = 10, a_b = 0):
   
      
    
    u_v_area_shuffle = []
    u_area_shuffle = []
    v_area_shuffle = []
     
    all_subjects = np.hstack([PFC['DM'][0], HP['DM'][0]])
    all_subjects_firing = np.hstack([PFC['Data'][0], HP['Data'][0]])
    
    sessions_n = np.arange(len(all_subjects))

    for i in range(perm_n):
        np.random.shuffle(sessions_n) # Shuffle PFC/HP sessions
        indices_HP = sessions_n[PFC['DM'][0].shape[0]:]
        indices_PFC = sessions_n[:PFC['DM'][0].shape[0]]

        PFC_shuffle_dm = all_subjects[indices_PFC]
        HP_shuffle_dm = all_subjects[indices_HP]
        
        PFC_shuffle_f = all_subjects_firing[indices_PFC]
        HP_shuffle_f = all_subjects_firing[indices_HP]
       
        HP_shuffle= [HP_shuffle_dm,HP_shuffle_f]
        PFC_shuffle= [PFC_shuffle_dm,PFC_shuffle_f]
   
        u_v_area = []
        u_area = []
        v_area = []
        
        for d in [HP_shuffle,PFC_shuffle]:
        
            C_1_b_1_all, C_2_b_1_all, C_3_b_1_all = value_reg_svd(d, n = n, plot_a = False, plot_b = True,  first_half = 1, a = 'perm', perm = True, t = task)    
            C_1_a_1_all, C_2_a_1_all, C_3_a_1_all = value_reg_svd(d, n = n, plot_a = True, plot_b = False,  first_half = 1, a = 'perm', perm = True, t = task)    
            
            C_1_b_2_all, C_2_b_2_all, C_3_b_2_all = value_reg_svd(d, n = n, plot_a = False, plot_b = True,  first_half = 2, a = 'perm', perm = True, t = task)     
            C_1_a_2_all, C_2_a_2_all, C_3_a_2_all = value_reg_svd(d, n = n, plot_a = True, plot_b = False,  first_half = 2, a = 'perm', perm = True, t = task)    
             
            k = 1
            
            C_1_b_1_all = scipy.stats.zscore(C_1_b_1_all[k],0)
            C_1_a_1_all = scipy.stats.zscore(C_1_a_1_all[k],0)
         
            C_1_b_2_all = scipy.stats.zscore(C_1_b_2_all[k],0)
            C_1_a_2_all = scipy.stats.zscore(C_1_a_2_all[k],0)
         
            
            C_2_b_1_all = scipy.stats.zscore(C_2_b_1_all[k],0)
            C_2_a_1_all = scipy.stats.zscore(C_2_a_1_all[k],0)
         
            C_2_b_2_all = scipy.stats.zscore(C_2_b_2_all[k],0)
            C_2_a_2_all = scipy.stats.zscore(C_2_a_2_all[k],0)
         
            C_3_b_1_all = scipy.stats.zscore(C_3_b_1_all[k],0)
            C_3_a_1_all = scipy.stats.zscore(C_3_a_1_all[k],0)
         
            C_3_b_2_all = scipy.stats.zscore(C_3_b_2_all[k],0)
            C_3_a_2_all = scipy.stats.zscore(C_3_a_2_all[k],0)
         
            value_1_1 = (np.concatenate((C_1_b_1_all, C_1_a_1_all),1))
            value_1_2 = (np.concatenate((C_1_b_2_all, C_1_a_2_all),1))
            value_2_1 = (np.concatenate((C_2_b_1_all, C_2_a_1_all),1))
            value_2_2 = (np.concatenate((C_2_b_2_all, C_2_a_2_all),1))
        
            value_3_1 = (np.concatenate((C_3_b_1_all, C_3_a_1_all),1))
            value_3_2 = (np.concatenate((C_3_b_2_all, C_3_a_2_all),1))
            
            if a_b == 0:
 
                value_1_1 = (np.concatenate((C_1_b_1_all, C_1_a_1_all),1))
                value_1_2 = (np.concatenate((C_1_b_2_all, C_1_a_2_all),1))
                value_2_1 = (np.concatenate((C_2_b_1_all, C_2_a_1_all),1))
                value_2_2 = (np.concatenate((C_2_b_2_all, C_2_a_2_all),1))
            
                value_3_1 = (np.concatenate((C_3_b_1_all, C_3_a_1_all),1))
                value_3_2 = (np.concatenate((C_3_b_2_all, C_3_a_2_all),1))
                
            elif a_b == 1:
         
                value_1_1 = C_1_b_1_all
                value_1_2 = C_1_b_2_all
                value_2_1 = C_2_b_1_all
                value_2_2 = C_2_b_2_all
            
                value_3_1 = C_3_b_1_all
                value_3_2 = C_3_b_2_all
            
            elif a_b == 2:
         
                value_1_1 = C_1_a_1_all
                value_1_2 = C_1_a_2_all
                value_2_1 = C_2_a_1_all
                value_2_2 = C_2_a_2_all
            
                value_3_1 = C_3_a_1_all
                value_3_2 = C_3_a_2_all
         
         
            # Task 1 2
          
                   
            n_neurons = value_1_1.shape[0]
            
            u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(value_1_1, full_matrices = False)
                
            #SVDsu.shape, s.shape, vh.shape for task 1 second half
            u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(value_1_2, full_matrices = False)
            
            #SVDsu.shape, s.shape, vh.shape for task 2 first half
            u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(value_2_1, full_matrices = False)
            
            #SVDsu.shape, s.shape, vh.shape for task 2 second half
            u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(value_2_2, full_matrices = False)
            
            #SVDsu.shape, s.shape, vh.shape for task 3 first half
            u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(value_3_1, full_matrices = False)
        
            #SVDsu.shape, s.shape, vh.shape for task 3 first half
            u_t3_2, s_t3_2, vh_t3_2 = np.linalg.svd(value_3_2, full_matrices = False)
            
            #Finding variance explained in second half of task 1 using the Us and Vs from the first half
            t_u = np.transpose(u_t1_1)  
            t_v = np.transpose(vh_t1_1)  
        
            t_u_t_1_2 = np.transpose(u_t1_2)   
            t_v_t_1_2 = np.transpose(vh_t1_2)  
        
            t_u_t_2_1 = np.transpose(u_t2_1)   
            t_v_t_2_1 = np.transpose(vh_t2_1)  
        
            t_u_t_2_2 = np.transpose(u_t2_2)  
            t_v_t_2_2 = np.transpose(vh_t2_2)  
        
            t_u_t_3_2 = np.transpose(u_t3_2)
            t_v_t_3_2 = np.transpose(vh_t3_2)  
            
            t_u_t_3_1 = np.transpose(u_t3_1)
            t_v_t_3_1 = np.transpose(vh_t3_1)  

            #Compare task 1 Second Half 
            s_task_1_2 = np.linalg.multi_dot([t_u_t_1_2, value_1_1, t_v_t_1_2])
            s_1_2 = s_task_1_2.diagonal()
            sum_c_task_1_2 = np.cumsum(abs(s_1_2))/n_neurons
            
            u_only_1_within = np.linalg.multi_dot([t_u_t_1_2, value_1_1])
            u_only_1_within_sq = np.sum(u_only_1_within**2, axis = 1)
            u_only_1_within_sq_sum = np.cumsum(u_only_1_within_sq)/n_neurons
            u_only_1_within_sq_sum = u_only_1_within_sq_sum/u_only_1_within_sq_sum[-1]
           
            # Using V
            v_only_1_within = np.linalg.multi_dot([value_1_1,t_v_t_1_2])
            v_only_1_within_sq = np.sum(v_only_1_within**2, axis = 0)
            v_only_1_within_sq_sum = np.cumsum(v_only_1_within_sq)/n_neurons
            v_only_1_within_sq_sum = v_only_1_within_sq_sum/v_only_1_within_sq_sum[-1]
          
        
           
            #Compare task 2 First Half from task 1 Last Half 
            s_task_2_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, value_2_2, t_v_t_1_2])
            s_2_1_from_t_1_2 = s_task_2_1_from_t_1_2.diagonal()
            sum_c_task_2_1_from_t_1_2 = np.cumsum(abs(s_2_1_from_t_1_2))/n_neurons
            
            
            u_only_1_2_between = np.linalg.multi_dot([t_u_t_1_2, value_2_2])
            u_only_1_2_between_sq = np.sum(u_only_1_2_between**2, axis = 1)
            u_only_1_2_between_sq_sum = np.cumsum(u_only_1_2_between_sq)/n_neurons
            u_only_1_2_between_sq_sum = u_only_1_2_between_sq_sum/u_only_1_2_between_sq_sum[-1]
          
            # Using V
            v_only_1_2_between = np.linalg.multi_dot([value_2_2,t_v_t_1_2])
            v_only_1_2_between_sq = np.sum(v_only_1_2_between**2, axis = 0)
            v_only_1_2_between_sq_sum = np.cumsum(v_only_1_2_between_sq)/n_neurons
            v_only_1_2_between_sq_sum = v_only_1_2_between_sq_sum/v_only_1_2_between_sq_sum[-1]
        
              
          
            #Compare task 2 Second Half from first half
            s_task_2_2_from_t_2_1 = np.linalg.multi_dot([t_u_t_2_1, value_2_2, t_v_t_2_1])     
            s_2_2_from_t_2_1 = s_task_2_2_from_t_2_1.diagonal()
            sum_c_task_2_2_from_t_2_1 = np.cumsum(abs(s_2_2_from_t_2_1))/n_neurons
          
            u_only_2_within = np.linalg.multi_dot([t_u_t_2_1, value_2_2]) 
            u_only_2_within_sq = np.sum(u_only_2_within**2, axis = 1)
            u_only_2_within_sq_sum = np.cumsum(u_only_2_within_sq)/n_neurons
            u_only_2_within_sq_sum = u_only_2_within_sq_sum/u_only_2_within_sq_sum[-1]
            
           
            # Using V
            v_only_2_within = np.linalg.multi_dot([value_2_2,t_v_t_2_1])
            v_only_2_within_sq = np.sum(v_only_2_within**2, axis = 0)
            v_only_2_within_sq_sum = np.cumsum(v_only_2_within_sq)/n_neurons
            v_only_2_within_sq_sum = v_only_2_within_sq_sum/v_only_2_within_sq_sum[-1]
          
         
                
            #Compare task 3 First Half from Task 2 Last Half 
            s_task_3_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, value_3_2, t_v_t_2_2])
            s_3_1_from_t_2_2 = s_task_3_1_from_t_2_2.diagonal()
            sum_c_task_3_1_from_t_2_2 = np.cumsum(abs(s_3_1_from_t_2_2))/n_neurons
           
            u_only_2_3_between = np.linalg.multi_dot([t_u_t_2_2, value_3_2])
            u_only_2_3_between_sq = np.sum(u_only_2_3_between**2, axis = 1)
            u_only_2_3_between_sq_sum = np.cumsum(u_only_2_3_between_sq)/n_neurons
            u_only_2_3_between_sq_sum = u_only_2_3_between_sq_sum/u_only_2_3_between_sq_sum[-1]
          
            # Using V
            v_only_2_3_between = np.linalg.multi_dot([value_3_2,t_v_t_2_2])
            v_only_2_3_between_sq = np.sum(v_only_2_3_between**2, axis = 0)
            v_only_2_3_between_sq_sum = np.cumsum(v_only_2_3_between_sq)/n_neurons
            v_only_2_3_between_sq_sum = v_only_2_3_between_sq_sum/v_only_2_3_between_sq_sum[-1]
               
             #Compare task 3 First Half from Task  Last Half 
            s_task_3_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, value_3_2, t_v_t_1_2])
            s_3_1_from_t_1_2 = s_task_3_1_from_t_1_2.diagonal()
            sum_c_task_3_1_from_t_1_2 = np.cumsum(abs(s_3_1_from_t_1_2))/n_neurons
           
            u_only_1_3_between = np.linalg.multi_dot([t_u_t_1_2, value_3_2])
            u_only_1_3_between_sq = np.sum(u_only_1_3_between**2, axis = 1)
            u_only_1_3_between_sq_sum = np.cumsum(u_only_1_3_between_sq)/n_neurons
            u_only_1_3_between_sq_sum = u_only_1_3_between_sq_sum/u_only_1_3_between_sq_sum[-1]
          
            # Using V
            v_only_1_3_between = np.linalg.multi_dot([value_3_2,t_v_t_1_2])
            v_only_1_3_between_sq = np.sum(v_only_1_3_between**2, axis = 0)
            v_only_1_3_between_sq_sum = np.cumsum(v_only_1_3_between_sq)/n_neurons
            v_only_1_3_between_sq_sum = v_only_1_3_between_sq_sum/v_only_1_3_between_sq_sum[-1]
         
                
            s_task_3_1_from_t_3_2 = np.linalg.multi_dot([t_u_t_3_1, value_3_2, t_v_t_3_1])
            s_3_1_from_t_3_2 = s_task_3_1_from_t_3_2.diagonal()
            sum_c_task_3_1_from_t_3_2 = np.cumsum(abs(s_3_1_from_t_3_2))/n_neurons
            
           
            average_within_1_2 = sum_c_task_1_2
            average_between_1_2 = sum_c_task_2_1_from_t_1_2
            average_within_1_3 = sum_c_task_1_2
            average_between_1_3 = sum_c_task_3_1_from_t_1_2  
            average_within_2_3 = sum_c_task_2_2_from_t_2_1
            average_between_2_3  = sum_c_task_3_1_from_t_2_2
         
                    
          
            uv_2_3 = (np.trapz(average_within_2_3) - np.trapz(average_between_2_3))/average_within_2_3.shape[0]
            uv_1_3 = (np.trapz(average_within_1_3) - np.trapz(average_between_1_3))/average_within_1_3.shape[0]
            uv_1_2 = (np.trapz(average_within_1_2) - np.trapz(average_between_1_2))/average_within_1_2.shape[0]
            
            
            u_2_3 = (np.trapz(u_only_2_within_sq_sum) - np.trapz(u_only_2_3_between_sq_sum))/u_only_2_within_sq_sum.shape[0]
            u_1_3 = (np.trapz(u_only_1_within_sq_sum) - np.trapz(u_only_1_3_between_sq_sum))/u_only_1_within_sq_sum.shape[0]
            u_1_2 = (np.trapz(u_only_1_within_sq_sum) - np.trapz(u_only_1_2_between_sq_sum))/u_only_1_within_sq_sum.shape[0]
        
        
            v_2_3 = (np.trapz(v_only_2_within_sq_sum) - np.trapz(v_only_2_3_between_sq_sum))/v_only_2_within_sq_sum.shape[0]
            v_1_3 = (np.trapz(v_only_1_within_sq_sum) - np.trapz(v_only_1_3_between_sq_sum))/v_only_1_within_sq_sum.shape[0]
            v_1_2 = (np.trapz(v_only_1_within_sq_sum) - np.trapz(v_only_1_2_between_sq_sum))/v_only_1_within_sq_sum.shape[0]
           
            if task == 0:
                    
                v_area.append(np.mean([v_2_3,v_1_3,v_1_2],0))
                u_area.append(np.mean([u_2_3,u_1_3,u_1_2],0))
                u_v_area.append(np.mean([uv_2_3,uv_1_3,uv_1_2],0))
               
            elif task == 1:
                    
                v_area.append(v_1_2)
                u_area.append(u_1_2)
                u_v_area.append(uv_1_2)
              
            elif task == 2:
                    
                v_area.append(v_1_3)
                u_area.append(u_1_3)
                u_v_area.append(uv_1_3)
                
           
            elif task == 3:
                    
                v_area.append(v_2_3)
                u_area.append(u_2_3)
                u_v_area.append(uv_2_3)
              
        v_area_shuffle.append(v_area)
        u_area_shuffle.append(u_area)
        u_v_area_shuffle.append(u_v_area)
           
  
    return  u_v_area_shuffle, u_area_shuffle, v_area_shuffle

   

    
def real_diff(d, n = 11, a = 'HP', task =0, a_b = 0):

    C_1_b_1_all, C_2_b_1_all, C_3_b_1_all = value_reg_svd(d, n = n, plot_a = False, plot_b = True,  first_half = 1, a = a, perm = False, t = task)  
    C_1_a_1_all, C_2_a_1_all, C_3_a_1_all = value_reg_svd(d, n = n, plot_a = True, plot_b = False,  first_half = 1, a = a, perm = False, t = task)    
    
    C_1_b_2_all, C_2_b_2_all, C_3_b_2_all = value_reg_svd(d, n = n, plot_a = False, plot_b = True,  first_half = 2, a = a, perm = False, t = task)       
    C_1_a_2_all, C_2_a_2_all, C_3_a_2_all = value_reg_svd(d, n = n, plot_a = True, plot_b = False,  first_half = 2, a = a, perm = False, t = task)      
    
  

    k = 1
    
    C_1_b_1_all = scipy.stats.zscore(C_1_b_1_all[k],0)
    C_1_a_1_all = scipy.stats.zscore(C_1_a_1_all[k],0)
 
    C_1_b_2_all = scipy.stats.zscore(C_1_b_2_all[k],0)
    C_1_a_2_all = scipy.stats.zscore(C_1_a_2_all[k],0)
 
    
    C_2_b_1_all = scipy.stats.zscore(C_2_b_1_all[k],0)
    C_2_a_1_all = scipy.stats.zscore(C_2_a_1_all[k],0)
 
    C_2_b_2_all = scipy.stats.zscore(C_2_b_2_all[k],0)
    C_2_a_2_all = scipy.stats.zscore(C_2_a_2_all[k],0)
 
    C_3_b_1_all = scipy.stats.zscore(C_3_b_1_all[k],0)
    C_3_a_1_all = scipy.stats.zscore(C_3_a_1_all[k],0)
 
    C_3_b_2_all = scipy.stats.zscore(C_3_b_2_all[k],0)
    C_3_a_2_all = scipy.stats.zscore(C_3_a_2_all[k],0)
    
    if a_b == 0:
 
        value_1_1 = (np.concatenate((C_1_b_1_all, C_1_a_1_all),1))
        value_1_2 = (np.concatenate((C_1_b_2_all, C_1_a_2_all),1))
        value_2_1 = (np.concatenate((C_2_b_1_all, C_2_a_1_all),1))
        value_2_2 = (np.concatenate((C_2_b_2_all, C_2_a_2_all),1))
    
        value_3_1 = (np.concatenate((C_3_b_1_all, C_3_a_1_all),1))
        value_3_2 = (np.concatenate((C_3_b_2_all, C_3_a_2_all),1))
    elif a_b == 1:
 
        value_1_1 = C_1_b_1_all
        value_1_2 = C_1_b_2_all
        value_2_1 = C_2_b_1_all
        value_2_2 = C_2_b_2_all
    
        value_3_1 = C_3_b_1_all
        value_3_2 = C_3_b_2_all
    
    elif a_b == 2:
 
        value_1_1 = C_1_a_1_all
        value_1_2 = C_1_a_2_all
        value_2_1 = C_2_a_1_all
        value_2_2 = C_2_a_2_all
    
        value_3_1 = C_3_a_1_all
        value_3_2 = C_3_a_2_all
 
    n_neurons = value_1_1.shape[0]
    
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(value_1_1, full_matrices = False)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(value_1_2, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(value_2_1, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(value_2_2, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(value_3_1, full_matrices = False)

    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_2, s_t3_2, vh_t3_2 = np.linalg.svd(value_3_2, full_matrices = False)
    
    #Finding variance explained in second half of task 1 using the Us and Vs from the first half
    t_u = np.transpose(u_t1_1)  
    t_v = np.transpose(vh_t1_1)  

    t_u_t_1_2 = np.transpose(u_t1_2)   
    t_v_t_1_2 = np.transpose(vh_t1_2)  

    t_u_t_2_1 = np.transpose(u_t2_1)   
    t_v_t_2_1 = np.transpose(vh_t2_1)  

    t_u_t_2_2 = np.transpose(u_t2_2)  
    t_v_t_2_2 = np.transpose(vh_t2_2)  

    t_u_t_3_2 = np.transpose(u_t3_2)
    t_v_t_3_2 = np.transpose(vh_t3_2)  
    
    t_u_t_3_1 = np.transpose(u_t3_1)
    t_v_t_3_1 = np.transpose(vh_t3_1)  

    #Compare task 1 Second Half 
    s_task_1_2 = np.linalg.multi_dot([t_u_t_1_2, value_1_1, t_v_t_1_2])
    s_1_2 = s_task_1_2.diagonal()
    sum_c_task_1_2 = np.cumsum(abs(s_1_2))/n_neurons
    
    u_only_1_within = np.linalg.multi_dot([t_u_t_1_2, value_1_1])
    u_only_1_within_sq = np.sum(u_only_1_within**2, axis = 1)
    u_only_1_within_sq_sum = np.cumsum(u_only_1_within_sq)/n_neurons
    u_only_1_within_sq_sum = u_only_1_within_sq_sum/u_only_1_within_sq_sum[-1]
   
    # Using V
    v_only_1_within = np.linalg.multi_dot([value_1_1,t_v_t_1_2])
    v_only_1_within_sq = np.sum(v_only_1_within**2, axis = 0)
    v_only_1_within_sq_sum = np.cumsum(v_only_1_within_sq)/n_neurons
    v_only_1_within_sq_sum = v_only_1_within_sq_sum/v_only_1_within_sq_sum[-1]
  

   
    #Compare task 2 First Half from task 1 Last Half 
    s_task_2_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, value_2_2, t_v_t_1_2])
    s_2_1_from_t_1_2 = s_task_2_1_from_t_1_2.diagonal()
    sum_c_task_2_1_from_t_1_2 = np.cumsum(abs(s_2_1_from_t_1_2))/n_neurons
    
    
    u_only_1_2_between = np.linalg.multi_dot([t_u_t_1_2, value_2_2])
    u_only_1_2_between_sq = np.sum(u_only_1_2_between**2, axis = 1)
    u_only_1_2_between_sq_sum = np.cumsum(u_only_1_2_between_sq)/n_neurons
    u_only_1_2_between_sq_sum = u_only_1_2_between_sq_sum/u_only_1_2_between_sq_sum[-1]
  
    # Using V
    v_only_1_2_between = np.linalg.multi_dot([value_2_2,t_v_t_1_2])
    v_only_1_2_between_sq = np.sum(v_only_1_2_between**2, axis = 0)
    v_only_1_2_between_sq_sum = np.cumsum(v_only_1_2_between_sq)/n_neurons
    v_only_1_2_between_sq_sum = v_only_1_2_between_sq_sum/v_only_1_2_between_sq_sum[-1]

        
    #Compare task 2 Firs Half from second half
    s_task_2_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_1, value_2_2, t_v_t_2_1])    
    s_2_1_from_t_2_2 = s_task_2_1_from_t_2_2.diagonal()
    sum_c_task_2_2_from_t_2_1 = np.cumsum(abs(s_2_1_from_t_2_2))/n_neurons
   
    u_only_2_within = np.linalg.multi_dot([t_u_t_2_1, value_2_2]) 
    u_only_2_within_sq = np.sum(u_only_2_within**2, axis = 1)
    u_only_2_within_sq_sum = np.cumsum(u_only_2_within_sq)/n_neurons
    u_only_2_within_sq_sum = u_only_2_within_sq_sum/u_only_2_within_sq_sum[-1]
    
   
    # Using V
    v_only_2_within = np.linalg.multi_dot([value_2_2,t_v_t_2_1])
    v_only_2_within_sq = np.sum(v_only_2_within**2, axis = 0)
    v_only_2_within_sq_sum = np.cumsum(v_only_2_within_sq)/n_neurons
    v_only_2_within_sq_sum = v_only_2_within_sq_sum/v_only_2_within_sq_sum[-1]
  
 
        
    #Compare task 3 First Half from Task 2 Last Half 
    s_task_3_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, value_3_2, t_v_t_2_2])
    s_3_1_from_t_2_2 = s_task_3_1_from_t_2_2.diagonal()
    sum_c_task_3_1_from_t_2_2 = np.cumsum(abs(s_3_1_from_t_2_2))/n_neurons
   
    u_only_2_3_between = np.linalg.multi_dot([t_u_t_2_2, value_3_2])
    u_only_2_3_between_sq = np.sum(u_only_2_3_between**2, axis = 1)
    u_only_2_3_between_sq_sum = np.cumsum(u_only_2_3_between_sq)/n_neurons
    u_only_2_3_between_sq_sum = u_only_2_3_between_sq_sum/u_only_2_3_between_sq_sum[-1]
  
    # Using V
    v_only_2_3_between = np.linalg.multi_dot([value_3_2,t_v_t_2_2])
    v_only_2_3_between_sq = np.sum(v_only_2_3_between**2, axis = 0)
    v_only_2_3_between_sq_sum = np.cumsum(v_only_2_3_between_sq)/n_neurons
    v_only_2_3_between_sq_sum = v_only_2_3_between_sq_sum/v_only_2_3_between_sq_sum[-1]
       
     #Compare task 3 First Half from Task  Last Half 
    s_task_3_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, value_3_2, t_v_t_1_2])
    s_3_1_from_t_1_2 = s_task_3_1_from_t_1_2.diagonal()
    sum_c_task_3_1_from_t_1_2 = np.cumsum(abs(s_3_1_from_t_1_2))/n_neurons
   
    u_only_1_3_between = np.linalg.multi_dot([t_u_t_1_2, value_3_2])
    u_only_1_3_between_sq = np.sum(u_only_1_3_between**2, axis = 1)
    u_only_1_3_between_sq_sum = np.cumsum(u_only_1_3_between_sq)/n_neurons
    u_only_1_3_between_sq_sum = u_only_1_3_between_sq_sum/u_only_1_3_between_sq_sum[-1]
  
    # Using V
    v_only_1_3_between = np.linalg.multi_dot([value_3_2,t_v_t_1_2])
    v_only_1_3_between_sq = np.sum(v_only_1_3_between**2, axis = 0)
    v_only_1_3_between_sq_sum = np.cumsum(v_only_1_3_between_sq)/n_neurons
    v_only_1_3_between_sq_sum = v_only_1_3_between_sq_sum/v_only_1_3_between_sq_sum[-1]
 
        
    s_task_3_1_from_t_3_2 = np.linalg.multi_dot([t_u_t_3_1, value_3_2, t_v_t_3_1])
    s_3_1_from_t_3_2 = s_task_3_1_from_t_3_2.diagonal()
    sum_c_task_3_1_from_t_3_2 = np.cumsum(abs(s_3_1_from_t_3_2))/n_neurons
        
   
    average_within_1_2 = sum_c_task_1_2
    average_between_1_2 = sum_c_task_2_1_from_t_1_2
    average_within_1_3 = sum_c_task_1_2
    average_between_1_3 = sum_c_task_3_1_from_t_1_2  
    average_within_2_3 = sum_c_task_2_2_from_t_2_1
    average_between_2_3  = sum_c_task_3_1_from_t_2_2
 
            
  
    uv_2_3 = (np.trapz(average_within_2_3) - np.trapz(average_between_2_3))/average_within_2_3.shape[0]
    uv_1_3 = (np.trapz(average_within_1_3) - np.trapz(average_between_1_3))/average_within_1_3.shape[0]
    uv_1_2 = (np.trapz(average_within_1_2) - np.trapz(average_between_1_2))/average_within_1_2.shape[0]
    
    
    u_2_3 = (np.trapz(u_only_2_within_sq_sum) - np.trapz(u_only_2_3_between_sq_sum))/u_only_2_within_sq_sum.shape[0]
    u_1_3 = (np.trapz(u_only_1_within_sq_sum) - np.trapz(u_only_1_3_between_sq_sum))/u_only_1_within_sq_sum.shape[0]
    u_1_2 = (np.trapz(u_only_1_within_sq_sum) - np.trapz(u_only_1_2_between_sq_sum))/u_only_1_within_sq_sum.shape[0]


    v_2_3 = (np.trapz(v_only_2_within_sq_sum) - np.trapz(v_only_2_3_between_sq_sum))/v_only_2_within_sq_sum.shape[0]
    v_1_3 = (np.trapz(v_only_1_within_sq_sum) - np.trapz(v_only_1_3_between_sq_sum))/v_only_1_within_sq_sum.shape[0]
    v_1_2 = (np.trapz(v_only_1_within_sq_sum) - np.trapz(v_only_1_2_between_sq_sum))/v_only_1_within_sq_sum.shape[0]
   
    if task == 0:
            
        v_area = np.mean([v_2_3,v_1_3,v_1_2],0)
        u_area = np.mean([u_2_3,u_1_3,u_1_2],0)
        u_v_area = np.mean([uv_2_3,uv_1_3,uv_1_2],0)
        
        within_u = np.mean([u_only_2_within_sq_sum,u_only_1_within_sq_sum],0)
        between_u = np.mean([u_only_2_3_between_sq_sum,u_only_1_3_between_sq_sum,u_only_1_2_between_sq_sum],0)
        
        within_v = np.mean([v_only_2_within_sq_sum,v_only_1_within_sq_sum],0)
        between_v = np.mean([v_only_2_3_between_sq_sum,v_only_1_3_between_sq_sum,v_only_1_2_between_sq_sum],0)
    
        within_uv =  np.mean([sum_c_task_1_2, sum_c_task_2_2_from_t_2_1, sum_c_task_3_1_from_t_3_2], axis = 0)
        between_uv =np.mean([sum_c_task_2_1_from_t_1_2, sum_c_task_3_1_from_t_2_2, sum_c_task_3_1_from_t_1_2], axis = 0)
   
    elif task == 1:
            
        v_area = v_1_2
        u_area = u_1_2
        u_v_area = uv_1_2
        within_u = u_only_1_within_sq_sum
        between_u = u_only_1_2_between_sq_sum
        within_v = v_only_1_within_sq_sum
        between_v = v_only_1_2_between_sq_sum
        within_uv =  average_within_1_2
        between_uv = average_between_1_2
   
    
    elif task == 2:
            
        v_area = v_1_3
        u_area = u_1_3
        u_v_area = uv_1_3
        within_u = u_only_1_within_sq_sum
        between_u = u_only_1_3_between_sq_sum
        within_v = v_only_1_within_sq_sum
        between_v = v_only_1_3_between_sq_sum
        within_uv =  average_within_1_3
        between_uv = average_between_1_3
   
    elif task == 3:
            
        v_area = v_2_3
        u_area = u_2_3
        u_v_area = uv_2_3
        
        within_u = u_only_2_within_sq_sum
        between_u = u_only_2_3_between_sq_sum
        within_v = v_only_2_within_sq_sum
        between_v = v_only_2_3_between_sq_sum
        within_uv =  average_within_2_3
        between_uv = average_between_2_3
   
    return u_v_area, u_area,v_area,within_u,between_u,within_v,between_v,within_uv,between_uv

