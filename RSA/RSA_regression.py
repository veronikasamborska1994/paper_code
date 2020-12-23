#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:34:17 2020

@author: veronikasamborska
"""

import numpy as np
import pylab as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from collections import OrderedDict
from palettable import wesanderson as wes
from scipy import io
import RSAs as rs


# ind_init = 25
# ind_choice = 36
# ind_reward = 43

def load():
    
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    
    # HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    # PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
 
     
def _cpd(X,y):
    
    '''Evaluate coefficient of partial determination for each predictor in X'''
    
    pdes = np.linalg.pinv(X)
    pe = np.matmul(pdes,y)
  
    Y_predict = np.matmul(X,pe)
    sse = np.sum((Y_predict - y)**2, axis=0)

    #sse = np.sum((ols.predict(X) - y)**2, axis=0)
    cpd = np.zeros([X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        pdes_i = np.linalg.pinv(X_i)
        pe_i = np.matmul(pdes_i,y)

        Y_predict_i = np.matmul(X_i,pe_i)
        sse_X_i = np.sum((Y_predict_i- y)**2, axis=0)

        #sse_X_i = np.sum((ols.predict(X_i) - y)**2, axis=0)
        cpd[i]=(sse_X_i-sse)/sse_X_i
    return cpd


def _CPD(X,y):
    '''Evaluate coefficient of partial determination for each predictor in X'''
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y)
    sse = np.sum((ols.predict(X) - y)**2, axis=0)
    cpd = np.zeros([y.shape[1],X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        ols.fit(X_i,y)
        sse_X_i = np.sum((ols.predict(X_i) - y)**2, axis=0)
        cpd[:,i]=(sse_X_i-sse)/sse_X_i
    return cpd


   
def task_ind(task, a_pokes, b_pokes):
    
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task));
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid
 
def extract_trials(data, t_start, t_end, perm_n = False): 
     
   if perm_n:
        dm = data[0]
        firing = data[1]
        
   else:
      dm = data['DM'][0]
      firing = data['Data'][0]
   
   neurons = 0
   for s in firing:
       neurons += s.shape[1]
       
    

   a_1_nr = np.zeros((neurons)); a_2_nr = np.zeros((neurons)); a_3_nr = np.zeros((neurons))
   a_1_r = np.zeros((neurons)); a_2_r = np.zeros((neurons)); a_3_r = np.zeros((neurons))

   b_1_nr = np.zeros((neurons)); b_2_nr = np.zeros((neurons)); b_3_nr = np.zeros((neurons))
   b_1_r = np.zeros((neurons)); b_2_r = np.zeros((neurons)); b_3_r = np.zeros((neurons))

   i_1 = np.zeros((neurons)); i_2 = np.zeros((neurons)); i_3 = np.zeros((neurons))

   n_neurons_cum = 0
   init_start = 24
   init_stop = 26 
   

   for  s, sess in enumerate(dm):
        
       
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        n_neurons_cum += n_neurons

        choices = DM[:,1]
        task =  DM[:,5]
        a_pokes = DM[:,6]
        b_pokes = DM[:,7]
        reward = DM[:,2]  
 
        taskid = task_ind(task, a_pokes, b_pokes)
       
        task_1 = np.where(taskid == 1)[0]
        task_2 = np.where(taskid == 2)[0]
        task_3 = np.where(taskid == 3)[0]
        
        task_1_a_r = np.where((choices == 1) & (reward == 1) & (taskid == 1))[0]                              
        task_2_a_r = np.where((choices == 1) & (reward == 1) & (taskid == 2))[0]            
        task_3_a_r = np.where((choices == 1) & (reward == 1) & (taskid == 3))[0]    

        task_1_a_nr = np.where((choices == 1) & (reward == 0) & (taskid == 1))[0]                                 
        task_2_a_nr = np.where((choices == 1) & (reward == 0) & (taskid == 2))[0]            
        task_3_a_nr = np.where((choices == 1) & (reward == 0) & (taskid == 3))[0]    
        
        task_1_b_r = np.where((choices == 0) & (reward == 1) & (taskid == 1))[0]                                 
        task_2_b_r = np.where((choices == 0) & (reward == 1) & (taskid == 2))[0]          
        task_3_b_r = np.where((choices == 0) & (reward == 1) & (taskid == 3))[0]       
        
        task_1_b_nr = np.where((choices == 0) & (reward == 0) & (taskid == 1))[0]                                 
        task_2_b_nr = np.where((choices == 0) & (reward == 0) & (taskid == 2))[0]            
        task_3_b_nr = np.where((choices == 0) & (reward == 0) & (taskid == 3))[0]    

        
        a_1_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_1_a_r,:, t_start:t_end],2),0)
        a_2_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_2_a_r,:, t_start:t_end],2),0)
        a_3_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_3_a_r,:, t_start:t_end],2),0)
         
        a_1_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_1_a_nr,:, t_start:t_end],2),0)
        a_2_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_2_a_nr,:, t_start:t_end],2),0)
        a_3_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_3_a_nr,:, t_start:t_end],2),0)
 
        b_1_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_1_b_r,:, t_start:t_end],2),0)
        b_2_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_2_b_r,:, t_start:t_end],2),0)
        b_3_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_3_b_r,:, t_start:t_end],2),0)
         
        b_1_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_1_b_nr,:, t_start:t_end],2),0)
        b_2_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_2_b_nr,:, t_start:t_end],2),0)
        b_3_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_3_b_nr,:, t_start:t_end],2),0)
        
        i_1[n_neurons_cum-n_neurons:n_neurons_cum] = np.mean(np.mean(firing_rates[task_1,:, init_start:init_stop],2),0)
        i_2[n_neurons_cum-n_neurons:n_neurons_cum] = np.mean(np.mean(firing_rates[task_2,:, init_start:init_stop],2),0)
        i_3[n_neurons_cum-n_neurons:n_neurons_cum] = np.mean(np.mean(firing_rates[task_3,:, init_start:init_stop],2),0)

   matrix_for_correlations = np.vstack([a_1_r,a_1_nr,a_2_r,a_2_nr,a_3_r,a_3_nr,\
                                        i_1,i_3,i_2,b_3_r,b_3_nr, b_2_r,b_2_nr, b_1_r,b_1_nr])
   return matrix_for_correlations

    
def simple_regression(data, perm = 1000):
    
  
    dm = data['DM'][0]
    firing = data['Data'][0]
    cpd_perm  = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
  
    cpd  = []
    
    for  s, sess in enumerate(dm):
        
       
        DM = dm[s]
        firing_rates = firing[s]
        task =  DM[:,5]
       
        task_1 = len(np.where(task == 1)[0])
        task_2 = len(np.where(task == 2)[0])
        task_3 = len(np.where(task == 3)[0])
        task_min = np.min([task_1,task_2,task_3])
        
       

     
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        choices = DM[:,1]
        reward = DM[:,2]  

        choices_current = choices-0.5
        ones = np.ones(len(choices_current)).reshape(len(choices_current),1)
         
        rew_ch = choices_current*reward
        ones = np.ones(len(rew_ch))
        predictors_all = OrderedDict([
                                    ('Choice', choices_current),
                                    ('Reward', reward),
                                    ('Rew Ch', rew_ch),
                                    ('ones', ones)
                                    ])
        X = np.vstack(predictors_all.values()).T.astype(float)
      
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
    
        cpd.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
        
        for i in range(perm):
            y_perm = np.roll(y,np.random.randint(task_min), axis = 0)
            cpd_perm[i].append(_CPD(X,y_perm).reshape(n_neurons, n_timepoints, n_predictors))
    
    cpd_perm   = np.stack([np.concatenate(cpd_i,0) for cpd_i in cpd_perm],0)
    cpd = np.concatenate(cpd,0)
    
    return cpd,cpd_perm


def perm_sessions_GLM(HP,PFC, perm = 1000):
    
    
  
    dm_HP = HP['DM'][0]
    firing_HP = HP['Data'][0]
  
    cpd_HP  = []
    cpd_PFC  = []
    
 
    for  s, sess in enumerate(dm_HP):
        
       
        DM = dm_HP[s]
        firing_rates = firing_HP[s]         
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        choices = DM[:,1]
        reward = DM[:,2]  

        choices_current = choices-0.5
        ones = np.ones(len(choices_current)).reshape(len(choices_current),1)
         
        rew_ch = choices_current*reward
        ones = np.ones(len(rew_ch))
        predictors_all = OrderedDict([
                                    ('Choice', choices_current),
                                    ('Reward', reward),
                                    ('Rew Ch', rew_ch),
                                    ('ones', ones)
                                    ])
        X = np.vstack(predictors_all.values()).T.astype(float)
      
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
    
        cpd_HP.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
        
        
    dm_PFC = PFC['DM'][0]
    firing_PFC = PFC['Data'][0]
    
     
 
    for  s, sess in enumerate(dm_PFC):
        
       
        DM = dm_PFC[s]
        firing_rates = firing_PFC[s]
       
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        choices = DM[:,1]
        reward = DM[:,2]  

        choices_current = choices-0.5
        ones = np.ones(len(choices_current)).reshape(len(choices_current),1)
         
        rew_ch = choices_current*reward
        ones = np.ones(len(rew_ch))
        predictors_all = OrderedDict([
                                    ('Choice', choices_current),
                                    ('Reward', reward),
                                    ('Rew Ch', rew_ch),
                                    ('ones', ones)
                                    ])
        X = np.vstack(predictors_all.values()).T.astype(float)
      
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
    
        cpd_PFC.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))

    cpd_PFC = np.concatenate(cpd_PFC,0)
    cpd_HP = np.concatenate(cpd_HP,0)

    
    if perm:
        
        all_subjects_DM = np.concatenate((HP['DM'][0],PFC['DM'][0]),0)     
        all_subjects_fr = np.concatenate((HP['Data'][0],PFC['Data'][0]),0)     
        nn = 0
        diff_perm = np.zeros((int(perm),63,len(predictors_all)))
        
        n_sessions = np.arange(len(HP['DM'][0])+len(PFC['DM'][0]))
   

        for i in range(perm):
            np.random.shuffle(n_sessions) # Shuffle PFC/HP sessions
            indices_HP = n_sessions[:len(HP['DM'][0])]
            indices_PFC = n_sessions[len(HP['DM'][0]):]
    
           
            DM_PFC_perm = all_subjects_DM[np.asarray(indices_PFC)]
            firing_PFC_perm = all_subjects_fr[np.asarray(indices_PFC)]
            
            cpd_PFC_perm = []
            for  s, sess in enumerate(DM_PFC_perm):
        
       
                DM = DM_PFC_perm[s]
                firing_rates = firing_PFC_perm[s]
               

                n_trials, n_neurons, n_timepoints = firing_rates.shape
                
                choices = DM[:,1]
                reward = DM[:,2]  
        
                choices_current = choices-0.5
                ones = np.ones(len(choices_current)).reshape(len(choices_current),1)
                 
                rew_ch = choices_current*reward
                ones = np.ones(len(rew_ch))
                predictors_all = OrderedDict([
                                            ('Choice', choices_current),
                                            ('Reward', reward),
                                            ('Rew Ch', rew_ch),
                                            ('ones', ones)
                                            ])
                X = np.vstack(predictors_all.values()).T.astype(float)
              
                n_predictors = X.shape[1]
                y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            
                cpd_PFC_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
              

            DM_HP_perm = all_subjects_DM[np.asarray(indices_HP)]
            firing_HP_perm = all_subjects_fr[np.asarray(indices_HP)]
            cpd_HP_perm = []
            for  s, sess in enumerate(DM_HP_perm):
        
       
                DM = DM_HP_perm[s]
                firing_rates = firing_HP_perm[s]
               

                n_trials, n_neurons, n_timepoints = firing_rates.shape
                
                choices = DM[:,1]
                reward = DM[:,2]  
        
                choices_current = choices-0.5
                ones = np.ones(len(choices_current)).reshape(len(choices_current),1)
                 
                rew_ch = choices_current*reward
                ones = np.ones(len(rew_ch))
                predictors_all = OrderedDict([
                                            ('Choice', choices_current),
                                            ('Reward', reward),
                                            ('Rew Ch', rew_ch),
                                            ('ones', ones)
                                            ])
                X = np.vstack(predictors_all.values()).T.astype(float)
              
                n_predictors = X.shape[1]
                y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            
                cpd_HP_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
            cpd_HP_perm = np.concatenate(cpd_HP_perm,0)
            cpd_PFC_perm = np.concatenate(cpd_PFC_perm,0)
            diff_perm[nn,:] = abs(np.mean(cpd_PFC_perm,0) - np.mean(cpd_HP_perm,0))
            nn += 1
            
    p = np.percentile(diff_perm,95, axis = 0)
    real_diff = np.abs(np.mean(cpd_PFC,0) - np.mean(cpd_HP,0))
    
    return p, real_diff,diff_perm

                     
def run_perm_cpd(HP, PFC):
    
    cpd_HP,cpd_perm_HP =  simple_regression(HP, perm = 5000)
    cpd_PFC,cpd_perm_PFC =  simple_regression(PFC, perm = 5000)
    p, real_diff, distribution = perm_sessions_GLM(HP,PFC, perm = 5000)
    
    time_controlled = np.max(p,0)
    indicies = np.where(real_diff > time_controlled)
    
    time_ms = np.asarray([0,   40,   80,  120,  160,  200,  240,  280,  320, 360,  400,  440,  480,  520,  560,  600,  640,  680,
               720,  760,  800,  840,  880,  920,  960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400,
               1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080, 2120,
               2160, 2200, 2240, 2280, 2320, 2360, 2400, 2440, 2480])

 
    c = wes.Royal2_5.mpl_colors

    cpd_HP_m = np.nanmean(cpd_HP,0)#*100
    cpd_PFC_m = np.nanmean(cpd_PFC,0)#*100
    
    plt.figure(figsize = (10,4))
    t = np.arange(0,63)
    cpds = [cpd_HP_m,cpd_PFC_m]
    cpds_perms = [cpd_perm_HP,cpd_perm_PFC]
  

    for i,cpd in enumerate(cpds):

        cpd = cpds[i][:,:-1]
        cpd_perm = cpds_perms[i][:,:,:,:-1]
        values_95 = np.max(np.percentile(np.mean(cpd_perm,1),95,0),0)
        array_pvals = np.ones((cpd.shape[0],cpd.shape[1]))
        for l in range(cpd.shape[1]):
            array_pvals[(np.where(cpd[:,l] > values_95[l])[0]),i] = 0.05
     
        p = ['Choice','Reward', 'Reward x Choice']
        
        for k in np.arange(cpd.shape[1]):
            ymax = np.max([np.max(cpd_HP_m[:,k]),np.max(cpd_PFC_m[:,k])])+0.001
            for l in range(cpd.shape[1]):
                array_pvals[(np.where(cpd[:,l] > values_95[l])[0]),l] = 0.05
     
            plt.subplot(1,3,k+1)
            plt.plot(cpd[:,k], color = c[i])
            p_vals = array_pvals[:,k]
            t05 = t[p_vals == 0.05]
            index = indicies[0][np.where(indicies[1] == k)[0]]  
            print(index)
            for s in index:
                plt.annotate('*',xy = [s, ymax])  
   
            if i == 0:
                plt.plot(t05, np.ones(t05.shape)*ymax+0.002, '.', markersize=3, color = c[i])
            else: 
                plt.plot(t05, (np.ones(t05.shape)*ymax)+0.003, '.', markersize=3, color = c[i])
 
            plt.vlines([25,35,42], 0, np.max(cpd[:,k]), color= 'grey', linestyle = '--', alpha = 0.5)
            plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    
            sns.despine()
            
    
      

def regression_RSA(HP, PFC, t_start, t_end, perm_n = 2):
    
   
    matrix_for_correlations_HP = extract_trials(HP, t_start, t_end)
    correlation_m = np.corrcoef(matrix_for_correlations_HP)
    np.fill_diagonal(correlation_m,np.nan)
    correlation_m_f= correlation_m.flatten()
    correlation_m_f = correlation_m_f[~np.isnan(correlation_m_f)]
    
    physical_rsa = np.asarray(rs.RSA_physical_rdm(),dtype=np.float32)
    np.fill_diagonal(physical_rsa,np.nan)
    physical_rsa  = physical_rsa.flatten()
    physical_rsa = physical_rsa[~np.isnan(physical_rsa)]
  
    choice_ab_rsa = np.asarray(rs.RSA_a_b_initiation_rdm(),dtype=np.float32)
    np.fill_diagonal(choice_ab_rsa,np.nan)
    choice_ab_rsa = choice_ab_rsa.flatten()
    choice_ab_rsa = choice_ab_rsa[~np.isnan(choice_ab_rsa)]
 
    
    reward_no_reward = np.asarray(rs.reward_rdm(),dtype=np.float32)
    np.fill_diagonal(reward_no_reward,np.nan)
    reward_no_reward = reward_no_reward.flatten()
    reward_no_reward = reward_no_reward[~np.isnan(reward_no_reward)]
 
    reward_at_choices = rs.reward_choice_space()
    reward_at_a =  np.asarray(reward_at_choices,dtype=np.float32)
    reward_at_b =  np.asarray(reward_at_choices,dtype=np.float32)
    reward_at_a[6:,6:] = 0
    reward_at_b[:6,:6] = 0
    np.fill_diagonal(reward_at_a,np.nan)
    np.fill_diagonal(reward_at_b,np.nan)
    reward_at_a = reward_at_a.flatten()
    reward_at_b = reward_at_b.flatten()
    reward_at_b = reward_at_b[~np.isnan(reward_at_b)]
    reward_at_a = reward_at_a[~np.isnan(reward_at_a)]
        
    choice_initiation_rsa =  np.asarray(rs.choice_vs_initiation(),dtype=np.float)
    np.fill_diagonal(choice_initiation_rsa,np.nan)
    choice_initiation_rsa = choice_initiation_rsa.flatten()
    choice_initiation_rsa = choice_initiation_rsa[~np.isnan(choice_initiation_rsa)]
   
    a_bs_task_specific_rsa = np.asarray(rs.a_bs_task_specific(),dtype=np.float)
    
    a_specific =  np.asarray(a_bs_task_specific_rsa,dtype=np.float32)
    b_specific =  np.asarray(a_bs_task_specific_rsa,dtype=np.float32)
    a_specific[6:,6:] = 0
    b_specific[:6,:6] = 0
    
    np.fill_diagonal(a_specific,np.nan)
    a_specific = a_specific.flatten()
    a_specific = a_specific[~np.isnan(a_specific)]

    np.fill_diagonal(b_specific,np.nan)
    b_specific = b_specific.flatten()
    b_specific = b_specific[~np.isnan(b_specific)]

    ones = np.ones(len(choice_ab_rsa))
    

    predictors = OrderedDict([('Space' , physical_rsa),
                              ('A vs B', choice_ab_rsa),
                              ('Reward',reward_no_reward),
                            #  ('Reward at A B ',reward_at_choices),
                              ('Reward at A ',reward_at_a),
                              ('Reward at B ',reward_at_b),
                              ('Choice vs Initiation',choice_initiation_rsa),
                              ('A Task Specific',a_specific),
                              ('B Task Specific',b_specific),

                              ('constant', ones)])                                        
           
    X = np.vstack(predictors.values()).T[:len(physical_rsa),:].astype(float)
    # Check if regression is rank deficient 
    print(X.shape[1])  
    rank = np.linalg.matrix_rank(X) 
    print(rank)
    y = correlation_m_f
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y)
    cpd_HP = _cpd(X,y)
    C_HP = ols.coef_
    
    
    matrix_for_correlations_PFC = extract_trials(PFC, t_start, t_end)
    correlation_m_PFC = np.corrcoef(matrix_for_correlations_PFC)
    np.fill_diagonal(correlation_m_PFC,np.nan)
    correlation_m_PFC_f= correlation_m_PFC.flatten()
    correlation_m_PFC_f = correlation_m_PFC_f[~np.isnan(correlation_m_PFC_f)]
    
    y = correlation_m_PFC_f
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y)
    cpd_PFC = _cpd(X,y)
    C_PFC = ols.coef_
   
    
    if perm_n:
        
        all_subjects_DM = np.concatenate((HP['DM'][0],PFC['DM'][0]),0)     
        all_subjects_fr = np.concatenate((HP['Data'][0],PFC['Data'][0]),0)     
        nn = 0
        diff_perm = np.zeros((int(perm_n),len(predictors)))
        diff_perm_C = np.zeros((int(perm_n),len(predictors)))
        
        n_sessions = np.arange(len(HP['DM'][0])+len(PFC['DM'][0]))
   

        for i in range(perm_n):
            np.random.shuffle(n_sessions) # Shuffle PFC/HP sessions
            indices_HP = n_sessions[:len(HP['DM'][0])]
            indices_PFC = n_sessions[len(HP['DM'][0]):]
    
           
            DM_PFC = all_subjects_DM[np.asarray(indices_PFC)]
            Data_PFC = all_subjects_fr[np.asarray(indices_PFC)]
            data_PFC = [DM_PFC,Data_PFC]

            DM_HP = all_subjects_DM[np.asarray(indices_HP)]
            Data_HP = all_subjects_fr[np.asarray(indices_HP)]
            data_HP = [DM_HP,Data_HP]

            matrix_for_correlations_perm_PFC = extract_trials(data_PFC, t_start, t_end, perm_n  = perm_n)
            
            matrix_for_correlations_perm_PFC = np.corrcoef(matrix_for_correlations_perm_PFC)
            np.fill_diagonal(matrix_for_correlations_perm_PFC,np.nan)
            matrix_for_correlations_perm_PFC= matrix_for_correlations_perm_PFC.flatten()
            y_perm_PFC = matrix_for_correlations_perm_PFC[~np.isnan(matrix_for_correlations_perm_PFC)]
    
            ols.fit(X,y_perm_PFC)
            cpd_PFC_perm = _cpd(X,y_perm_PFC)
            C_PFC_perm = ols.coef_
 
            matrix_for_correlations_perm_HP = extract_trials(data_HP, t_start, t_end, perm_n  = perm_n)
            matrix_for_correlations_perm_HP  = np.corrcoef(matrix_for_correlations_perm_HP)
            np.fill_diagonal(matrix_for_correlations_perm_HP,np.nan)
            matrix_for_correlations_perm_HP= matrix_for_correlations_perm_HP.flatten()
            y_perm_HP = matrix_for_correlations_perm_HP[~np.isnan(matrix_for_correlations_perm_HP)]
    
            ols.fit(X,y_perm_HP)
            cpd_HP_perm = _cpd(X,y_perm_HP)
            C_HP_perm = ols.coef_

            
            diff_perm[nn,:] = abs(cpd_PFC_perm - cpd_HP_perm)
            diff_perm_C[nn,:] = abs(C_PFC_perm - C_HP_perm)

            nn += 1
    


    p = np.percentile(diff_perm,95, axis = 0)
    p_C = np.percentile(diff_perm_C,95, axis = 0)
       
    return cpd_PFC, C_PFC,cpd_HP, C_HP, matrix_for_correlations_PFC,matrix_for_correlations_HP,p,p_C


def across_time(HP, PFC):
    cue = np.arange(34,62)
    reward =np.arange(35,63)
    perm_n = 1000
    isl = wes.Royal2_5.mpl_colors

    ## Flipping the signs of Coefficients
    C_PFC_list = []
    C_HP_list = []
    corr_HP = []
    corr_PFC = []
    p_value_list = []
    p_C_list = []
    
    cpd_PFC_list =[]
    cpd_HP_list =[]
      
    for start,end in zip(cue, reward):
  
        cpd_PFC, C_PFC,cpd_HP, C_HP, matrix_for_correlations_PFC,matrix_for_correlations_HP,p,p_C = regression_RSA(HP,PFC, start, end, perm_n = perm_n)       
 
        C_PFC_list.append(C_PFC)
        C_HP_list.append(C_HP)
        
        
        cpd_PFC_list.append(cpd_PFC)
        cpd_HP_list.append(cpd_HP)
      
        
        p_value_list.append(p)
        p_C_list.append(p_C)
        
        corr_HP.append(matrix_for_correlations_HP)
        corr_PFC.append(matrix_for_correlations_PFC)
        
    C_p_value_multiple_comparisons = np.max(p_C_list,0)
    cpd_p_value_multiple_comparisons = np.max(p_value_list,0)
   
    space_HP  = []; space_PFC  = []
    a_b_HP = []; a_b_PFC = []
    
    rew_HP = []; rew_PFC = []
    
    rew_a_HP = []; rew_a_PFC = []
    rew_b_HP = []; rew_b_PFC = []

    ch_init_HP = []; ch_init_PFC = []
    a_spec_HP = []; a_spec_PFC = [];
    b_spec_HP = []; b_spec_PFC = [];
    
    
    space_HP_cpd = []; space_PFC_cpd  = []
    a_b_HP_cpd = []; a_b_PFC_cpd = []
    
    rew_HP_cpd = []; rew_PFC_cpd = []
    
    rew_a_HP_cpd = []; rew_a_PFC_cpd = []
    rew_b_HP_cpd = []; rew_b_PFC_cpd = []

    ch_init_HP_cpd = []; ch_init_PFC_cpd = []
    a_spec_HP_cpd = []; a_spec_PFC_cpd = [];
    b_spec_HP_cpd = []; b_spec_PFC_cpd = [];

    difference_space = []; difference_a_b = []
    difference_rew = []; difference_rew_a = []
    difference_rew_b = []; difference_ch_init = []
    difference_a_spec = []; difference_b_spec = []
    
    difference_space_C = []; difference_a_b_C = []
    difference_rew_C = []; difference_rew_a_C = []
    difference_rew_b_C = []; difference_ch_init_C = []
    difference_a_spec_C = []; difference_b_spec_C = []



    for i,ii in enumerate(C_PFC_list):
        space_HP.append(C_HP_list[i][0]); space_PFC.append(C_PFC_list[i][0])
        
        a_b_HP.append(C_HP_list[i][1]); a_b_PFC.append(C_PFC_list[i][1])
        
        rew_HP.append(C_HP_list[i][2]); rew_PFC.append(C_PFC_list[i][2])
      
        rew_a_HP.append(C_HP_list[i][3]); rew_a_PFC.append(C_PFC_list[i][3])
        
        rew_b_HP.append(C_HP_list[i][4]); rew_b_PFC.append(C_PFC_list[i][4])
        
        ch_init_HP.append(C_HP_list[i][5]); ch_init_PFC.append(C_PFC_list[i][5])
        
        a_spec_HP.append(C_HP_list[i][6]); a_spec_PFC.append(C_PFC_list[i][6])
        
        b_spec_HP.append(C_HP_list[i][7]); b_spec_PFC.append(C_PFC_list[i][7])
    
        #####
        space_HP_cpd.append(cpd_HP_list[i][0]); space_PFC_cpd.append(cpd_PFC_list[i][0])
        
        a_b_HP_cpd.append(cpd_HP_list[i][1]);  a_b_PFC_cpd.append(cpd_PFC_list[i][1])
        
        rew_HP_cpd.append(cpd_HP_list[i][2]); rew_PFC_cpd.append(cpd_PFC_list[i][2])
      
        rew_a_HP_cpd.append(cpd_HP_list[i][3]); rew_a_PFC_cpd.append(cpd_PFC_list[i][3])
        
        rew_b_HP_cpd.append(cpd_HP_list[i][4]); rew_b_PFC_cpd.append(cpd_PFC_list[i][4])
        
        ch_init_HP_cpd.append(cpd_HP_list[i][5]); ch_init_PFC_cpd.append(cpd_PFC_list[i][5])
        
        a_spec_HP_cpd.append(cpd_HP_list[i][6]); a_spec_PFC_cpd.append(cpd_PFC_list[i][6])
        
        b_spec_HP_cpd.append(cpd_HP_list[i][7]); b_spec_PFC_cpd.append(cpd_PFC_list[i][7])
      
        difference_space_C.append((abs(C_HP_list[i][0]- C_PFC_list[i][0]))); difference_a_b_C.append((abs(C_HP_list[i][1]- C_PFC_list[i][1])))
        difference_rew_C.append((abs(C_HP_list[i][2]- C_PFC_list[i][2]))); difference_rew_a_C.append((abs(C_HP_list[i][3]- C_PFC_list[i][3])))
        difference_rew_b_C.append((abs(C_HP_list[i][4]- C_PFC_list[i][4]))); difference_ch_init_C.append((abs(C_HP_list[i][5]- C_PFC_list[i][5])))
        difference_a_spec_C.append((abs(C_HP_list[i][6]- C_PFC_list[i][6]))); difference_b_spec_C.append((abs(C_HP_list[i][6]- C_PFC_list[i][6])))

        
        difference_space.append((abs(cpd_HP_list[i][0]- cpd_PFC_list[i][0]))); difference_a_b.append((abs(cpd_HP_list[i][1]- cpd_PFC_list[i][1])))
        difference_rew.append((abs(cpd_HP_list[i][2]- cpd_PFC_list[i][2]))); difference_rew_a.append((abs(cpd_HP_list[i][3]- cpd_PFC_list[i][3])))
        difference_rew_b.append((abs(cpd_HP_list[i][4]- cpd_PFC_list[i][4]))); difference_ch_init.append((abs(cpd_HP_list[i][5]- cpd_PFC_list[i][5])))
        difference_a_spec.append((abs(cpd_HP_list[i][6]- cpd_PFC_list[i][6]))); difference_b_spec.append((abs(cpd_HP_list[i][6]- cpd_PFC_list[i][6])))


    space_sig = np.where(difference_space >= cpd_p_value_multiple_comparisons[0])[0]
    a_b_sig = np.where(difference_a_b >= cpd_p_value_multiple_comparisons[1])[0]
    rew_sig = np.where(difference_rew >= cpd_p_value_multiple_comparisons[2])[0]
    rew_a_sig = np.where(difference_rew_a >= cpd_p_value_multiple_comparisons[3])[0]
    rew_b_sig = np.where(difference_rew_b >= cpd_p_value_multiple_comparisons[4])[0]
    
    
    ch_init_sig = np.where(difference_ch_init >= cpd_p_value_multiple_comparisons[5])[0]
    a_spec_sig = np.where(difference_a_spec >= cpd_p_value_multiple_comparisons[6])[0]
    b_spec_sig = np.where(difference_b_spec >= cpd_p_value_multiple_comparisons[7])[0]
    
    
    space_sig_C = np.where(difference_space >= C_p_value_multiple_comparisons[0])[0]
    a_b_sig_C = np.where(difference_a_b >= C_p_value_multiple_comparisons[1])[0]
    rew_sig_C = np.where(difference_rew >= C_p_value_multiple_comparisons[2])[0]
    rew_a_sig_C = np.where(difference_rew_a >= C_p_value_multiple_comparisons[3])[0]
    rew_b_sig_C = np.where(difference_rew_b >= C_p_value_multiple_comparisons[4])[0]
    
    
    ch_init_sig_C = np.where(difference_ch_init >= C_p_value_multiple_comparisons[5])[0]
    a_spec_sig_C = np.where(difference_a_spec >= C_p_value_multiple_comparisons[6])[0]
    b_spec_sig_C = np.where(difference_b_spec >= C_p_value_multiple_comparisons[7])[0]

    
    plt.figure(figsize = (10,2))
    plt.subplot(1,8,1)
    plt.plot(np.asarray(space_HP), color = isl[0])
    plt.plot(np.asarray(space_PFC), color = isl[3])
   # plt.hlines(0,0,len(cue), color = 'grey', alpha = 0.5)
    plt.xticks([2,9],['C','R'])
    for s in space_sig_C:
        plt.annotate('*',xy = [s, (np.max([space_HP,space_PFC])+0.01)])  
    plt.ylim((np.min([space_HP,space_PFC])-0.03),(np.max([space_HP,space_PFC])+0.03))
    plt.ylabel('Coefficient')  
    plt.title('Space')
    
    plt.subplot(1,8,2)
    plt.plot(np.asarray(a_b_HP), color = isl[0])
    plt.plot(np.asarray(a_b_PFC), color = isl[3])
   # plt.hlines(0,0,len(cue), color = 'grey', alpha = 0.5)  
    for s in a_b_sig_C:
        plt.annotate('*',xy = [s, (np.max([a_b_HP,a_b_PFC])+0.01)])
    plt.ylim((np.min([a_b_HP,a_b_PFC])-0.03),(np.max([a_b_HP,a_b_PFC])+0.03))
    plt.title('Choice A vs B')
    plt.xticks([2,9],['C','R'])
  
        
    plt.subplot(1,8,3)
    plt.plot(np.asarray(rew_HP), color = isl[0])
    plt.plot(np.asarray(rew_PFC), color = isl[3])
   # plt.hlines(0,0,len(cue), color = 'grey', alpha = 0.5)
    for s in rew_sig_C:
        plt.annotate('*',xy = [s, (np.max([rew_HP,rew_PFC])+0.01)])
    plt.ylim((np.min([rew_HP,rew_PFC])-0.03),(np.max([rew_HP,rew_PFC])+0.03))
    plt.title('Reward')
    plt.xticks([2,9],['C','R'])
     
    plt.subplot(1,8,4)
    plt.plot(np.asarray(rew_a_HP), color = isl[0])
    plt.plot(np.asarray(rew_a_PFC), color = isl[3])
   # plt.hlines(0,0,len(cue), color = 'grey', alpha = 0.5)
    for s in rew_a_sig_C:
        plt.annotate('*',xy = [s, (np.max([rew_a_HP,rew_a_PFC])+0.01)]) 
    plt.ylim((np.min([rew_a_HP,rew_a_PFC])-0.03),(np.max([rew_a_HP,rew_a_PFC])+0.03))    
    plt.title('Reward at A')
    plt.xticks([2,9],['C','R'])
  
    plt.subplot(1,8,5)
    plt.plot(np.asarray(rew_b_HP), color = isl[0])
    plt.plot(np.asarray(rew_b_PFC), color = isl[3])
    #plt.hlines(0,0,len(cue), color = 'grey', alpha = 0.5) 
    for s in rew_b_sig_C:
        plt.annotate('*',xy = [s, (np.max([rew_b_HP,rew_b_PFC])+0.01)]) 
    plt.ylim((np.min([rew_b_HP,rew_b_PFC])-0.03),(np.max([rew_b_HP,rew_b_PFC])+0.03))
    plt.title('Reward at B')
    plt.xticks([2,9],['C','R'])
 
    plt.subplot(1,8,6)
    plt.plot(np.asarray(ch_init_HP), color = isl[0])
    plt.plot(np.asarray(ch_init_PFC), color = isl[3])
    #plt.hlines(0,0,len(cue), color = 'grey', alpha = 0.5)
    for s in ch_init_sig_C:
        plt.annotate('*',xy = [s, (np.max([ch_init_HP,ch_init_PFC])+0.01)])
    plt.ylim((np.min([ch_init_HP,ch_init_PFC])-0.03),(np.max([ch_init_HP,ch_init_PFC])+0.03)) 
    plt.title('Choice vs Initiation')
    plt.xticks([2,9],['C','R'])

    plt.subplot(1,8,7)
    plt.plot(np.asarray(a_spec_HP), color = isl[0])
    plt.plot(np.asarray(a_spec_PFC), color = isl[3])
   # plt.hlines(0,0,len(cue), color = 'grey', alpha = 0.5)
    plt.title('A specific')
    for s in a_spec_sig_C:
        plt.annotate('*',xy = [s, (np.max([a_spec_HP,a_spec_PFC])+0.01)])
    plt.ylim((np.min([a_spec_HP,a_spec_PFC])-0.03),(np.max([a_spec_HP,a_spec_PFC])+0.03))  
    plt.xticks([2,9],['C','R'])

    plt.subplot(1,8,8)
    plt.plot(np.asarray(b_spec_HP), color = isl[0])
    plt.plot(np.asarray(b_spec_PFC), color = isl[3])
    #plt.hlines(0,0,len(cue), color = 'grey', alpha = 0.5)
    for s in b_spec_sig_C:
        plt.annotate('*',xy = [s, (np.max([b_spec_HP,b_spec_PFC])+0.01)])
    plt.ylim((np.min([b_spec_HP,b_spec_PFC])-0.03),(np.max([b_spec_HP,b_spec_PFC])+0.03))
    plt.title('B specific')
    plt.xticks([2,9],['C','R'])

  
    plt.tight_layout()
    sns.despine()
    
    
    plt.figure(figsize = (10,2))
    plt.subplot(1,8,1)
    plt.plot(np.asarray(space_HP_cpd)*100, color = isl[0])
    plt.plot(np.asarray(space_PFC_cpd)*100, color = isl[3])
    plt.xticks([2,9],['C','R'])
    for s in space_sig:
        plt.annotate('*',xy = [s, (np.max([space_HP_cpd,space_PFC_cpd])+0.01)*100])  
    plt.ylim(-0.1,(np.max([space_HP_cpd,space_PFC_cpd])+0.03)*100)
    plt.ylabel('CPD')  
    plt.title('Space')
    
    plt.subplot(1,8,2)
    plt.plot(np.asarray(a_b_HP_cpd)*100, color = isl[0])
    plt.plot(np.asarray(a_b_PFC_cpd)*100, color = isl[3])
    for s in a_b_sig:
        plt.annotate('*',xy = [s, (np.max([a_b_HP_cpd,a_b_PFC_cpd])+0.01)*100])
    plt.ylim(-0.1,(np.max([a_b_HP_cpd,a_b_PFC_cpd])+0.03)*100)
    plt.title('Choice A vs B')
    plt.xticks([2,9],['C','R'])
  
        
    plt.subplot(1,8,3)
    plt.plot(np.asarray(rew_HP_cpd)*100,color = isl[0])
    plt.plot(np.asarray(rew_PFC_cpd)*100, color = isl[3])
    for s in rew_sig:
        plt.annotate('*',xy = [s, (np.max([rew_HP_cpd,rew_PFC_cpd])+0.01)*100])
    plt.ylim(-0.1,(np.max([rew_HP_cpd,rew_PFC_cpd])+0.03)*100)
    plt.title('Reward')
    plt.xticks([2,9],['C','R'])
     
    plt.subplot(1,8,4)
    plt.plot(np.asarray(rew_a_HP_cpd)*100, color = isl[0])
    plt.plot(np.asarray(rew_a_PFC_cpd)*100, color = isl[3])
    for s in rew_a_sig:
        plt.annotate('*',xy = [s, (np.max([rew_a_HP_cpd,rew_a_PFC_cpd])+0.01)*100]) 
    plt.ylim(-0.1,(np.max([rew_a_HP_cpd,rew_a_PFC_cpd])+0.03)*100)    
    plt.title('Reward at A')
    plt.xticks([2,9],['C','R'])
  
    plt.subplot(1,8,5)
    plt.plot(np.asarray(rew_b_HP_cpd)*100, color = isl[0])
    plt.plot(np.asarray(rew_b_PFC_cpd)*100, color = isl[3])
    for s in rew_b_sig:
        plt.annotate('*',xy = [s, (np.max([rew_b_HP_cpd,rew_b_PFC_cpd])+0.01)*100]) 
    plt.ylim(-0.1,(np.max([rew_b_HP_cpd,rew_b_PFC_cpd])+0.03)*100)
    plt.title('Reward at B')
    plt.xticks([2,9],['C','R'])
 
    plt.subplot(1,8,6)
    plt.plot(np.asarray(ch_init_HP_cpd)*100, color = isl[0])
    plt.plot(np.asarray(ch_init_PFC_cpd)*100, color = isl[3])
    for s in ch_init_sig:
        plt.annotate('*',xy = [s, (np.max([ch_init_HP_cpd,ch_init_PFC_cpd])+0.01)*100])
    plt.ylim(-0.1,(np.max([ch_init_HP_cpd,ch_init_PFC_cpd])+0.03)*100) 
    plt.title('Choice vs Initiation')
    plt.xticks([2,9],['C','R'])

    plt.subplot(1,8,7)
    plt.plot(np.asarray(a_spec_HP_cpd)*100, color = isl[0])
    plt.plot(np.asarray(a_spec_PFC_cpd)*100, color = isl[3])
    plt.title('A specific')
    for s in a_spec_sig:
        plt.annotate('*',xy = [s, (np.max([a_spec_HP_cpd,a_spec_PFC_cpd])+0.01)*100])
    plt.ylim(-0.1,(np.max([a_spec_HP_cpd,a_spec_PFC_cpd])+0.03)*100)  
    plt.xticks([2,9],['C','R'])

    plt.subplot(1,8,8)
    plt.plot(np.asarray(b_spec_HP_cpd)*100, color = isl[0])
    plt.plot(np.asarray(b_spec_PFC_cpd)*100, color = isl[3])
    for s in b_spec_sig:
        plt.annotate('*',xy = [s, (np.max([b_spec_HP_cpd,b_spec_PFC_cpd])+0.01)*100])
    plt.ylim(-0.1,(np.max([b_spec_HP_cpd,b_spec_PFC_cpd])+0.03)*100)
    plt.title('B specific')
    plt.xticks([2,9],['C','R'])

  
    plt.tight_layout()
    sns.despine()

     
    
    ## Figure out negative correlations
    
    c_HP = corr_HP[10]
    c_PFC = corr_PFC[10]
    all_corr = [np.corrcoef(c_HP)-np.mean(np.corrcoef(c_HP)), np.corrcoef(c_PFC)-np.mean(np.corrcoef(c_PFC))]
    for i, ii in enumerate(all_corr):
        plt.subplot(1,2,i+1)
        plt.imshow(ii, aspect = 1,cmap = cmap)
        plt.xticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                           '1 A T3 R','1 A T3 NR', ' 2 I T1',\
                           '2 I T2', '3 I T3', '3 B T1 R',\
                           '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'), rotation = 'vertical')
        plt.colorbar()
               
        plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                           '1 A T3 R','1 A T3 NR',' 2 I T1',\
                           '2 I T2', '3 I T3', '3 B T1 R',\
                           '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
        plt.tight_layout()
        
    physical_rsa = np.asarray(rs.RSA_physical_rdm(),dtype=np.float32)
    choice_ab_rsa = np.asarray(rs.RSA_a_b_initiation_rdm(),dtype=np.float32)   
    reward_no_reward = np.asarray(rs.reward_rdm(),dtype=np.float32)
    reward_at_choices = rs.reward_choice_space()
    reward_at_a =  np.asarray(reward_at_choices,dtype=np.float32)
    reward_at_b =  np.asarray(reward_at_choices,dtype=np.float32)
    reward_at_a[6:,6:] = 0
    reward_at_b[:6,:6] = 0
    choice_initiation_rsa =  np.asarray(rs.choice_vs_initiation(),dtype=np.float)   
    a_bs_task_specific_rsa = np.asarray(rs.a_bs_task_specific(),dtype=np.float)
    a_specific =  np.asarray(a_bs_task_specific_rsa,dtype=np.float32)
    a_specific[6:,6:]= 0
    
    all_rsas =[physical_rsa,choice_ab_rsa,reward_no_reward,reward_at_a,reward_at_b,choice_initiation_rsa,a_specific]
    for i,ii in enumerate(all_rsas):
        
        plt.subplot(1,7,i+1)
        plt.imshow(ii,cmap = cmap)
   
    plt.tight_layout()
    
    

        
