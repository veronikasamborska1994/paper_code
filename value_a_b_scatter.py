#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 12:21:05 2020

@author: veronikasamborska
"""


#!/usr/bin/env python3
"""
Created on Wed Oct  7 17:25:08 2020

X_1.@author: veronikasamborska
"""


from sklearn.linear_model import LogisticRegression

import numpy as np
import pylab as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from collections import OrderedDict
from palettable import wesanderson as wes
from scipy import io
from itertools import combinations 
import scipy
import palettable
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)


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



def plot_correlations(PFC, HP, n = 11, c_1 = 1):
    n = 11
    init_t = 25
    ch_t = 36
    r_t = 42
    c_1 = 1
    
    C_1_HP_b, C_2_HP_b, C_3_HP_b = time_in_block(HP, area = 'HP', n = n, plot_a = False, plot_b = True, perm = False)
    C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = time_in_block(PFC, area = 'PFC', n = n, plot_a = False, plot_b = True, perm = False)
    
    C_1_HP_a, C_2_HP_a, C_3_HP_a = time_in_block(HP, area = 'HP', n = n, plot_a = True, plot_b = False, perm = False)
    C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = time_in_block(PFC, area = 'PFC', n = n, plot_a = True, plot_b = False, perm = False)
    
    C_1_HP = [C_1_HP_a,C_1_HP_b]; C_2_HP = [C_2_HP_a,C_2_HP_b]; C_3_HP = [C_3_HP_a,C_3_HP_b]
    C_1_PFC = [C_1_PFC_a,C_1_PFC_b]; C_2_PFC = [C_2_PFC_a,C_2_PFC_b]; C_3_PFC = [C_3_PFC_a,C_3_PFC_b]
    task_mean  = []
    isl = wes.Royal2_5.mpl_colors

    for i,ii in enumerate(C_1_HP):
            
        scatter_HP_1 = C_1_HP[i][c_1,:];     scatter_PFC_1  = C_1_PFC[i][c_1,:]  
        scatter_HP_2 = C_2_HP[i][c_1,:];   scatter_PFC_2 = C_2_PFC[i][c_1,:]   
        scatter_HP_3 = C_3_HP[i][c_1,:];    scatter_PFC_3 = C_3_PFC[i][c_1,:]


        scatter_HP_1_init = C_1_HP[i][c_1,:,init_t];     scatter_PFC_1_init  = C_1_PFC[i][c_1,:,init_t]
        scatter_HP_2_init = C_2_HP[i][c_1,:,init_t];   scatter_PFC_2_init= C_2_PFC[i][c_1,:,init_t]   
        scatter_HP_3_init = C_3_HP[i][c_1,:,init_t];    scatter_PFC_3_init = C_3_PFC[i][c_1,:,init_t]


        scatter_HP_1_ch = C_1_HP[i][c_1,:,ch_t];     scatter_PFC_1_ch  = C_1_PFC[i][c_1,:,ch_t]     
        scatter_HP_2_ch = C_2_HP[i][c_1,:,ch_t];   scatter_PFC_2_ch = C_2_PFC[i][c_1,:,ch_t]  
        scatter_HP_3_ch = C_3_HP[i][c_1,:,ch_t];    scatter_PFC_3_ch = C_3_PFC[i][c_1,:,ch_t]

        
        scatter_HP_1_rew = C_1_HP[i][c_1,:,r_t];     scatter_PFC_1_rew  = C_1_PFC[i][c_1,:,r_t]
        scatter_HP_2_rew = C_2_HP[i][c_1,:,r_t];   scatter_PFC_2_rew = C_2_PFC[i][c_1,:,r_t]
        scatter_HP_3_rew = C_3_HP[i][c_1,:,r_t];    scatter_PFC_3_rew = C_3_PFC[i][c_1,:,r_t]

        reg_x = [scatter_HP_1_init,scatter_HP_2_init,scatter_HP_3_init,\
                 scatter_HP_1_ch,scatter_HP_2_ch,scatter_HP_3_ch,\
                 scatter_HP_1_rew,scatter_HP_2_rew,scatter_HP_3_rew,\
                 scatter_PFC_1_init,scatter_PFC_2_init,scatter_PFC_3_init,\
                 scatter_PFC_1_ch,scatter_PFC_2_ch,scatter_PFC_3_ch,\
                 scatter_PFC_1_rew,scatter_PFC_2_rew,scatter_PFC_3_rew]
                        
           
        reg_y = [scatter_HP_2,scatter_HP_3,scatter_HP_1,\
                 scatter_HP_2,scatter_HP_3,scatter_HP_1,\
                 scatter_HP_2,scatter_HP_3,scatter_HP_1,\
                 scatter_PFC_2,scatter_PFC_3,scatter_PFC_1,\
                 scatter_PFC_2,scatter_PFC_3,scatter_PFC_1,\
                 scatter_PFC_2,scatter_PFC_3,scatter_PFC_1]
            
        coefs = []
        for r,reg in enumerate(reg_y):
            y = reg_y[r]
            x = reg_x[r]#.reshape(len(reg_x[r]),1)
            coef = []
            for yy in y.T:
                coef.append(np.corrcoef(yy,x)[0][1])
            # ols = LinearRegression()
            # ols.fit(x,y)
            coefs.append(coef) # Predictor loadings            
        task_mean.append([np.mean(coefs[:3],0),np.mean(coefs[3:6],0), np.mean(coefs[6:9],0),np.mean(coefs[9:12],0),np.mean(coefs[12:15],0), np.mean(coefs[15:18],0)])
   
       
         
    dff_PFC_HP_perm, pfc_hp_perm = perumute_sessions(HP, PFC, c_1 = 1, n = n, perm_n = 5000, init_t = init_t, ch_t = ch_t, r_t = r_t)
    dff_PFC_HP_perm = np.asarray(dff_PFC_HP_perm)
    pfc_hp_perm  = np.asarray(pfc_hp_perm)
    _95th = np.percentile(dff_PFC_HP_perm,95,0)
    
    # A>B in CA1
    b_a = np.abs(np.asarray(task_mean[1])- np.asarray(task_mean[0]))
    dff_PFC_HP = np.asarray(b_a[:3])- np.asarray(b_a[3:])
    indx = np.where(dff_PFC_HP>np.max(_95th))
    
    # A vs B
    _95th_a_b = np.percentile(pfc_hp_perm,95,0)
    
    #real a/b diff
    task_mean = np.asarray(task_mean)
    a_b_real = task_mean[:,3:,:] - task_mean[:,:3,:]
    significant_diff_a = np.where(a_b_real[0]>np.max(_95th_a_b[0]))
    significant_diff_b = np.where(a_b_real[1]>np.max(_95th_a_b[1]))
    max_ind = np.max(task_mean)

    plt.figure(figsize = (10,3))
    fig = 0
    c = isl[0]    
    for i in task_mean[0]:
        fig +=1 

        if fig >3:
            fig-=3
            c = isl[3] 
        plt.subplot(1,3,fig)
        plt.plot(i, color = c)
        plt.ylim(np.min(task_mean)-0.03,np.max(task_mean)+0.07)
       
        p = indx[1][np.where(indx[0] == fig-1)[0]]
        p_a = significant_diff_a[1][np.where(significant_diff_a[0] ==fig-1)[0]]
        p_b = significant_diff_b[1][np.where(significant_diff_b[0] ==fig-1)[0]]

        plt.plot(p, np.ones(len(p))*max_ind+0.02, '.', markersize=3, color= 'grey')
        plt.plot(p_a, np.ones(len(p_a))*max_ind+0.04, '.', markersize=3, color= 'pink')
        plt.plot(p_b, np.ones(len(p_b))*max_ind+0.05, '.', markersize=3, color= 'black')

        sns.despine()
   
         
    
    fig = 0
    c = isl[0]    
    #plt.figure()
    for i in task_mean[1]:
        fig +=1 
        if fig >3:
            fig-=3
            c = isl[3] 

        plt.subplot(1,3,fig)
        plt.plot(i, color = c, linestyle  ='--')
        plt.ylim(np.min(task_mean)-0.03,np.max(task_mean)+0.07)
        plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
        sns.despine()


 
     



def perumute_sessions(HP, PFC, c_1 = 1, n = 6, perm_n = 500, init_t = 25, ch_t = 36, r_t = 42):
    
 
    C_1_HP_b, C_2_HP_b, C_3_HP_b = time_in_block(HP, area = 'HP', n = n, plot_a = False, plot_b = True, perm = False)
    C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = time_in_block(PFC, area = 'PFC', n = n, plot_a = False, plot_b = True, perm = False)
    
    C_1_HP_a, C_2_HP_a, C_3_HP_a = time_in_block(HP, area = 'HP', n = n, plot_a = True, plot_b = False, perm = False)
    C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = time_in_block(PFC, area = 'PFC', n = n, plot_a = True, plot_b = False, perm = False)
 
    all_subjects = np.hstack([PFC['DM'][0], HP['DM'][0]])
    all_subjects_firing = np.hstack([PFC['Data'][0], HP['Data'][0]])
    
    sessions_n = np.arange(len(all_subjects))
    pfc_hp = [] 
    dff_PFC_HP_perm = []
    for i in range(perm_n):
        np.random.shuffle(sessions_n) # Shuffle PFC/HP sessions
        indices_HP = sessions_n[PFC['DM'][0].shape[0]:]
        indices_PFC = sessions_n[:PFC['DM'][0].shape[0]]

        PFC_shuffle_dm = all_subjects[np.asarray(indices_PFC)]
        HP_shuffle_dm = all_subjects[np.asarray(indices_HP)]
        
        PFC_shuffle_f = all_subjects_firing[np.asarray(indices_PFC)]
        HP_shuffle_f = all_subjects_firing[np.asarray(indices_HP)]
       
        HP_shuffle= [HP_shuffle_dm,HP_shuffle_f]
        PFC_shuffle= [PFC_shuffle_dm,PFC_shuffle_f]

        C_1_HP_b, C_2_HP_b, C_3_HP_b = time_in_block(HP_shuffle, area = 'HP', n = n, plot_a = False, plot_b = True,  perm = True)
        C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = time_in_block(PFC_shuffle, area = 'PFC', n = n, plot_a = False, plot_b = True,  perm = True)
        
        C_1_HP_a, C_2_HP_a, C_3_HP_a = time_in_block(HP_shuffle, area = 'HP', n = n, plot_a = True, plot_b = False,  perm = True)
        C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = time_in_block(PFC_shuffle, area = 'PFC', n = n, plot_a = True, plot_b = False ,  perm = True)
        
        task_mean = []
        C_1_HP = [C_1_HP_a,C_1_HP_b]; C_2_HP = [C_2_HP_a,C_2_HP_b]; C_3_HP = [C_3_HP_a,C_3_HP_b]
        C_1_PFC = [C_1_PFC_a,C_1_PFC_b]; C_2_PFC = [C_2_PFC_a,C_2_PFC_b]; C_3_PFC = [C_3_PFC_a,C_3_PFC_b]
        task_mean  = []
    
        for i,ii in enumerate(C_1_HP):
                
            scatter_HP_1 = C_1_HP[i][c_1,:];     scatter_PFC_1  = C_1_PFC[i][c_1,:]  
            scatter_HP_2 = C_2_HP[i][c_1,:];   scatter_PFC_2 = C_2_PFC[i][c_1,:]   
            scatter_HP_3 = C_3_HP[i][c_1,:];    scatter_PFC_3 = C_3_PFC[i][c_1,:]
    
    
            scatter_HP_1_init = C_1_HP[i][c_1,:,init_t];     scatter_PFC_1_init  = C_1_PFC[i][c_1,:,init_t]
            scatter_HP_2_init = C_2_HP[i][c_1,:,init_t];   scatter_PFC_2_init= C_2_PFC[i][c_1,:,init_t]   
            scatter_HP_3_init = C_3_HP[i][c_1,:,init_t];    scatter_PFC_3_init = C_3_PFC[i][c_1,:,init_t]
    
    
            scatter_HP_1_ch = C_1_HP[i][c_1,:,ch_t];     scatter_PFC_1_ch  = C_1_PFC[i][c_1,:,ch_t]     
            scatter_HP_2_ch = C_2_HP[i][c_1,:,ch_t];   scatter_PFC_2_ch = C_2_PFC[i][c_1,:,ch_t]  
            scatter_HP_3_ch = C_3_HP[i][c_1,:,ch_t];    scatter_PFC_3_ch = C_3_PFC[i][c_1,:,ch_t]
    
            
            scatter_HP_1_rew = C_1_HP[i][c_1,:,r_t];     scatter_PFC_1_rew  = C_1_PFC[i][c_1,:,r_t]
            scatter_HP_2_rew = C_2_HP[i][c_1,:,r_t];   scatter_PFC_2_rew = C_2_PFC[i][c_1,:,r_t]
            scatter_HP_3_rew = C_3_HP[i][c_1,:,r_t];    scatter_PFC_3_rew = C_3_PFC[i][c_1,:,r_t]
    
            reg_x = [scatter_HP_1_init,scatter_HP_2_init,scatter_HP_3_init,\
                     scatter_HP_1_ch,scatter_HP_2_ch,scatter_HP_3_ch,\
                     scatter_HP_1_rew,scatter_HP_2_rew,scatter_HP_3_rew,\
                     scatter_PFC_1_init,scatter_PFC_2_init,scatter_PFC_3_init,\
                     scatter_PFC_1_ch,scatter_PFC_2_ch,scatter_PFC_3_ch,\
                     scatter_PFC_1_rew,scatter_PFC_2_rew,scatter_PFC_3_rew]
                            
               
            reg_y = [scatter_HP_2,scatter_HP_3,scatter_HP_1,\
                     scatter_HP_2,scatter_HP_3,scatter_HP_1,\
                     scatter_HP_2,scatter_HP_3,scatter_HP_1,\
                     scatter_PFC_2,scatter_PFC_3,scatter_PFC_1,\
                     scatter_PFC_2,scatter_PFC_3,scatter_PFC_1,\
                     scatter_PFC_2,scatter_PFC_3,scatter_PFC_1]
                
            coefs = []
            for r,reg in enumerate(reg_y):
                y = reg_y[r]
                x = reg_x[r]#.reshape(len(reg_x[r]),1)
                coef = []
                for yy in y.T:
                    coef.append(np.corrcoef(yy,x)[0][1])
               
                coefs.append(coef) # Predictor loadings            
            task_mean.append([np.mean(coefs[:3],0),np.mean(coefs[3:6],0), np.mean(coefs[6:9],0),np.mean(coefs[9:12],0),np.mean(coefs[12:15],0), np.mean(coefs[15:18],0)])

        b_a = np.asarray(task_mean[1])- np.asarray(task_mean[0])
        
        dff_PFC_HP = []
        for i,ii in enumerate(b_a):
           if i <3:
               dff_PFC_HP.append(b_a[i]-b_a[i+3])
               
        dff_PFC_HP_perm.append(dff_PFC_HP)
        pfc_hp.append((np.asarray(task_mean)[:,3:]- np.asarray(task_mean)[:,:3]))
        
    return dff_PFC_HP_perm,pfc_hp


    
def time_in_block(data, area = 'PFC', n = 10, plot_a = False, plot_b = False, perm = True):
   
    if perm:
        dm = data[0]
        firing = data[1]

    else:
        dm = data['DM'][0]
        firing = data['Data'][0]

    C_1 = []; C_2 = []; C_3 = []
    average = rew_prev_behaviour(data, n = n, perm = perm)

    for  s, sess in enumerate(dm):
        
       
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        choices = DM[:,1]
        reward = DM[:,2]  

        task =  DM[:,5]
        a_pokes = DM[:,6]
        b_pokes = DM[:,7]
        
        taskid = task_ind(task, a_pokes, b_pokes)
      
        
        task_1 = np.where(taskid == 1)[0]
        task_2 = np.where(taskid == 2)[0]
        task_3 = np.where(taskid == 3)[0]

        reward_current = reward
        choices_current = choices-0.5

       
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
        
       
      
        firing_rates_1 = firing_rates[task_1][n:]
        
        a_1 = np.where(choices_1 == 0.5)[0]
        b_1 = np.where(choices_1 == -0.5)[0]
        
        if plot_a == True:
            rewards_1 = rewards_1[a_1] 
            choices_1 = choices_1[a_1]    
            value_1 = value_1[a_1]
            ones_1  = ones_1[a_1]
            firing_rates_1 = firing_rates_1[a_1]
          
        elif plot_b == True:
            
            rewards_1 = rewards_1[b_1] 
            choices_1 = choices_1[b_1]
            value_1 = value_1[b_1]
            ones_1  = ones_1[b_1]
            firing_rates_1 = firing_rates_1[b_1]
         
        predictors_all = OrderedDict([
                                    ('Reward', rewards_1),
                                    ('Value',value_1), 
                                   ('ones', ones_1)
                                    ])
        
        X_1 = np.vstack(predictors_all.values()).T[:trials_1,:].astype(float)
        
        n_predictors = X_1.shape[1]
        y_1 = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoint
        ols = LinearRegression()
        ols.fit(X_1,y_1)
        C_1.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
         
        
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
            rewards_2 = rewards_2[a_2] 
            choices_2 = choices_2[a_2]
            value_2 = value_2[a_2]
            ones_2  = ones_2[a_2]
            firing_rates_2 = firing_rates_2[a_2]
          
        elif plot_b == True:
            
            rewards_2 = rewards_2[b_2] 
            choices_2 = choices_2[b_2]
            value_2 = value_2[b_2]
            ones_2  = ones_2[b_2]
            firing_rates_2 = firing_rates_2[b_2]
          

        predictors_all = OrderedDict([
                                    ('Reward', rewards_2),
                                    ('Value',value_2),                                 
                                    ('ones', ones_2)
                                    ])
        
        X_2 = np.vstack(predictors_all.values()).T[:trials_2,:].astype(float)
        
        n_predictors = X_2.shape[1]
        y_2 = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        ols = LinearRegression()
        ols.fit(X_2,y_2)
        C_2.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
  
    
        
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
            rewards_3 = rewards_3[a_3] 
            choices_3 = choices_3[a_3]
            value_3 = value_3[a_3]
            ones_3  = ones_3[a_3]

            firing_rates_3 = firing_rates_3[a_3]
      
           
        elif plot_b == True:
            rewards_3 = rewards_3[b_3] 
            choices_3 = choices_3[b_3]
          
            value_3 = value_3[b_3]
            ones_3  = ones_3[b_3]

            firing_rates_3 = firing_rates_3[b_3]
      
        predictors_all = OrderedDict([
                                    ('Rew', rewards_3),
                                    ('Value',value_3),                
                                    ('ones', ones_3)
                                    ])
        
        X_3 = np.vstack(predictors_all.values()).T[:trials_3,:].astype(float)
        y_3 = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        ols = LinearRegression()
        ols.fit(X_3,y_3)
        C_3.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
 
    
    C_1 = np.concatenate(C_1,0)
    
    C_2 = np.concatenate(C_2,0)
    
    C_3 = np.concatenate(C_3,0)
   
    C_2_inf = [~np.isinf(C_2[0]).any(axis=1)]; C_2_nan = [~np.isnan(C_2[0]).any(axis=1)]
    C_3_inf = [~np.isinf(C_3[0]).any(axis=1)];  C_3_nan = [~np.isnan(C_3[0]).any(axis=1)]
    C_1_inf = [~np.isinf(C_1[0]).any(axis=1)];  C_1_nan = [~np.isnan(C_1[0]).any(axis=1)]
    nans = np.asarray(C_1_inf) & np.asarray(C_1_nan) & np.asarray(C_3_inf) & np.asarray(C_3_nan) & np.asarray(C_2_inf)& np.asarray(C_2_nan)
    C_1 = np.transpose(C_1[:,nans[0],:],[2,0,1]); C_2 = np.transpose(C_2[:,nans[0],:],[2,0,1]);  C_3 = np.transpose(C_3[:,nans[0],:],[2,0,1])
   
   
        
   
  
    return C_1,C_2,C_3


def perm_roll(data, n = 11, perm = 1000):
    
  
    dm = data['DM'][0]
    firing = data['Data'][0]
    cpd_perm  = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
  
    average = rew_prev_behaviour(data, n = n, perm = False)
    cpd  = []
    
    for  s, sess in enumerate(dm):
        
       
        DM = dm[s]
        firing_rates = firing[s]
     
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        choices = DM[:,1]
        reward = DM[:,2]  

        choices_current = choices-0.5
        
        previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1]      
        previous_choices = scipy.linalg.toeplitz(0.5-choices, np.zeros((1,n)))[n-1:-1]        
        interactions = scipy.linalg.toeplitz((((0.5-choices)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1]
      
        ones = np.ones(len(interactions)).reshape(len(interactions),1)
         
        X = np.hstack([previous_rewards,previous_choices,interactions,ones])
        
        value = np.matmul(X, average)
        
        choices_current = choices_current[n:]
        reward = reward[n:]
        value_choice = choices_current*value
        rew_ch = choices_current*reward
        ones = np.ones(len(rew_ch))
        firing_rates = firing_rates[n:]
        predictors_all = OrderedDict([
                                    ('Choice', choices_current),
                                    ('Reward', reward),
                                    ('Value',value), 
                                    ('Value Сhoice',value_choice), 
                                    ( 'Rew Ch', rew_ch),
#
                                    ('ones', ones)
                                    ])
        X = np.vstack(predictors_all.values()).T.astype(float)
      
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
    
        cpd.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
        
        for i in range(perm):
            y_perm = np.roll(y,np.random.randint(n), axis = 0)
            cpd_perm[i].append(_CPD(X,y_perm).reshape(n_neurons, n_timepoints, n_predictors))
    
    cpd_perm   = np.stack([np.concatenate(cpd_i,0) for cpd_i in cpd_perm],0)
    cpd = np.concatenate(cpd,0)
    return cpd,cpd_perm
            
                     
def run_perm_cpd(HP, PFC):
    cpd_HP,cpd_perm_HP =  perm_roll(HP, n = 11, perm = 5000)
    cpd_PFC,cpd_perm_PFC =  perm_roll(PFC, n = 11, perm = 5000)
    time_ms = np.asarray([0,   40,   80,  120,  160,  200,  240,  280,  320, 360,  400,  440,  480,  520,  560,  600,  640,  680,
               720,  760,  800,  840,  880,  920,  960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400,
               1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080, 2120,
               2160, 2200, 2240, 2280, 2320, 2360, 2400, 2440, 2480])

 
    c = wes.Royal2_5.mpl_colors

    cpd_HP_m = np.nanmean(cpd_HP,0)
    cpd_PFC_m = np.nanmean(cpd_PFC,0)

    plt.figure(figsize = (10,4))
    t = np.arange(0,63)
    cpd = cpd_PFC_m[:,:-1]
    cpd_perm = cpd_perm_PFC[:,:,:,:-1]
    values_95 = np.max(np.percentile(np.mean(cpd_perm,1),95,0),0)
    array_pvals = np.ones((cpd.shape[0],cpd.shape[1]))
    for i in range(cpd.shape[1]):
        array_pvals[(np.where(cpd[:,i] > values_95[i])[0]),i] = 0.05
 
    ymax = np.max(cpd)
    plt.subplot(1,2,1)
    plt.title('PFC')
    p = ['Choice','Reward', 'Value', 'Value x Choice', 'Reward x Choice']
                                    
#
    for i in np.arange(cpd.shape[1]):
      #  perm_plot = np.max(np.percentile(cpd_perm[:,:,i],95,axis = 0),axis=0)
        plt.plot(cpd[:,i], color = c[i], label = p[i])
        y = ymax*(1.2+0.02*i)
        p_vals = array_pvals[:,i]
        t05 = t[p_vals == 0.05]
        plt.plot(t05, np.ones(t05.shape)*y, '.', markersize=3, color=c[i])
        plt.vlines([25,35,42], 0,np.max(cpd), color= 'grey', linestyle = '--', alpha = 0.5)
        plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
        plt.legend()

        sns.despine()

    cpd = cpd_HP_m[:,:-1]
    cpd_perm = cpd_perm_HP[:,:,:,:-1]
    values_95 = np.max(np.percentile(np.mean(cpd_perm,1),95,0),0)
    array_pvals = np.ones((cpd.shape[0],cpd.shape[1]))
    for i in range(cpd.shape[1]):
        array_pvals[(np.where(cpd[:,i] > values_95[i])[0]),i] = 0.05
 
    ymax = np.max(cpd)
    plt.subplot(1,2,2)
    plt.title('HP')
    for i in np.arange(cpd.shape[1]):
      #  perm_plot = np.max(np.percentile(cpd_perm[:,:,i],95,axis = 0),axis=0)
        plt.plot(cpd[:,i], color = c[i], label = p[i])
        y = ymax*(1.2+0.02*i)
        p_vals = array_pvals[:,i]
        t05 = t[p_vals == 0.05]
        plt.plot(t05, np.ones(t05.shape)*y, '.', markersize=3, color=c[i])
        plt.vlines([25,35,42], 0,np.max(cpd), color= 'grey', linestyle = '--', alpha = 0.5)
        plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
        plt.legend()
        sns.despine()
        

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


   
  
  
def correlations_ab(d, n = 11, a = 'HP', perm = False, c_1 = 1):
    
   
    C_1_b, C_2_b, C_3_b = time_in_block(d, area = 'HP', n = n, plot_a = False, plot_b = True, perm = perm)
    
    C_1_a, C_2_a, C_3_a = time_in_block(d, area = 'HP', n = n, plot_a = True, plot_b = False, perm = perm)


    mean_value_a = np.mean([np.corrcoef(C_1_a[c_1].T,C_2_a[c_1].T),np.corrcoef(C_1_a[c_1].T,C_3_a[c_1].T), np.corrcoef(C_2_a[c_1].T,C_3_a[c_1].T)],0)
    mean_value_b = np.mean([np.corrcoef(C_1_b[c_1].T,C_2_b[c_1].T),np.corrcoef(C_1_b[c_1].T,C_3_b[c_1].T), np.corrcoef(C_2_b[c_1].T,C_3_b[c_1].T)],0)
    
    # plt.figure()
    # cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    # plt.imshow(mean_value[63:,:63:], cmap =cmap)
    # plt.colorbar()
    # plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    
    # plt.yticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    
    diag_a = np.sum(np.diagonal(mean_value_a[63:,:63:]))
    diag_b = np.sum(np.diagonal(mean_value_b[63:,:63:]))

      
    return mean_value_a, mean_value_b, diag_a,diag_b

def plot_diagonal_sums(HP,PFC, perm =  False, perm_n =  1000):
    
    
    n = 11
    perm_n =  5000
    mean_value_a_HP, mean_value_b_HP, diag_a_HP ,diag_b_HP =  correlations_ab(HP, n = 11, a = 'HP', perm = False,c_1 =1 )
    mean_value_a_PFC, mean_value_b_PFC, diag_a_PFC,diag_b_PFC = correlations_ab(PFC, n = 11, a = 'PFC',perm = False,c_1 =1 )
    
    hp = np.hstack((mean_value_a_HP,mean_value_b_HP))
    pfc = np.hstack((mean_value_a_PFC,mean_value_b_PFC))
    hp_pfc = np.vstack((hp,pfc))

    plt.figure()
    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    plt.imshow(hp_pfc, cmap =cmap)
    plt.colorbar()
    plt.xticks([0,10,25,35,42,50,60,63,73,88,98,105,113,123,\
                    126, 136, 151, 161, 168, 176, 186,\
                    189, 199, 214, 224, 231, 239, 249], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                               '-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                   '-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                               '-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    
    plt.yticks([0,10,25,35,42,50,60,63,73,88,98,105,113,123,\
                    126, 136, 151, 161, 168, 176, 186,\
                    189, 199, 214, 224, 231, 239, 249], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                               '-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                   '-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                               '-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])   
    all_diff_real_a = diag_a_PFC-diag_a_HP
    all_diff_real_b = diag_b_PFC-diag_b_HP

    all_shuffle_a = []
    all_shuffle_b = []

    all_subjects = np.hstack([PFC['DM'][0], HP['DM'][0]])
    all_subjects_firing = np.hstack([PFC['Data'][0], HP['Data'][0]])
    
    sessions_n = np.arange(len(all_subjects))

    for i in range(perm_n):
        np.random.shuffle(sessions_n) # Shuffle PFC/HP sessions
        indices_HP = sessions_n[PFC['DM'][0].shape[0]:]
        indices_PFC = sessions_n[:PFC['DM'][0].shape[0]]

        PFC_shuffle_dm = all_subjects[np.asarray(indices_PFC)]
        HP_shuffle_dm = all_subjects[np.asarray(indices_HP)]
        
        PFC_shuffle_f = all_subjects_firing[np.asarray(indices_PFC)]
        HP_shuffle_f = all_subjects_firing[np.asarray(indices_HP)]
       
        HP_shuffle= [HP_shuffle_dm,HP_shuffle_f]
        PFC_shuffle= [PFC_shuffle_dm,PFC_shuffle_f]

        _all_diff_a = []
        _all_diff_b = []

        for d in [HP_shuffle,PFC_shuffle]:
            
            mean_value_a_HP, mean_value_b_HP, diag_a_HP ,diag_b_HP =  correlations_ab(d, n = n, a = 'perm', perm = True, c_1 =1 )
            _all_diff_a.append(diag_a_HP)
            _all_diff_b.append(diag_b_HP)
 
        all_shuffle_a.append(_all_diff_a)
        all_shuffle_b.append(_all_diff_b)
 
    diff_all_a  = []
    diff_all_b  = []
    
    for i,ii in enumerate(all_shuffle_a):
        diff_all_a.append(all_shuffle_a[i][0]- all_shuffle_a[i][1])
        diff_all_b.append(all_shuffle_b[i][0]- all_shuffle_b[i][1])
  
    _all_95_a = np.percentile(diff_all_a,95)
    _all_95_b = np.percentile(diff_all_b,95)
    
    plt.figure(figsize = (4,5))
    plt.subplot(1,2,1)
    plt.hist(diff_all_a, color = 'grey')
    plt.vlines(all_diff_real_a,ymin = 0, ymax = max(np.histogram(diff_all_a)[0]))
    plt.vlines(_all_95_a,ymin = 0, ymax = max(np.histogram(diff_all_a)[0]), color = 'red')
    
    plt.subplot(1,2,2)
    plt.hist(diff_all_b, color = 'grey')
    plt.vlines(all_diff_real_b,ymin = 0, ymax = max(np.histogram(diff_all_b)[0]))
    plt.vlines(_all_95_b, ymin = 0, ymax = max(np.histogram(diff_all_b)[0]), color = 'red')

    sns.despine()



