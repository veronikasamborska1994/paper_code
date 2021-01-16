

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
    std_err = []
    
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



    
def plot_figure(PFC, HP, n = 11, c_1 = 1):
    
    n = 11
    c_1 = 3
    
    init_t = 25
    ch_t = 36
    r_t = 42
    
    C_1_HP, C_2_HP, C_3_HP = chosen_value(HP, n = n, perm = False)
    C_1_PFC ,C_2_PFC, C_3_PFC = chosen_value(PFC,  n = n, perm = False)
    
    isl = wes.Royal2_5.mpl_colors

         
    scatter_HP_init_1 = C_1_HP[c_1,:,init_t];   scatter_PFC_init_1 = C_1_PFC[c_1,:,init_t]

    scatter_HP_init_2 = C_2_HP[c_1,:,init_t];   scatter_PFC_init_2 = C_2_PFC[c_1,:,init_t]

    scatter_HP_init_3 = C_3_HP[c_1,:,init_t];   scatter_PFC_init_3 = C_3_PFC[c_1,:,init_t]

 
    scatter_HP_ch_1 = C_1_HP[c_1,:,ch_t];     scatter_PFC_ch_1 = C_1_PFC[c_1,:,ch_t]

    scatter_HP_ch_2 = C_2_HP[c_1,:,ch_t];     scatter_PFC_ch_2 = C_2_PFC[c_1,:,ch_t]

    scatter_HP_ch_3 = C_3_HP[c_1,:,ch_t];     scatter_PFC_ch_3 = C_3_PFC[c_1,:,ch_t]

    
    scatter_HP_rew_1 = C_1_HP[c_1,:,r_t];    scatter_PFC_rew_1 = C_1_PFC[c_1,:,r_t]

    scatter_HP_rew_2 = C_2_HP[c_1,:,r_t];    scatter_PFC_rew_2 = C_2_PFC[c_1,:,r_t]

    scatter_HP_rew_3 = C_3_HP[c_1,:,r_t];    scatter_PFC_rew_3 = C_3_PFC[c_1,:,r_t]

    
    reg_x = [scatter_HP_init_1,scatter_HP_init_2,scatter_HP_init_3,\
             scatter_HP_ch_1,scatter_HP_ch_2,scatter_HP_ch_3,\
             scatter_HP_rew_1,scatter_HP_rew_2,scatter_HP_rew_3,\
             scatter_PFC_init_1,scatter_PFC_init_2,scatter_PFC_init_3,\
             scatter_PFC_ch_1,scatter_PFC_ch_2,scatter_PFC_ch_3,\
             scatter_PFC_rew_1,scatter_PFC_rew_2,scatter_PFC_rew_3]
                
    reg_y = [scatter_HP_init_2,scatter_HP_init_3,scatter_HP_init_1,\
             scatter_HP_ch_2,scatter_HP_ch_3,scatter_HP_ch_1,\
             scatter_HP_rew_2,scatter_HP_rew_3,scatter_HP_rew_1,\
             scatter_PFC_init_2,scatter_PFC_init_3,scatter_PFC_init_1,\
             scatter_PFC_ch_2,scatter_PFC_ch_3,scatter_PFC_ch_1,\
             scatter_PFC_rew_2,scatter_PFC_rew_3,scatter_PFC_rew_1]
        
    coefs = []
    for r,reg in enumerate(reg_y):
        y = reg_y[r].reshape([len(reg_y[r]),-1]) 
        x = reg_x[r].reshape([len(reg_x[r]),-1]) 

        ols = LinearRegression()
        ols.fit(x,y)
        coefs.append(ols.coef_[0]) # Predictor loadings

    task_mean = [np.mean(coefs[:3]),np.mean(coefs[3:6]), np.mean(coefs[6:9]),np.mean(coefs[9:12]),np.mean(coefs[12:15]), np.mean(coefs[15:18])]
    diff =[]
    for i,ii in enumerate(task_mean):
        if i <3:
            diff.append(task_mean[i+3]-task_mean[i])
    

    correlation_differences  = perumute_sessions(HP, PFC, c_1 = c_1, n = n, perm_n = 2, init_t = init_t, ch_t = ch_t, r_t = r_t)
     
    plt.figure(figsize = (10,15))
    plt.figure(figsize = (5,10))
    correlation_differences = np.asarray(correlation_differences)
 
    for h,hist in enumerate(correlation_differences.T):

        plt.subplot(3,1,h+1)
        c = isl[0]
        if r >2:
            c = isl[3] 
        plt.hist(hist, 10,color = c)      
        plt.vlines(diff[h], 0, np.max(np.histogram(hist,10)[0]), color = 'black', label = 'data')
        plt.vlines(np.percentile(hist,99), 0, np.max(np.histogram(hist,10)[0]), color = 'grey', linestyle  = 'dotted', label =  '<.001')
        plt.vlines(np.percentile(hist,95), 0, np.max(np.histogram(hist,10)[0]), color = 'grey', linestyle  = '--', label =  '<.05')
        plt.legend()
        plt.xlabel('Correlation')
        plt.ylabel('Count')
        sns.despine()
    
    
    # scatter_HP_x_init = np.concatenate((C_1_HP[1,:,init_t],C_1_HP[1,:,init_t],C_2_HP[1,:,init_t]))
    # scatter_HP_y_init = np.concatenate((C_2_HP[1,:,init_t],C_3_HP[1,:,init_t],C_3_HP[1,:,init_t]))
  
    # scatter_HP_x_choice = np.concatenate((C_1_HP[1,:,ch_t],C_1_HP[1,:,ch_t],C_2_HP[1,:,ch_t]))
    # scatter_HP_y_choice = np.concatenate((C_2_HP[1,:,ch_t],C_3_HP[1,:,ch_t],C_3_HP[1,:,ch_t]))
  
    # scatter_HP_x_reward = np.concatenate((C_1_HP[1,:,r_t],C_1_HP[1,:,r_t],C_2_HP[1,:,r_t]))
    # scatter_HP_y_reward = np.concatenate((C_2_HP[1,:,r_t],C_3_HP[1,:,r_t],C_3_HP[1,:,r_t]))
   
    
    
    # scatter_PFC_x_init = np.concatenate((C_1_PFC[1,:,init_t],C_1_PFC[1,:,init_t],C_2_PFC[1,:,init_t]))
    # scatter_PFC_y_init = np.concatenate((C_2_PFC[1,:,init_t],C_3_PFC[1,:,init_t],C_3_PFC[1,:,init_t]))
  
    # scatter_PFC_x_choice = np.concatenate((C_1_PFC[1,:,ch_t],C_1_PFC[1,:,ch_t],C_2_PFC[1,:,ch_t]))
    # scatter_PFC_y_choice = np.concatenate((C_2_PFC[1,:,ch_t],C_3_PFC[1,:,ch_t],C_3_PFC[1,:,ch_t]))
  
    # scatter_PFC_x_reward = np.concatenate((C_1_PFC[1,:,r_t],C_1_PFC[1,:,r_t],C_2_PFC[1,:,r_t]))
    # scatter_PFC_y_reward = np.concatenate((C_2_PFC[1,:,r_t],C_3_PFC[1,:,r_t],C_3_PFC[1,:,r_t]))

    # reg_x = [scatter_HP_x_init,scatter_HP_x_choice,scatter_HP_x_reward,\
    #          scatter_PFC_x_init, scatter_PFC_x_choice, scatter_PFC_x_reward]
               

    # reg_y = [scatter_HP_y_init,scatter_HP_y_choice,scatter_HP_y_reward,\
    #          scatter_PFC_y_init, scatter_PFC_y_choice, scatter_PFC_y_reward]
    
    # real_corr = []    
        
    # plt.figure(figsize = (10,15))
    # for r,reg in enumerate(reg_y):
    #     plt.subplot(2,3,r+1)
    #     c = isl[0]
    #     if r >2:
    #         c = isl[3] 
            
    #     sns.regplot(reg_x[r],reg_y[r], fit_reg = True, color = c)
    #     corr = np.corrcoef(reg_x[r],reg_y[r])[0,1]
        
        
    #     #plt.xlabel('Initiation in One Task')
    #     #plt.ylabel('Initiation in Another Task')
    #     plt.annotate('r = ' + str(np.around(corr,3)), [np.max(reg_x[r]),np.max(reg_y[r])])

    #     sns.despine()
    #     plt.tight_layout()
        
 
    
       



def perumute_sessions(HP, PFC, c_1 = 1, n = 6, perm_n = 500, init_t = 25, ch_t = 36, r_t = 42):

    all_subjects = np.hstack([PFC['DM'][0], HP['DM'][0]])
    all_subjects_firing = np.hstack([PFC['Data'][0], HP['Data'][0]])
    
    sessions_n = np.arange(len(all_subjects))
    correlation_differences = []

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

        
        C_1_HP, C_2_HP, C_3_HP = chosen_value(HP_shuffle,  n = n, perm = True)
        C_1_PFC ,C_2_PFC, C_3_PFC = chosen_value(PFC_shuffle,  n = n,  perm = True)
        
           
        HP_init_1 = (C_1_HP[c_1,:,init_t]);  PFC_init_1 = (C_1_PFC[c_1,:,init_t])

        HP_init_2 = (C_2_HP[c_1,:,init_t]);   PFC_init_2 = (C_2_PFC[c_1,:,init_t])

        HP_init_3 = (C_3_HP[c_1,:,init_t]);    PFC_init_3 = (C_3_PFC[c_1,:,init_t])

 
        HP_ch_1 = (C_1_HP[c_1,:,ch_t]);    PFC_ch_1 = (C_1_PFC[c_1,:,ch_t])

        HP_ch_2 = (C_2_HP[c_1,:,ch_t]);    PFC_ch_2= (C_2_PFC[c_1,:,ch_t])

        HP_ch_3= (C_3_HP[c_1,:,ch_t]);    PFC_ch_3 = (C_3_PFC[c_1,:,ch_t])

    
        HP_rew_1 = (C_1_HP[c_1,:,r_t]);    PFC_rew_1 = (C_1_PFC[c_1,:,r_t])

        HP_rew_2 = (C_2_HP[c_1,:,r_t]);    PFC_rew_2 = (C_2_PFC[c_1,:,r_t])

        HP_rew_3 = (C_3_HP[c_1,:,r_t]);    PFC_rew_3 = (C_3_PFC[c_1,:,r_t])

       
        reg_x = [HP_init_1, HP_init_2, HP_init_3,\
                 HP_ch_1, HP_ch_2,HP_ch_3,\
                 HP_rew_1, HP_rew_2, HP_rew_3,\
                 PFC_init_1, PFC_init_2,PFC_init_3,\
                 PFC_ch_1, PFC_ch_2, PFC_ch_3,\
                 PFC_rew_1,PFC_rew_2, PFC_rew_3]
                    
        reg_y = [HP_init_2, HP_init_3, HP_init_1,\
                 HP_ch_2, HP_ch_3,HP_ch_1,\
                 HP_rew_2, HP_rew_3, HP_rew_1,\
                 PFC_init_2, PFC_init_3,PFC_init_1,\
                 PFC_ch_2, PFC_ch_3, PFC_ch_1,\
                 PFC_rew_2,PFC_rew_3, PFC_rew_1]
        coefs = []
        for r,reg in enumerate(reg_y):
            y = reg_y[r].reshape([len(reg_y[r]),-1]) 
            x = reg_x[r].reshape([len(reg_x[r]),-1]) 
    
            ols = LinearRegression()
            ols.fit(x,y)
            coefs.append(ols.coef_[0]) # Predictor loadings
    
        task_mean = [np.mean(coefs[:3]),np.mean(coefs[3:6]), np.mean(coefs[6:9]),np.mean(coefs[9:12]),np.mean(coefs[12:15]), np.mean(coefs[15:18])]
        diff =[]
        for i,ii in enumerate(task_mean):
            if i <3:
                diff.append(task_mean[i+3]-task_mean[i])
                
        correlation_differences.append(diff)
 
    return correlation_differences

    
def chosen_value(data,  n = 10, perm = True):
   
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
        choice_value_1 = choices_1*value_1

         
        predictors_all = OrderedDict([
                                    ('Choice',choices_1),
                                    ('Reward', rewards_1),
                                    ('Value',value_1), 
                                    ('Value Choice',choice_value_1),                
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
        choice_value_2 = choices_2*value_2

 
        predictors_all = OrderedDict([                                    
                                   ('Choice',choices_2),
                                    ('Reward', rewards_2),
                                    ('Value',value_2),      
                                    ('Value Choice',choice_value_2),                
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
        value_3 = np.matmul(X_3, average)
        rewards_3 = rewards_3[n:]
        choices_3 = choices_3[n:]
        choice_value_3 = choices_3*value_3
  
        
        ones_3 = np.ones(len(choices_3))
        trials_3 = len(choices_3)

        firing_rates_3 = firing_rates[task_3][n:]
        
    
    
            
        predictors_all = OrderedDict([(
                                    'Choice',choices_3),
                                    ('Rew', rewards_3),
                                    ('Value',value_3),      
                                    ('Value Choice',choice_value_3),                
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



  
def correlations(d, n = 11, a = 'HP', perm = False):
    c_1 = 3
    
    C_1,C_2, C_3 = chosen_value(d, n = n, perm = perm)
   
    mean_value = np.mean([np.corrcoef(C_1[c_1].T,C_2[c_1].T),np.corrcoef(C_1[c_1].T,C_3[c_1].T), np.corrcoef(C_2[c_1].T,C_3[c_1].T)],0)
    
    # plt.figure()
    # cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    # plt.imshow(mean_value[63:,:63:], cmap =cmap)
    # plt.colorbar()
    # plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    
    # plt.yticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    
    diag = np.sum(np.diagonal(mean_value[63:,:63:]))
     
      
    return mean_value, diag

def plot_diagonal_sums(HP,PFC, perm =  False, perm_n =  1000):
    
    
    perm_n = 2
    perm_n =  5000
    mean_value_HP, diag_HP = correlations(HP, n = 11, a = 'HP', perm = False)
    mean_value_PFC, diag_PFC  = correlations(PFC, n = 11, a = 'PFC',perm = False)
    
    # plt.figure()
    # cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    # plt.imshow(mean_value[63:,:63:], cmap =cmap)
    # plt.colorbar()
    # plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    
    # plt.yticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])   
    
    all_diff_real = diag_PFC-diag_HP
    all_shuffle = []
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

        _all_diff = []
        for d in [HP_shuffle,PFC_shuffle]:
            
            pemr_mean,perm_diag  = correlations(d, n = 11,  a = 'perm', perm = True)
            _all_diff.append(perm_diag)
            
        all_shuffle.append(_all_diff)
        
    diff_all  = []
    
    for i,ii in enumerate(all_shuffle):
        diff_all.append(all_shuffle[i][0]- all_shuffle[i][1])
        
    _all_95 = np.percentile(diff_all,95)
    
    
    #plt.figure(figsize = (4,5))
    plt.hist(diff_all, color = 'grey')
    plt.vlines(all_diff_real,ymin = 0, ymax = max(np.histogram(diff_all)[0]))
    plt.vlines(_all_95,ymin = 0, ymax = max(np.histogram(diff_all)[0]), color = 'red')
  
   
    sns.despine()


