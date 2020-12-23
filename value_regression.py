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
    
       
    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap

     
    
  
def task_ind(task, a_pokes, b_pokes):
    
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task));
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid


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
    
    #tstats = cope/np.sqrt(varcope)
    
    return cope,varcope


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
         #cov = results.cov_params()
         #std_err.append(np.sqrt(np.diag(cov)))

    average = np.mean(results_array,0)
    #std = np.std(results_array,0)/len(dm)

    # average = np.mean(results_array,0)
    # c = 'green'
    # plt.plot(np.arange(len(average))[n*2:-1], average[n*2:-1], color = c, label = 'PFC')
    # plt.fill_between(np.arange(len(average))[n*2:-1], average[n*2:-1]+std[n*2:-1], average[n*2:-1]- std[n*2:-1],alpha = 0.2, color =c)
    # plt.hlines(0, xmin = np.arange(len(average))[n*2:-1][0],xmax = np.arange(len(average))[n*2:-1][-1])
    # length = len(np.arange(len(average))[n*2:-1])
    # plt.xticks(np.arange(len(average))[n*2:-1],np.arange(length))
    # sns.despine()
    # plt.legend()
    
    return average



def generalisation_plot(C_1,C_2,C_3, c_1, reward_times_to_choose = np.asarray([25,35,42]), task_check = 0):
    
    c_1 = c_1
    C_1_rew = C_1[c_1]; C_2_rew = C_2[c_1]; C_3_rew = C_3[c_1]
    C_1_rew_count = C_1[c_1]; C_2_rew_count = C_2[c_1]; C_3_rew_count = C_3[c_1]
   
    reward_times_to_choose = reward_times_to_choose
    
    C_1_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_2_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_3_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
   
    j = 0
    for i in reward_times_to_choose:
        if i ==reward_times_to_choose[0]:
            C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i-20:i],1)
            C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i-20:i],1)
            C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i-20:i],1)
        if i ==reward_times_to_choose[1] or i == reward_times_to_choose[2]:
            C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i-2:i+2],1)
            C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i-2:i+2],1)
            C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i-2:i+2],1)
        elif i == reward_times_to_choose[3]:
            C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i:],1)
            C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i:],1)
            C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i:],1)
         
        j +=1
    
   
    C_1_rew_count_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_2_rew_count_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_3_rew_count_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    j = 0
    for i in reward_times_to_choose:
        if i ==reward_times_to_choose[0]:
            C_1_rew_count_proj[:,j] = np.mean(C_1_rew_count[:,i-20:i],1)
            C_2_rew_count_proj[:,j] = np.mean(C_2_rew_count[:,i-20:i],1)
            C_3_rew_count_proj[:,j] = np.mean(C_3_rew_count[:,i-20:i],1)
        if i ==reward_times_to_choose[1] or i == reward_times_to_choose[2]:
            C_1_rew_count_proj[:,j] = np.mean(C_1_rew_count[:,i-2:i+2],1)
            C_2_rew_count_proj[:,j] = np.mean(C_2_rew_count[:,i-2:i+2],1)
            C_3_rew_count_proj[:,j] = np.mean(C_3_rew_count[:,i-2:i+2],1)
        elif i == reward_times_to_choose[3]:
            C_1_rew_count_proj[:,j] = np.mean(C_1_rew_count[:,i:],1)
            C_2_rew_count_proj[:,j] = np.mean(C_2_rew_count[:,i:],1)
            C_3_rew_count_proj[:,j] = np.mean(C_3_rew_count[:,i:],1)
     
        j +=1
        
    
        
    cpd_1_2_rew, cpd_1_2_rew_var = regression_code_session(C_2_rew_count, C_1_rew_proj);  
    cpd_1_3_rew, cpd_1_3_rew_var = regression_code_session(C_3_rew_count, C_1_rew_proj); 
    cpd_2_3_rew, cpd_2_3_rew_var = regression_code_session(C_3_rew_count, C_2_rew_proj)
    
   
    if task_check == 0:
       value_to_value = (cpd_1_2_rew + cpd_1_3_rew + cpd_2_3_rew)#/np.sqrt((cpd_1_2_rew_var+cpd_1_3_rew_var+cpd_2_3_rew_var))

    elif  task_check == 1: # B close in space
        #value_to_value  = reg_f.regression_code(C_2_rew_count, C_1_rew_proj)
        value_to_value  = cpd_1_2_rew

    elif  task_check == 2: # initi in the same location
        #value_to_value  = reg_f.regression_code(C_3_rew_count, C_1_rew_proj)
        value_to_value  = cpd_1_3_rew

    elif  task_check == 3: # init moves to B
        #value_to_value  = reg_f.regression_code(C_3_rew_count, C_2_rew_proj)
        value_to_value  = cpd_2_3_rew

    
    return value_to_value


def run_permute_a_b():  
        
    
     HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
     PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
     
     perm_HP_all, perm_PFC_all = a_b_check(PFC, HP, n = 11, perm = 1000, task_check = 0, reward_times_to_choose = np.asarray([20,25,36,42]), c_1 = 1)
     perm_HP_1, perm_PFC_all_1 = a_b_check(PFC, HP, n = 11, perm = 1000, task_check = 1, reward_times_to_choose = np.asarray([20,25,35,42]), c_1 = 1) 
     perm_HP_2, perm_PFC_all_2 = a_b_check(PFC, HP, n = 11, perm = 1000, task_check = 2, reward_times_to_choose = np.asarray([20,25,35,42]), c_1 = 1)
     perm_HP_3, perm_PFC_all_3 = a_b_check(PFC, HP, n = 11, perm = 1000, task_check = 3, reward_times_to_choose = np.asarray([20,25,35,42]), c_1 = 1)
     
     real_dif_PFC_0, real_dif_HP_0 = real_diff_a_b(HP, PFC, n = 11, task_check = 0, c_1 = 1, reward_times_to_choose= np.asarray([20,25,36,42]))
     real_dif_PFC_1, real_dif_HP_1 = real_diff_a_b(HP, PFC, n = 11, task_check = 1, c_1 = 1, reward_times_to_choose= np.asarray([20,25,35,42]))
     real_dif_PFC_2, real_dif_HP_2 = real_diff_a_b(HP, PFC, n = 11, task_check = 2, c_1 = 1, reward_times_to_choose= np.asarray([20,25,35,42]))
     real_dif_PFC_3, real_dif_HP_3 = real_diff_a_b(HP, PFC, n = 11, task_check = 3, c_1 = 1, reward_times_to_choose= np.asarray([20,25,35,42]))


     a_b_pval_0_HP = np.where(real_dif_HP_0[:-1].T >perm_HP_all[:-1])
     a_b_pval_0_PFC = np.where(real_dif_PFC_0[:-1].T >perm_PFC_all[:-1])

     a_b_pval_1_HP = np.where(real_dif_HP_1[:-1].T >perm_HP_1[:-1])
     a_b_pval_2_HP = np.where(real_dif_HP_2[:-1].T>perm_HP_2[:-1])
     a_b_pval_3_HP = np.where(real_dif_HP_3[:-1].T>perm_HP_3[:-1])

     a_b_pval_1_PFC = np.where(real_dif_PFC_1[:-1].T>perm_PFC_all_1[:-1])
     a_b_pval_2_PFC = np.where(real_dif_PFC_2[:-1].T>perm_PFC_all_2[:-1])
     a_b_pval_3_PFC = np.where(real_dif_PFC_3[:-1].T>perm_PFC_all_3[:-1])
    

def run_permute_animals():  
        
    
     HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
     PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')


     ## Statistical Difference between PFC and HP
     perms_b_all_0,perms_a_all_0,perms_md_0 = plot(HP, PFC, c_1 = 1, n = 11, reward_times_to_choose = np.asarray([20,25,35,42]),task_check = 0)
   
   
     perms_b_all_1,perms_a_all_1, perms_md_1 = plot(HP, PFC, c_1 = 1, n = 11,
                                     reward_times_to_choose = np.asarray([20,25,35,42]),task_check = 1)
     
     perms_b_all_2,perms_a_all_2, perms_md_2 = plot(HP, PFC, c_1 = 1, n = 11,
                                     reward_times_to_choose = np.asarray([20,25,35,42]),task_check = 2)
    
     perms_b_all_3,perms_a_all_3, perms_md_3 = plot(HP, PFC, c_1 = 1, n = 11,
                                     reward_times_to_choose = np.asarray([20,25,35,42]),task_check = 3)
      
def run_permute_sessions():     
    
     # A vs B permute sessions
     
     perms_b_s_all,perms_as_all, perms_ab_all = perumute_sessions(HP, PFC, c_1 = 1, n = 11 , reward_times_to_choose = np.asarray([20,25,35,42]), task_check = 0, perm_n = 500)
     perms_b_s_1,perms_as_1, perms_ab_1 = perumute_sessions(HP, PFC, c_1 = 1, n = 11 , reward_times_to_choose = np.asarray([20,25,35,42]), task_check = 1, perm_n = 500)
     perms_b_s_2,perms_as_2, perms_ab_2 = perumute_sessions(HP, PFC, c_1 = 1, n = 11 , reward_times_to_choose = np.asarray([20,25,35,42]), task_check = 2, perm_n = 500)
     perms_b_s_3,perms_as_3, perms_ab_3 = perumute_sessions(HP, PFC, c_1 = 1, n = 11 , reward_times_to_choose = np.asarray([20,25,35,42]), task_check = 3, perm_n = 500)
     
     
     plot_figure(PFC, HP, reward_times_to_choose = np.asarray([20,25,35,42]), task_check = 0, n = 11, c_1 = 1, perms_b_all_1 = perms_b_s_all, perms_a_all_1 = perms_as_all, meta_diff = perms_ab_all  )
    
     plot_figure(PFC, HP, reward_times_to_choose = np.asarray([20,25,35,42]), task_check = 1, n = 11, c_1 = 1, perms_b_all_1 = perms_b_s_1, perms_a_all_1 = perms_as_1, meta_diff = perms_ab_1  )

     plot_figure(PFC, HP, reward_times_to_choose = np.asarray([20,25,35,42]), task_check = 2, n = 11, c_1 = 1, perms_b_all_1 = perms_b_s_2, perms_a_all_1 = perms_as_2, meta_diff = perms_ab_2  )
  
     plot_figure(PFC, HP, reward_times_to_choose = np.asarray([20,25,35,42]), task_check = 3, n = 11, c_1 = 1, perms_b_all_1 = perms_b_s_3, perms_a_all_1 = perms_as_3, meta_diff = perms_ab_3  )

    
def plot_figure(PFC, HP, reward_times_to_choose = np.asarray([20,25,35,42]), task_check = 0, n = 11, c_1 = 1, perms_b_all_1 = [1,2,3], perms_a_all_1 = [1,2,3], meta_diff = [1,2,3]):
    
    
     C_1_HP_b, C_2_HP_b, C_3_HP_b = time_in_block(HP, area = 'HP', n = n, plot_a = False, plot_b = True, perm = False)
     C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = time_in_block(PFC, area = 'PFC', n = n, plot_a = False, plot_b = True, perm = False)
    
     C_1_HP_a, C_2_HP_a, C_3_HP_a = time_in_block(HP, area = 'HP', n = n, plot_a = True, plot_b = False, perm = False)
     C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = time_in_block(PFC, area = 'PFC', n = n, plot_a = True, plot_b = False, perm = False)
   
     value_to_value_PFC_a_val = generalisation_plot(C_1_PFC_a,C_2_PFC_a,C_3_PFC_a, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
     value_to_value_PFC_b_val = generalisation_plot(C_1_PFC_b,C_2_PFC_b,C_3_PFC_b, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
     value_to_value_PFC_a_val = value_to_value_PFC_a_val[:-1]
     value_to_value_PFC_b_val = value_to_value_PFC_b_val[:-1]

     value_to_value_HP_a_val = generalisation_plot(C_1_HP_a,C_2_HP_a,C_3_HP_a, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
     value_to_value_HP_b_val = generalisation_plot(C_1_HP_b,C_2_HP_b,C_3_HP_b, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
     value_to_value_HP_a_val = value_to_value_HP_a_val[:-1]
     value_to_value_HP_b_val = value_to_value_HP_b_val[:-1]

     isl = wes.Royal2_5.mpl_colors
    
     plt.figure(figsize = (6,4))
    
     max_ind = np.max([np.max(value_to_value_PFC_a_val), np.max(value_to_value_HP_a_val),np.max(value_to_value_PFC_b_val),\
                          np.max(value_to_value_HP_b_val)])+1.5
     min_ind = np.min([np.min(value_to_value_PFC_a_val), np.min(value_to_value_HP_a_val),np.min(value_to_value_PFC_b_val),\
                          np.min(value_to_value_HP_b_val)])-0.5
             
     plt.subplot(2,3,1)
     plt.plot(value_to_value_PFC_a_val[1],color = isl[0],  label = 'PFC A')
     plt.plot(value_to_value_HP_a_val[1],color = isl[3],  label = 'CA1 A')
     plt.ylim(min_ind,max_ind)   
     p = perms_a_all_1[0][np.where(perms_a_all_1[1] ==1)[0]]
     p_meta = meta_diff[0][np.where(meta_diff[1] ==1)[0]]
     plt.plot(p, np.ones(len(p))+max_ind-2, '.', markersize=3, color= 'grey')
     plt.plot(p_meta, np.ones(len(p_meta))+max_ind-1, '.', markersize=3, color= 'red')

     plt.ylabel(str(task_check))
     plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
     plt.legend()
     
     plt.subplot(2,3,4)
     plt.plot(value_to_value_PFC_b_val[1],color = isl[0], label = 'PFC B')
     plt.plot(value_to_value_HP_b_val[1],color = isl[3], label = 'CA1 B')
     plt.ylim(min_ind,max_ind)
     p = perms_b_all_1[0][np.where(perms_b_all_1[1] ==1)[0]]
     plt.plot(p, np.ones(len(p))+max_ind-2, '.', markersize=3, color= 'grey')
     plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])

     plt.legend()

     plt.subplot(2,3,2)
     plt.plot(value_to_value_PFC_a_val[2],color = isl[0],  label = 'PFC A')
     plt.plot(value_to_value_HP_a_val[2],color = isl[3],  label = 'CA1 A')
     plt.ylim(min_ind,max_ind)     
     p = perms_a_all_1[0][np.where(perms_a_all_1[1] == 2)[0]]
     plt.plot(p, np.ones(len(p))+max_ind-1.5, '.', markersize=3, color= 'grey')
     p_meta = meta_diff[0][np.where(meta_diff[1] ==2)[0]]
     plt.plot(p_meta, np.ones(len(p_meta))+max_ind-1, '.', markersize=3, color= 'red')

     plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
 
     plt.legend()

     plt.subplot(2,3,5)
     plt.plot(value_to_value_PFC_b_val[2],color = isl[0], label = 'PFC B')
     plt.plot(value_to_value_HP_b_val[2],color = isl[3], label = 'CA1 B')
     plt.ylim(min_ind,max_ind)
     p = perms_b_all_1[0][np.where(perms_b_all_1[1] == 2)[0]]
     plt.plot(p, np.ones(len(p))+max_ind-2, '.', markersize=3, color= 'grey')
     plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
     plt.legend()

     plt.subplot(2,3,3)
     plt.plot(value_to_value_PFC_a_val[3],color = isl[0],  label = 'PFC A')
     plt.plot(value_to_value_HP_a_val[3],color = isl[3],  label = 'CA1 A')
     plt.ylim(min_ind,max_ind)     
     p = perms_a_all_1[0][np.where(perms_a_all_1[1] == 3)[0]]
     plt.plot(p, np.ones(len(p))+max_ind-2, '.', markersize=3, color= 'grey')
     p_meta = meta_diff[0][np.where(meta_diff[1] ==3)[0]]
     plt.plot(p_meta, np.ones(len(p_meta))+max_ind-1, '.', markersize=3, color= 'red')

     plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
     plt.legend()

     plt.subplot(2,3,6)
     plt.plot(value_to_value_PFC_b_val[3],color = isl[0], label = 'PFC B')
     plt.plot(value_to_value_HP_b_val[3],color = isl[3], label = 'CA1 B')
     plt.ylim(min_ind,max_ind)
     p = perms_b_all_1[0][np.where(perms_b_all_1[1] == 3)[0]]
     plt.plot(p, np.ones(len(p))+max_ind-2, '.', markersize=3, color= 'grey')
     plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
     plt.legend()

     sns.despine()
     plt.tight_layout()

     

        
    
def real_diff_a_b(HP, PFC, n = 11, task_check = 3, c_1 = 1, reward_times_to_choose= np.asarray([20,25,35,42])):
    
    C_1_HP_b, C_2_HP_b, C_3_HP_b = time_in_block(HP, area = 'HP', n = n, plot_a = False, plot_b = True, perm = False)
    C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = time_in_block(PFC, area = 'PFC', n = n, plot_a = False, plot_b = True, perm = False)
    
    C_1_HP_a, C_2_HP_a, C_3_HP_a = time_in_block(HP, area = 'HP', n = n, plot_a = True, plot_b = False, perm = False)
    C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = time_in_block(PFC, area = 'PFC', n = n, plot_a = True, plot_b = False, perm = False)
   
   
    value_to_value_PFC_a_val = generalisation_plot(C_1_PFC_a,C_2_PFC_a,C_3_PFC_a, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_PFC_b_val = generalisation_plot(C_1_PFC_b,C_2_PFC_b,C_3_PFC_b, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
  
    value_to_value_HP_a_val = generalisation_plot(C_1_HP_a,C_2_HP_a,C_3_HP_a, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_HP_b_val = generalisation_plot(C_1_HP_b,C_2_HP_b,C_3_HP_b, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    real_dif_PFC = value_to_value_PFC_a_val-value_to_value_PFC_b_val
    real_dif_HP = value_to_value_HP_a_val-value_to_value_HP_b_val
    
  
    return real_dif_PFC,real_dif_HP


def plot(HP, PFC, c_1 = 1, n = 6 , reward_times_to_choose = [1,2,3,4], task_check = 0):
    
 
    C_1_HP_b, C_2_HP_b, C_3_HP_b = time_in_block(HP, area = 'HP', n = n, plot_a = False, plot_b = True, perm = False)
    C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = time_in_block(PFC, area = 'PFC', n = n, plot_a = False, plot_b = True, perm = False)
    
    C_1_HP_a, C_2_HP_a, C_3_HP_a = time_in_block(HP, area = 'HP', n = n, plot_a = True, plot_b = False, perm = False)
    C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = time_in_block(PFC, area = 'PFC', n = n, plot_a = True, plot_b = False, perm = False)
    
   
    value_to_value_PFC_a_val = generalisation_plot(C_1_PFC_a,C_2_PFC_a,C_3_PFC_a, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_PFC_b_val = generalisation_plot(C_1_PFC_b,C_2_PFC_b,C_3_PFC_b, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_PFC_a_val = value_to_value_PFC_a_val[:-1]
    value_to_value_PFC_b_val = value_to_value_PFC_b_val[:-1]

    value_to_value_HP_a_val = generalisation_plot(C_1_HP_a,C_2_HP_a,C_3_HP_a, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_HP_b_val = generalisation_plot(C_1_HP_b,C_2_HP_b,C_3_HP_b, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_HP_a_val = value_to_value_HP_a_val[:-1]
    value_to_value_HP_b_val = value_to_value_HP_b_val[:-1]

    difference_a = []
    difference_b = []
    meta_difference = []


    all_subjects = [PFC['DM'][0][:9], PFC['DM'][0][9:25],PFC['DM'][0][25:39],PFC['DM'][0][39:],HP['DM'][0][:16], HP['DM'][0][16:24],HP['DM'][0][24:]]
    all_subjects_firing = [PFC['Data'][0][:9], PFC['Data'][0][9:25],PFC['Data'][0][25:39],PFC['Data'][0][39:],HP['Data'][0][:16], HP['Data'][0][16:24],HP['Data'][0][24:]]

    animals_PFC = [0,1,2,3]
    animals_HP = [4,5,6]
    m, n = len(animals_PFC), len(animals_HP)
  
    for indices_PFC in combinations(range(m + n), m):
        indices_HP = [i for i in range(m + n) if i not in indices_PFC]
       
        PFC_shuffle_dm = np.concatenate(np.asarray(all_subjects)[np.asarray(indices_PFC)])
        HP_shuffle_dm = np.concatenate(np.asarray(all_subjects)[np.asarray(indices_HP)])
        
        PFC_shuffle_f = np.concatenate(np.asarray(all_subjects_firing)[np.asarray(indices_PFC)])
        HP_shuffle_f = np.concatenate(np.asarray(all_subjects_firing)[np.asarray(indices_HP)])
        HP_shuffle= [HP_shuffle_dm,HP_shuffle_f]
        PFC_shuffle= [PFC_shuffle_dm,PFC_shuffle_f]

        C_1_HP_b, C_2_HP_b, C_3_HP_b = time_in_block(HP_shuffle, area = 'HP', n = n, plot_a = False, plot_b = True,  perm = True)
        C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = time_in_block(PFC_shuffle, area = 'PFC', n = n, plot_a = False, plot_b = True,  perm = True)
        
        C_1_HP_a, C_2_HP_a, C_3_HP_a = time_in_block(HP_shuffle, area = 'HP', n = n, plot_a = True, plot_b = False,  perm = True)
        C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = time_in_block(PFC_shuffle, area = 'PFC', n = n, plot_a = True, plot_b = False ,  perm = True)
        
        value_to_value_PFC_a_perm = generalisation_plot(C_1_PFC_a,C_2_PFC_a,C_3_PFC_a, c_1,reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_PFC_b_perm = generalisation_plot(C_1_PFC_b,C_2_PFC_b,C_3_PFC_b, c_1,reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_PFC_a_perm = value_to_value_PFC_a_perm[:-1]
        value_to_value_PFC_b_perm = value_to_value_PFC_b_perm[:-1]

    
        value_to_value_HP_a_perm =  generalisation_plot(C_1_HP_a,C_2_HP_a,C_3_HP_a, c_1,reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_HP_b_perm = generalisation_plot(C_1_HP_b,C_2_HP_b,C_3_HP_b, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_HP_a_perm = value_to_value_HP_a_perm[:-1]
        value_to_value_HP_b_perm = value_to_value_HP_b_perm[:-1]
        
      

        difference_a.append((value_to_value_PFC_a_perm-value_to_value_HP_a_perm))
        
        difference_b.append((value_to_value_PFC_b_perm-value_to_value_HP_b_perm))
        md = (value_to_value_HP_b_perm - value_to_value_HP_a_perm) - (value_to_value_PFC_b_perm-value_to_value_PFC_a_perm)
        meta_difference.append(md)
        
         
    perm_a = np.max(np.percentile(difference_a,95,0),1)
    
    perm_b = np.max(np.percentile(difference_b,95,0),1)
    perm_md = np.max(np.percentile(meta_difference,95,0),1)

    a = (value_to_value_PFC_a_val - value_to_value_HP_a_val)
    b = (value_to_value_PFC_b_val - value_to_value_HP_b_val)
    real_md = (value_to_value_HP_b_val - value_to_value_HP_a_val) - (value_to_value_PFC_b_val-value_to_value_PFC_a_val)
   
    perms_b = np.where(b.T > perm_b)
    perms_a = np.where(a.T > perm_a)
    perms_md = np.where(real_md.T >perm_md)
    
    return perms_b,perms_a,perms_md

def perumute_sessions(HP, PFC, c_1 = 1, n = 6 , reward_times_to_choose = [1,2,3,4], task_check = 0, perm_n = 500):
    
 
    C_1_HP_b, C_2_HP_b, C_3_HP_b = time_in_block(HP, area = 'HP', n = n, plot_a = False, plot_b = True, perm = False)
    C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = time_in_block(PFC, area = 'PFC', n = n, plot_a = False, plot_b = True, perm = False)
    
    C_1_HP_a, C_2_HP_a, C_3_HP_a = time_in_block(HP, area = 'HP', n = n, plot_a = True, plot_b = False, perm = False)
    C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = time_in_block(PFC, area = 'PFC', n = n, plot_a = True, plot_b = False, perm = False)
   
   
    value_to_value_PFC_a_val = generalisation_plot(C_1_PFC_a,C_2_PFC_a,C_3_PFC_a, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_PFC_b_val = generalisation_plot(C_1_PFC_b,C_2_PFC_b,C_3_PFC_b, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_PFC_a_val = value_to_value_PFC_a_val[:-1]
    value_to_value_PFC_b_val = value_to_value_PFC_b_val[:-1]

    value_to_value_HP_a_val = generalisation_plot(C_1_HP_a,C_2_HP_a,C_3_HP_a, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_HP_b_val = generalisation_plot(C_1_HP_b,C_2_HP_b,C_3_HP_b, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_HP_a_val = value_to_value_HP_a_val[:-1]
    value_to_value_HP_b_val = value_to_value_HP_b_val[:-1]

    difference_a = []
    difference_b = []
    difference_a_b = []

    all_subjects = np.hstack([PFC['DM'][0], HP['DM'][0]])
    all_subjects_firing = np.hstack([PFC['Data'][0], HP['Data'][0]])
    
    sessions_n = np.arange(len(all_subjects))
  
    for i in range(perm_n):
        np.random.shuffle(sessions_n) # Shuffle PFC/HP sessions
        indices_HP = sessions_n[46:]
        indices_PFC = sessions_n[:46]

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
        
        value_to_value_PFC_a_perm = generalisation_plot(C_1_PFC_a,C_2_PFC_a,C_3_PFC_a, c_1,reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_PFC_b_perm = generalisation_plot(C_1_PFC_b,C_2_PFC_b,C_3_PFC_b, c_1,reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_PFC_a_perm = value_to_value_PFC_a_perm[:-1]
        value_to_value_PFC_b_perm = value_to_value_PFC_b_perm[:-1]

    
        value_to_value_HP_a_perm =  generalisation_plot(C_1_HP_a,C_2_HP_a,C_3_HP_a, c_1,reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_HP_b_perm = generalisation_plot(C_1_HP_b,C_2_HP_b,C_3_HP_b, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_HP_a_perm = value_to_value_HP_a_perm[:-1]
        value_to_value_HP_b_perm = value_to_value_HP_b_perm[:-1]
        
      

        difference_a.append((value_to_value_PFC_a_perm-value_to_value_HP_a_perm))
        
        difference_b.append((value_to_value_PFC_b_perm-value_to_value_HP_b_perm))
        
        difference_a_b.append((value_to_value_PFC_b_perm-value_to_value_PFC_a_perm)-(value_to_value_HP_b_perm-value_to_value_HP_a_perm))
        
        
        
         
    perm_a = np.max(np.percentile(difference_a,95,0),1)
    
    perm_b = np.max(np.percentile(difference_b,95,0),1)
    perm_a_b = np.max(np.percentile(difference_a_b,95,0),1)
    

    a = (value_to_value_PFC_a_val - value_to_value_HP_a_val)
    b = (value_to_value_PFC_b_val - value_to_value_HP_b_val)
    a_b = (value_to_value_PFC_b_val- value_to_value_PFC_a_val)- (value_to_value_HP_b_val- value_to_value_HP_a_val)
   
    perms_b = np.where(b.T > perm_b)
    perms_a = np.where(a.T > perm_a)
    perms_ab = np.where(a_b.T > perm_a_b)
    
    return perms_b,perms_a, perms_ab


    
def time_in_block(data, area = 'PFC', n = 10, plot_a = False, plot_b = False, perm = True):
   
    if perm:
        dm = data[0]
        firing = data[1]

    else:
        dm = data['DM'][0]
        firing = data['Data'][0]

    C_1 = []; C_2 = []; C_3 = []
    cpd_1 = []; cpd_2 = []; cpd_3 = []
    average = rew_prev_behaviour(data, n = n, perm = perm)

    for  s, sess in enumerate(dm):
        
       
        DM = dm[s]
        firing_rates = firing[s]
       # firing_rates = scipy.stats.zscore(firing_rates,0)
        #firing_rates = firing_rates - np.mean(firing_rates,0)

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
        #average_val_ex_ch = np.concatenate([average[n].reshape(1),average[n*2:]])
        #X_exl_1 = np.concatenate([X_1[:,n].reshape(len(X_1),1),X_1[:,n*2:]],1)
        #value = np.matmul(X[:,n*2:], average[n*2:])
        value_1 =np.matmul(X_1, average)
        #value_1 =np.matmul(X_exl_1, average_val_ex_ch)

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
           # rewards_1 = scipy.stats.zscore(rewards_1)
          #  value_1 = scipy.stats.zscore(value_1)

          
        elif plot_b == True:
            
            rewards_1 = rewards_1[b_1] 
            choices_1 = choices_1[b_1]
            value_1 = value_1[b_1]
            ones_1  = ones_1[b_1]
            firing_rates_1 = firing_rates_1[b_1]
          #  rewards_1 = scipy.stats.zscore(rewards_1)
          #  value_1 = scipy.stats.zscore(value_1)

        predictors_all = OrderedDict([
                                #    ('Choice', choices_1),
                                    ('Reward', rewards_1),
                                    ('Value',value_1), 
                                   #  ('Value Сhoice',value_1_choice_1), 
                               #     ('Prev Rew Ch', prev_ch_reward_1),

                           #      ('Prev Rew', prev_reward_1),
                                 #  ('Prev Ch', prev_choice_1),
                                   ('ones', ones_1)
                                    ])
        
        X_1 = np.vstack(predictors_all.values()).T[:trials_1,:].astype(float)
        
        n_predictors = X_1.shape[1]
        y_1 = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
       # tstats,x = regression_code_session(y_1, X_1)
        #tstats =  reg_f.regression_code(y_1, X_1)
        ols = LinearRegression()
        ols.fit(X_1,y_1)
        C_1.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
        #C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_1.append(_CPD(X_1,y_1).reshape(n_neurons, n_timepoints, n_predictors))
        
        
        rewards_2 = reward_current[task_2]
        choices_2 = choices_current[task_2]
        
        previous_rewards_2 = scipy.linalg.toeplitz(rewards_2, np.zeros((1,n)))[n-1:-1]
         
        previous_choices_2 = scipy.linalg.toeplitz(0.5-choices_2, np.zeros((1,n)))[n-1:-1]
         
        interactions_2 = scipy.linalg.toeplitz((((0.5-choices_2)*(rewards_2-0.5))*2),np.zeros((1,n)))[n-1:-1]
         

        ones = np.ones(len(interactions_2)).reshape(len(interactions_2),1)
         
        X_2 = np.hstack([previous_rewards_2,previous_choices_2,interactions_2,ones])
        # average_val_ex_ch = np.concatenate([average[n].reshape(1),average[n*2:]])
        # X_exl_2 = np.concatenate([X_2[:,n].reshape(len(X_2),1),X_2[:,n*2:]],1)
        # value = np.matmul(X[:,n*2:], average[n*2:])
        value_2 =np.matmul(X_2, average)
        #value_2 =np.matmul(X_exl_2, average_val_ex_ch)

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
           # rewards_2 = scipy.stats.zscore(rewards_2)
           # value_2 = scipy.stats.zscore(value_2)

        elif plot_b == True:
            
            rewards_2 = rewards_2[b_2] 
            choices_2 = choices_2[b_2]
            value_2 = value_2[b_2]
            ones_2  = ones_2[b_2]
            firing_rates_2 = firing_rates_2[b_2]
           # rewards_2 = scipy.stats.zscore(rewards_2)
           # value_2 = scipy.stats.zscore(value_2)

        predictors_all = OrderedDict([
                                #     ('Choice', choices_2),
                                    ('Reward', rewards_2),
                                    ('Value',value_2), 
                                   #  ('Value Сhoice',value_2_choice_2), 
                                 #   ('Prev Rew Ch', prev_ch_reward_2),
#
                               #    ('Prev Rew', prev_reward_2),
                                  # ('Prev Ch', prev_choice_2),
                                    ('ones', ones_2)
                                    ])
        
        X_2 = np.vstack(predictors_all.values()).T[:trials_2,:].astype(float)
        
        n_predictors = X_2.shape[1]
        y_2 = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
       # tstats,x = regression_code_session(y_2, X_2)
        ols = LinearRegression()
        ols.fit(X_2,y_2)
        C_2.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
        cpd_2.append(_CPD(X_2,y_2).reshape(n_neurons, n_timepoints, n_predictors))
  
    
        
        rewards_3 = reward_current[task_3]
        choices_3 = choices_current[task_3]
        
        previous_rewards_3 = scipy.linalg.toeplitz(rewards_3, np.zeros((1,n)))[n-1:-1]
         
        previous_choices_3 = scipy.linalg.toeplitz(0.5-choices_3, np.zeros((1,n)))[n-1:-1]
         
        interactions_3 = scipy.linalg.toeplitz((((0.5-choices_3)*(rewards_3-0.5))*2),np.zeros((1,n)))[n-1:-1]
         

        ones = np.ones(len(interactions_3)).reshape(len(interactions_3),1)
         
        X_3 = np.hstack([previous_rewards_3,previous_choices_3,interactions_3,ones])
       #  average_val_ex_ch = np.concatenate([average[n].reshape(1),average[n*2:]])
       #  X_exl_3 = np.concatenate([X_3[:,n].reshape(len(X_3),1),X_3[:,n*2:]],1)
       # # value = np.matmul(X[:,n*2:], average[n*2:])
        value_3 =np.matmul(X_3, average)
        # value_3 =np.matmul(X_exl_3, average_val_ex_ch)

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
          #  rewards_3 = scipy.stats.zscore(rewards_3)
          #  value_3 = scipy.stats.zscore(value_3)

           
        elif plot_b == True:
            rewards_3 = rewards_3[b_3] 
            choices_3 = choices_3[b_3]
          
            value_3 = value_3[b_3]
            ones_3  = ones_3[b_3]

            firing_rates_3 = firing_rates_3[b_3]
         #   rewards_3 = scipy.stats.zscore(rewards_3)
         #   value_3 = scipy.stats.zscore(value_3)

           
          
  
        predictors_all = OrderedDict([
                                  #   ('Ch', choices_3),
                                    ('Rew', rewards_3),
                                    ('Value',value_3), 
                                   # ('Value Сhoice',value_3_choice_3), 
#                                   
                                 #   ('Prev Rew Ch', prev_ch_reward_3),
                              #    ('Prev Rew', prev_reward_3),
                                  # ('Prev Ch', prev_choice_3),
                                    ('ones', ones_3)
                                    ])
        
        X_3 = np.vstack(predictors_all.values()).T[:trials_3,:].astype(float)
        rank = np.linalg.matrix_rank(X_1)
        print(rank)
        n_predictors = X_3.shape[1]
        print(n_predictors)
        y_3 = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        #tstats,x = regression_code_session(y_3, X_3)
        ols = LinearRegression()
        ols.fit(X_3,y_3)
        C_3.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings

       # C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_3.append(_CPD(X_3,y_3).reshape(n_neurons, n_timepoints, n_predictors))
        
    
    C_1 = np.concatenate(C_1,0)
    
    C_2 = np.concatenate(C_2,0)
    
    C_3 = np.concatenate(C_3,0)
   
    C_2_inf = [~np.isinf(C_2[0]).any(axis=1)]; C_2_nan = [~np.isnan(C_2[0]).any(axis=1)]
    C_3_inf = [~np.isinf(C_3[0]).any(axis=1)];  C_3_nan = [~np.isnan(C_3[0]).any(axis=1)]
    C_1_inf = [~np.isinf(C_1[0]).any(axis=1)];  C_1_nan = [~np.isnan(C_1[0]).any(axis=1)]
    nans = np.asarray(C_1_inf) & np.asarray(C_1_nan) & np.asarray(C_3_inf) & np.asarray(C_3_nan) & np.asarray(C_2_inf)& np.asarray(C_2_nan)
    C_1 = np.transpose(C_1[:,nans[0],:],[2,0,1]); C_2 = np.transpose(C_2[:,nans[0],:],[2,0,1]);  C_3 = np.transpose(C_3[:,nans[0],:],[2,0,1])
   
    
    cpd_1 = np.nanmean(np.concatenate(cpd_1,0), axis = 0)
    cpd_2 = np.nanmean(np.concatenate(cpd_2,0), axis = 0)
    cpd_3 = np.nanmean(np.concatenate(cpd_3,0), axis = 0)
    cpd = np.nanmean([cpd_1,cpd_2,cpd_3],0)

    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors+wes.Moonrise1_5.mpl_colors
        
    # j = 0
    # plt.figure()
    # pred = list(predictors_all.keys())
    # pred = pred
    # for ii,i in enumerate(cpd.T):
    #     plt.plot(i, color = c[j],label = pred[j])
      
    #     j+=1
    # plt.legend()
    # plt.title(area)

    # sns.despine()

  
    return C_1,C_2,C_3
    
def a_b_check(PFC, HP, n =11, perm = 1000,task_check = 2, reward_times_to_choose = np.asarray([20,25,35,42]), c_1 = 1):
    
     C_1_a_PFC, C_1_b_PFC, C_2_a_PFC, C_2_b_PFC,  C_3_a_PFC, C_3_b_PFC =  perm_a_b(PFC, area = 'PFC', n = n, perm = perm)
     C_1_a_HP, C_1_b_HP, C_2_a_HP, C_2_b_HP,  C_3_a_HP, C_3_b_HP =  perm_a_b(HP, area = 'HP', n = n, perm = perm)
     
     C_1_a_PFC = np.transpose(C_1_a_PFC,[0,3,1,2]);   C_1_b_PFC = np.transpose(C_1_b_PFC,[0,3,1,2])
     C_2_a_PFC = np.transpose(C_2_a_PFC,[0,3,1,2]);   C_2_b_PFC = np.transpose(C_2_b_PFC,[0,3,1,2])
     C_3_a_PFC = np.transpose(C_3_a_PFC,[0,3,1,2]);   C_3_b_PFC = np.transpose(C_3_b_PFC,[0,3,1,2])

     C_1_a_HP = np.transpose(C_1_a_HP,[0,3,1,2]);   C_1_b_HP = np.transpose(C_1_b_HP,[0,3,1,2])
     C_2_a_HP = np.transpose(C_2_a_HP,[0,3,1,2]);   C_2_b_HP = np.transpose(C_2_b_HP,[0,3,1,2])
     C_3_a_HP = np.transpose(C_3_a_HP,[0,3,1,2]);   C_3_b_HP = np.transpose(C_3_b_HP,[0,3,1,2])

   
     difference_a_b_HP = []
     difference_a_b_PFC =[]
     
  
     for n in range(perm):
         value_to_value_PFC_a_perm = generalisation_plot(C_1_a_PFC[n],C_2_a_PFC[n],C_3_a_PFC[n], c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
         value_to_value_PFC_b_perm  = generalisation_plot(C_1_b_PFC[n],C_2_b_PFC[n],C_3_b_PFC[n], c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        
    
         value_to_value_HP_a_perm  = generalisation_plot(C_1_a_HP[n],C_2_a_HP[n],C_3_a_HP[n], c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
         value_to_value_HP_b_perm  = generalisation_plot(C_1_b_HP[n],C_2_b_HP[n],C_3_b_HP[n], c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
         
         
         difference_a_b_HP.append((value_to_value_HP_a_perm-value_to_value_HP_b_perm))
        
         difference_a_b_PFC.append((value_to_value_PFC_a_perm-value_to_value_PFC_b_perm))
         
        
     
        
     perm_HP = np.max(np.percentile(difference_a_b_HP,95,0),1)
    
     perm_PFC = np.max(np.percentile(difference_a_b_PFC,95,0),1)
     return perm_HP,perm_PFC
   
    # perms_b = np.where(b.T > perm_b)
    # perms_a = np.where(a.T > perm_a)
   

def perm_a_b(data, area = 'PFC', n = 10, perm = 2):
  
    dm = data['DM'][0]
    firing = data['Data'][0]
    C_1_a  = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
    C_2_a  = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
    C_3_a = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
    
    C_1_b  = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
    C_2_b  = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
    C_3_b = [[] for i in range(perm)] # To store permuted predictor loadings for each session.

    average = rew_prev_behaviour(data, n = n, perm = False)

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
          
        
        a_1 = np.where(choices_1 == 0.5)[0]
        b_1 = np.where(choices_1 == -0.5)[0]
        a_2 = np.where(choices_2 == 0.5)[0]
        b_2 = np.where(choices_2 == -0.5)[0]
        a_3 = np.where(choices_3 == 0.5)[0]
        b_3 = np.where(choices_3 == -0.5)[0]
        rewards_1_a = rewards_1[a_1] 
        value_1_a = value_1[a_1]
        ones_1_a  = ones_1[a_1]
        rewards_1_b = rewards_1[b_1] 
        value_1_b = value_1[b_1]
        ones_1_b  = ones_1[b_1]
        
        rewards_2_a = rewards_2[a_2] 
        value_2_a = value_2[a_2]
        ones_2_a  = ones_2[a_2]
        rewards_2_b = rewards_2[b_2] 
        value_2_b = value_2[b_2]
        ones_2_b  = ones_2[b_2]
        
        rewards_3_a = rewards_3[a_3] 
        value_3_a = value_3[a_3]
        ones_3_a  = ones_3[a_3]
        rewards_3_b = rewards_3[b_3] 
        value_3_b = value_3[b_3]
        ones_3_b  = ones_3[b_3]
        for i in range(perm):
            b1_a1 = np.concatenate([a_1,b_1])
            np.random.shuffle(b1_a1)
            b_1 = b1_a1[:len(b_1)]
            a_1 = b1_a1[len(b_1):]
            
            b2_a2 = np.concatenate([a_2,b_2])
            np.random.shuffle(b2_a2)
            b_2 = b2_a2[:len(b_2)]
            a_2 = b2_a2[len(b_2):]
            
            b3_a3 = np.concatenate([a_3,b_3])
            np.random.shuffle(b3_a3)
            b_3 = b3_a3[:len(b_3)]
            a_3 = b3_a3[len(b_3):]
            
            # A
            
            firing_rates_1_a = firing_rates_1[a_1]
          
              
             
            predictors_all = OrderedDict([
                                    #    ('Choice', choices_1),
                                        ('Reward', rewards_1_a),
                                        ('Value',value_1_a), 
                                
                                       ('ones', ones_1_a)
                                        ])
            
            X_1 = np.vstack(predictors_all.values()).T[:trials_1,:].astype(float)
            
            n_predictors = X_1.shape[1]
            y_1 = firing_rates_1_a.reshape([len(firing_rates_1_a),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression()
            ols.fit(X_1,y_1)
            C_1_a[i].append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
            
            # B
               
            
            firing_rates_1_b = firing_rates_1[b_1]
            
            
              
            predictors_all = OrderedDict([
                                    #    ('Choice', choices_1),
                                        ('Reward', rewards_1_b),
                                        ('Value',value_1_b), 
                                
                                       ('ones', ones_1_b)
                                        ])
            
            X_1 = np.vstack(predictors_all.values()).T[:trials_1,:].astype(float)
            
            n_predictors = X_1.shape[1]
            y_1 = firing_rates_1_b.reshape([len(firing_rates_1_b),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression()
            ols.fit(X_1,y_1)
            C_1_b[i].append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
        
            # pdes = np.linalg.pinv(X_1)
  
            # pe = np.matmul(pdes,y_1)
            
            # res = y_1 - np.matmul(X_1,pe)

            
            firing_rates_2_a = firing_rates_2[a_2]
             

            predictors_all = OrderedDict([
                                    #     ('Choice', choices_2),
                                        ('Reward', rewards_2_a),
                                        ('Value',value_2_a), 
                                     
                                        ('ones', ones_2_a)
                                        ])
            
            X_2 = np.vstack(predictors_all.values()).T[:trials_2,:].astype(float)
            
            n_predictors = X_2.shape[1]
            y_2 = firing_rates_2_a.reshape([len(firing_rates_2_a),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression()
            ols.fit(X_2,y_2)
            C_2_a[i].append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
      
        
      
            firing_rates_2_b = firing_rates_2[b_2]
             
            predictors_all = OrderedDict([
                                    #     ('Choice', choices_2),
                                        ('Reward', rewards_2_b),
                                        ('Value',value_2_b), 
                                      
                                        ('ones', ones_2_b)
                                        ])
            
            X_2 = np.vstack(predictors_all.values()).T[:trials_2,:].astype(float)
            
            n_predictors = X_2.shape[1]
            y_2 = firing_rates_2_b.reshape([len(firing_rates_2_b),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression()
            ols.fit(X_2,y_2)
            C_2_b[i].append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
     
    
            firing_rates_3_a = firing_rates_3[a_3]
               
               
      
            predictors_all = OrderedDict([
                                      #   ('Ch', choices_3),
                                        ('Rew', rewards_3_a),
                                        ('Value',value_3_a), 
                                      
                                        ('ones', ones_3_a)
                                        ])
            
            X_3 = np.vstack(predictors_all.values()).T[:trials_3,:].astype(float)
            rank = np.linalg.matrix_rank(X_1)
            print(rank)
            n_predictors = X_3.shape[1]
            print(n_predictors)
            y_3 = firing_rates_3_a.reshape([len(firing_rates_3_a),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression()
            ols.fit(X_3,y_3)
            C_3_a[i].append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
    

            firing_rates_3_b = firing_rates_3[b_3]
           
              
      
            predictors_all = OrderedDict([
                                      #   ('Ch', choices_3),
                                        ('Rew', rewards_3_b),
                                        ('Value',value_3_b), 
                                      
                                        ('ones', ones_3_b)
                                        ])
            
            X_3 = np.vstack(predictors_all.values()).T[:trials_3,:].astype(float)
            rank = np.linalg.matrix_rank(X_1)
            print(rank)
            n_predictors = X_3.shape[1]
            print(n_predictors)
            y_3 = firing_rates_3_b.reshape([len(firing_rates_3_b),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression()
            ols.fit(X_3,y_3)
            C_3_b[i].append((ols.coef_.reshape(n_neurons, n_timepoints, n_predictors))) # Predictor loadings
    
    if perm: # Evaluate P values.
       C_1_a   = np.stack([np.concatenate(C_i,0) for C_i in C_1_a],0)
       C_1_b   = np.stack([np.concatenate(C_i,0) for C_i in C_1_b],0)

       C_2_a   = np.stack([np.concatenate(C_i,0) for C_i in C_2_a],0)
       C_2_b   = np.stack([np.concatenate(C_i,0) for C_i in C_2_b],0)

       C_3_a   = np.stack([np.concatenate(C_i,0) for C_i in C_3_a],0)
       C_3_b   = np.stack([np.concatenate(C_i,0) for C_i in C_3_b],0)

  
    return C_1_a, C_1_b, C_2_a, C_2_b,  C_3_a, C_3_b
  
 

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
    cpd_HP,cpd_perm_HP =  perm_roll(HP, n = 11, perm = 2)
    cpd_PFC,cpd_perm_PFC =  perm_roll(PFC, n = 11, perm = 2)
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
        
   
  
