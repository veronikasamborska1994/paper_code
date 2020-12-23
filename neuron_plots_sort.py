#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:32:07 2020

@author: veronikasamborska
"""


from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pylab as plt
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/time_analysis')

from sklearn.linear_model import LinearRegression
import seaborn as sns
from collections import OrderedDict
import regression_function as reg_f
from palettable import wesanderson as wes
from scipy import io
from itertools import combinations 
import regressions as re 
import scipy
import palettable
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)
import value_reg as vg

def load():
    
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap


def heatplot_sort(data, ch = 1, task_1_2  =  True, task_1_3 =  False, task_2_3 =  False): 
    
    dm = data['DM'][0]
    fr = data['Data'][0]

    neurons = 0
    for s in fr:
        neurons += s.shape[1]
    ch_1_r = np.zeros((neurons,63));  ch_2_r= np.zeros((neurons,63))
    ch_1_nr = np.zeros((neurons,63));  ch_2_nr= np.zeros((neurons,63))

    n_neurons_cum = 0

    for  s, sess in enumerate(fr):
        DM = dm[s]
       
       
        firing_rates = fr[s]

        n_trials, n_neurons, n_timepoints = firing_rates.shape
        n_neurons_cum += n_neurons
      
        choices = DM[:,1]
        reward = DM[:,2]  

        task =  DM[:,5]
        a_pokes = DM[:,6]
        b_pokes = DM[:,7]

          
        taskid = vg.task_ind(task, a_pokes, b_pokes)
      
        
        if task_1_2 == True:
            
            taskid_1 = 1
            taskid_2 = 2
            task_title = '1 2'
            
        elif task_2_3 == True:
            
            taskid_1 = 2
            taskid_2 = 3
            task_title = '2 3'

        elif task_1_3 == True:
            
            taskid_1 = 1
            taskid_2 = 3
            task_title = '1 3'

        task_1_r = np.where((taskid == taskid_1) & (choices == ch) & (reward == 1))[0] # Find indicies for task 1 A
        task_1_nr = np.where((taskid == taskid_1) & (choices == ch & (reward == 0)))[0] # Find indicies for task 1 A
 
        task_2_r = np.where((taskid == taskid_2) & (choices == ch)& (reward == 1))[0] # Find indicies for task 1 A
        task_2_nr = np.where((taskid == taskid_2) & (choices == ch)& (reward == 0))[0] # Find indicies for task 1 A
   
    
                        
        ch_1_r[n_neurons_cum-n_neurons:n_neurons_cum,:] =  np.mean(firing_rates[task_1_r,:, :],0) 
        ch_1_nr[n_neurons_cum-n_neurons:n_neurons_cum,:] = np.mean(firing_rates[task_1_nr,:, :],0)
        ch_2_r[n_neurons_cum-n_neurons:n_neurons_cum,:] = np.mean(firing_rates[task_2_r,:, :],0)
        ch_2_nr[n_neurons_cum-n_neurons:n_neurons_cum,:] = np.mean(firing_rates[task_2_nr,:, :],0)
            
    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    #cmap =  'bone'
  
    ch_1 = np.mean([ch_1_r,ch_1_nr],0)
    ch_2 = np.mean([ch_2_r,ch_2_nr],0)

    ch_1 = ch_1/(np.tile(np.max(ch_1,1), [ch_1.shape[1],1]).T+1e-08)
#
    ch_2 = ch_2/(np.tile(np.max(ch_2,1), [ch_2.shape[1],1]).T+1e-08)

    peak_ind_1 = np.argmax(ch_1,1) 

    ordering_1 = np.argsort(peak_ind_1)
    activity_sorted_1 = ch_1[ordering_1,:]
    plt.subplot(2,2,1)
    plt.imshow(activity_sorted_1, aspect='auto', cmap = cmap)  
    plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    

 
    activity_sorted_2_1 = ch_2[ordering_1,:]
    plt.subplot(2,2,2)
    plt.imshow(activity_sorted_2_1, aspect='auto', cmap = cmap)  
    plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    

    peak_ind_2 = np.argmax(ch_2,1)
    ordering_2 = np.argsort(peak_ind_2)
    
    activity_sorted_2_2 = ch_2[ordering_2,:]
    plt.subplot(2,2,4)
    plt.imshow(activity_sorted_2_2, aspect='auto', cmap = cmap)  
    plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    

    activity_sorted_1_2= ch_1[ordering_2,:]
    plt.subplot(2,2,3)
    plt.imshow(activity_sorted_1_2, aspect='auto',cmap = cmap)  
    plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    
 
    choice_argmax = np.where((np.argmax(ch_1,1)> 30) & (np.argmax(ch_1,1) <42))[0]
    plt.plot()
   
    init_argmax_2 = np.where((np.argmax(ch_2,1)> 22) & (np.argmax(ch_2,1) < 28))[0]
   
    # 2 to 3, B to Initiation
      
    task_1_init = np.mean(ch_1[init_argmax_2],0)
    task_1_init_std = np.std(ch_1[init_argmax_2],0)/np.sqrt(len(ch_1[init_argmax_2]))

     
    task_2_choice = np.mean(ch_2[choice_argmax],0)
    task_2_init_std = np.std(ch_2[choice_argmax],0)/np.sqrt(len(ch_2[choice_argmax]))

    # 
    plt.figure(figsize=(7,3))
    
    plt.subplot(1,2,1)
    plt.plot(task_2_choice,color = 'grey')
    plt.fill_between(np.arange(len(task_2_choice)), task_2_choice-task_2_init_std, task_2_choice+task_2_init_std, alpha=0.2,color = 'grey')

    plt.title('firing in task 3 of cells \n responding to choice in task 2 ')
    plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    

    plt.subplot(1,2,2)
    plt.plot(task_1_init,color = 'pink')
    plt.fill_between(np.arange(len(task_1_init)), task_1_init-task_1_init_std, task_1_init+task_1_init_std, alpha=0.2,color = 'pink')

    plt.title('firing in task 2 of cells \n responding to initiation in task 3 ')
    plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    

    sns.despine()
 
#plotting_from_dat(PFC, area = 'PFC_new')
#plotting_from_dat(HP, area = 'HP_new')

def plotting_from_dat(data, area = 'PFC'):
    
   
    pdf = PdfPages('/Users/veronikasamborska/Desktop/'+ area+'from_dat.pdf')
    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    
     
    dm = data['DM'][0]
    fr = data['Data'][0]
    neuron_ID = 0

    for s,session in enumerate(dm):     
        # Get raw spike data across the task 
        firing_rate = fr[s]
        DM = dm[s]
        n_trials, n_neurons, n_times = firing_rate.shape

        choices = DM[:,1]
        b_pokes = DM[:,7]
        a_pokes = DM[:,6]
        i_pokes = DM[:,8]
        reward = DM[:,2]  

        task = DM[:,5]
        _t_1 = np.where(task ==1)[0]
        _t_2 = np.where(task ==2)[0]
        _t_3 = np.where(task ==3)[0]

         
        x_points = [132,232,232,332,332,432,432,532]
        y_points = [2.8,3.8,1.8,4.8,0.8,3.8,1.8,2.8]
        
        global _4; global _2; global _7; global _1; global _9; global _3;global _8; global _6

        _4 = [x_points[0],y_points[0]]
        _2 = [x_points[1],y_points[1]]
        _7 = [x_points[2],y_points[2]]
        _1 = [x_points[3],y_points[3]]
        _9 = [x_points[4],y_points[4]]
        _3 = [x_points[5],y_points[5]]
        _8 = [x_points[6],y_points[6]]
        _6 = [x_points[7],y_points[7]]

        a_1 = globals()['_' +str(int(a_pokes[_t_1][0]))]; a_2 = globals()['_' +str(int(a_pokes[_t_2][0]))]; a_3 = globals()['_' +str(int(a_pokes[_t_3][0]))]
        b_1 = globals()['_' +str(int(b_pokes[_t_1][0]))]; b_2 = globals()['_' +str(int(b_pokes[_t_2][0]))]; b_3 = globals()['_' +str(int(b_pokes[_t_3][0]))]
        i_1 = globals()['_' +str(int(i_pokes[_t_1][0]))]; i_2 = globals()['_' +str(int(i_pokes[_t_2][0]))]; i_3 = globals()['_' +str(int(i_pokes[_t_3][0]))]
 
        ind_init = 25
        ind_choice = 36
        ind_reward = 42
        
        a_rew_1_f = np.mean(firing_rate[np.where((task ==1)& (choices ==1) & (reward == 1))[0]],0)
        a_nrew_1_f = np.mean(firing_rate[np.where((task ==1)& (choices ==1) & (reward == 0))[0]],0)
        
        a_rew_2_f = np.mean(firing_rate[np.where((task ==2)& (choices ==1) & (reward == 1))[0]],0)
        a_nrew_2_f = np.mean(firing_rate[np.where((task ==2)& (choices ==1) & (reward == 0))[0]],0)
    
        a_rew_3_f = np.mean(firing_rate[np.where((task ==3)& (choices ==1) & (reward == 1))[0]],0)
        a_nrew_3_f = np.mean(firing_rate[np.where((task ==3)& (choices ==1) & (reward == 0))[0]],0)
    
    
        b_rew_1_f = np.mean(firing_rate[np.where((task ==1)& (choices == 0) & (reward == 1))[0]],0)
        b_nrew_1_f = np.mean(firing_rate[np.where((task ==1)& (choices == 0) & (reward == 0))[0]],0)
        
        b_rew_2_f = np.mean(firing_rate[np.where((task ==2)& (choices == 0) & (reward == 1))[0]],0)
        b_nrew_2_f = np.mean(firing_rate[np.where((task ==2)& (choices == 0) & (reward == 0))[0]],0)
    
        b_rew_3_f = np.mean(firing_rate[np.where((task ==3)& (choices == 0) & (reward == 1))[0]],0)
        b_nrew_3_f = np.mean(firing_rate[np.where((task ==3)& (choices == 0) & (reward == 0))[0]],0)
    
    
        i_rew_1_f = np.mean(firing_rate[np.where((task ==1)&  (reward == 1))[0]],0)
        i_nrew_1_f = np.mean(firing_rate[np.where((task ==1)&  (reward == 0))[0]],0)
        
        i_rew_2_f = np.mean(firing_rate[np.where((task ==2)&  (reward == 1))[0]],0)
        i_nrew_2_f = np.mean(firing_rate[np.where((task ==2)&  (reward == 0))[0]],0)
    
        i_rew_3_f = np.mean(firing_rate[np.where((task ==3)&  (reward == 1))[0]],0)
        i_nrew_3_f = np.mean(firing_rate[np.where((task ==3)&  (reward == 0))[0]],0)
   
        plt.ioff()
        
      
        
        
        
        task_arrays = np.zeros(shape=(n_trials,3))
        task_arrays[:_t_2[0],0] = 1
        task_arrays[_t_2[0]:_t_3[0],1] = 1
        task_arrays[_t_3[0]:,2] = 1
        
        isl = wes.Royal2_5.mpl_colors
        isl_1 = wes.Moonrise5_6.mpl_colors


 
        vector_for_normalising = np.concatenate([a_rew_1_f,a_nrew_1_f,a_rew_2_f,a_nrew_2_f,\
                                                 a_rew_3_f,a_nrew_3_f,b_rew_1_f,b_nrew_1_f,b_rew_2_f,b_nrew_2_f,\
                                                 b_rew_3_f,b_nrew_3_f,\
                                                 i_rew_1_f,i_nrew_1_f,i_rew_2_f,i_nrew_2_f,\
                                                 i_rew_3_f,i_nrew_3_f], axis = 1)
    
    
 
        normalised = vector_for_normalising/(np.tile(np.max(vector_for_normalising,1), [vector_for_normalising.shape[1],1]).T+.0000001)
            
        a_rew_1_f = normalised[:, :n_times]   
        a_nrew_1_f = normalised[:, n_times:n_times*2]
        
        a_rew_2_f = normalised[:,n_times*2:n_times*3]
        a_nrew_2_f = normalised[:, n_times*3:n_times*4]
        
        a_rew_3_f = normalised[:,n_times*4:n_times*5]
        a_nrew_3_f = normalised[:, n_times*5:n_times*6]
       
        b_rew_1_f = normalised[:, n_times*6:n_times*7]
        b_nrew_1_f = normalised[:,n_times*7:n_times*8]
        
        b_rew_2_f = normalised[:, n_times*8:n_times*9]
        b_nrew_2_f = normalised[:,n_times*9:n_times*10]
        
        b_rew_3_f = normalised[:, n_times*10:n_times*11]
        b_nrew_3_f = normalised[:, n_times*11:n_times*12]
       
        i_rew_1_f = normalised[:, n_times*12:n_times*13]
        i_nrew_1_f = normalised[:,n_times*13 :n_times*14]   
        
        i_rew_2_f = normalised[:, n_times*14 :n_times*15]
        i_nrew_2_f = normalised[:, n_times*15 :n_times*16]
        
        i_rew_3_f = normalised[:, n_times*16 :n_times*17]
        i_nrew_3_f = normalised[:, n_times*17 :]
       
                           
        for neuron in range(n_neurons):
            
            neuron_ID +=1
            #Port Firing
            fig = plt.figure(figsize=(8, 15))
            grid = plt.GridSpec(2, 1, hspace=0.7, wspace=0.4)
            fig.add_subplot(grid[0]) 

            plt.scatter(x_points,y_points,s =100, c = 'black')

            plt.plot(np.arange(a_1[0]-30,a_1[0]+33,1), a_rew_1_f[neuron]+a_1[1]+1, color = isl[0], label ='task 1 A')   
            plt.plot(np.arange(a_1[0]-30,a_1[0]+33,1), a_nrew_1_f[neuron]+a_1[1]+1, color = isl[0], linestyle = ':')   
  
            plt.plot(np.arange(a_2[0]-30,a_2[0]+33,1), a_rew_2_f[neuron]+a_2[1]+1, color = isl[1], label ='task 2 A')   
            plt.plot(np.arange(a_2[0]-30,a_2[0]+33,1), a_nrew_2_f[neuron]+a_2[1]+1, color = isl[1], linestyle = ':')   
  
            plt.plot(np.arange(a_3[0]-30,a_3[0]+33,1), a_rew_3_f[neuron]+a_3[1]+1, color = isl[2],label ='task 3 A')   
            plt.plot(np.arange(a_3[0]-30,a_3[0]+33,1), a_nrew_3_f[neuron]+a_3[1]+1, color = isl[2], linestyle = ':')   
  
             
            plt.plot(np.arange(b_1[0]-30,b_1[0]+33,1), b_rew_1_f[neuron]+b_1[1]+1, color = isl[0], label ='task 1 B')     
            plt.plot(np.arange(b_1[0]-30,b_1[0]+33,1), b_nrew_1_f[neuron]+b_1[1]+1, color = isl[0], linestyle = ':')   

            plt.plot(np.arange(b_2[0]-30,b_2[0]+33,1), b_rew_2_f[neuron]+b_2[1]+1, color = isl[1],  label ='task 2 B')      
            plt.plot(np.arange(b_2[0]-30,b_2[0]+33,1), b_nrew_2_f[neuron]+b_2[1]+1, color = isl[1], linestyle = ':')   
  
            plt.plot(np.arange(b_3[0]-30,b_3[0]+33,1), b_rew_3_f[neuron]+b_3[1]+1, color = isl[2],  label ='task 3 B')      
            plt.plot(np.arange(b_3[0]-30,b_3[0]+33,1), b_nrew_3_f[neuron]+b_3[1]+1, color = isl[2], linestyle = ':')   
                                
               
            plt.plot(np.arange(i_1[0]-30,i_1[0]+33,1), i_rew_1_f[neuron]+i_1[1]+1, color = isl_1[0],  label ='task 1 I')    
            plt.plot(np.arange(i_1[0]-30,i_1[0]+33,1), i_nrew_1_f[neuron]+i_1[1]+1, color = isl_1[0], linestyle = ':')   

            plt.plot(np.arange(i_2[0]-30,i_2[0]+33,1), i_rew_2_f[neuron]+i_2[1]+1, color = isl_1[1],  label ='task 2 I')     
            plt.plot(np.arange(i_2[0]-30,i_2[0]+33,1), i_nrew_2_f[neuron]+i_2[1]+1, color = isl_1[1], linestyle = ':')   
  
            plt.plot(np.arange(i_3[0]-30,i_3[0]+33,1), i_rew_3_f[neuron]+i_3[1]+1, color = isl_1[2],  label ='task 3 I')      
            plt.plot(np.arange(i_3[0]-30,i_3[0]+33,1), i_nrew_3_f[neuron]+i_3[1]+1, color = isl_1[2], linestyle = ':')   
            
            inds = [a_1, b_1, b_1, b_2, b_2, b_3, b_3, i_1, i_1, i_2,i_2, i_3,i_3]
            for i,ind in enumerate(inds):
                                         
                plt.vlines(np.arange(ind[0]-30,ind[0]+33,1)[ind_init], ymin = ind[1]+1, ymax = ind[1]+2,linestyle = '--' ,color = 'Grey', linewidth=0.5)
                plt.vlines(np.arange(ind[0]-30,ind[0]+33,1)[ind_choice], ymin = ind[1]+1, ymax = ind[1]+2, linestyle = '--', color = 'Black', linewidth=0.5)
                plt.vlines(np.arange(ind[0]-30,ind[0]+33,1)[ind_reward],  ymin = ind[1]+1, ymax = ind[1]+2, linestyle = '--', color = 'Pink', linewidth=0.5)

            plt.legend()
            # Pokes 
            plt.axis('off')
            
            # Heatmap  
            fig.add_subplot(grid[1]) 
            heatplot = firing_rate[:,neuron,:]
           # heatplot = (heatplot - np.min(heatplot,1)[:, None]) / (np.max(heatplot,1)[:, None]+1e-08 - np.min(heatplot,1)[:, None])
            heatplot = heatplot/(np.tile(np.max(heatplot,1), [heatplot.shape[1],1]).T+1e-08)
     
            heatplot_con = np.concatenate([heatplot,task_arrays], axis = 1)

            plt.imshow(heatplot_con,cmap= cmap,aspect = 'auto')
            plt.xticks([ind_init, ind_choice, ind_reward], ('I', 'C', 'O'))
            plt.title('{}'.format(neuron_ID))
            
            pdf.savefig()
            plt.clf()
    pdf.close()
    
    
 