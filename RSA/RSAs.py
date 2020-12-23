#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:19:56 2019

@author: veronikasamborska
"""
import numpy as np

def RSA_physical_rdm():

    # RSA Physical Space coding
    port_a_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_r[0:6] = True
    
    port_a_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_nr[0:6] = True
    
    port_a_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_r[0:6] = True
    
    port_a_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_nr[0:6] = True
    
    port_a_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_r[0:6] = True
    
    port_a_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_nr[0:6] = True
    
    port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(15)])
    port_2_initiation_task_1[6:8] = True
    
    port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(15)])
    port_2_initiation_task_2[6:8] = True
    
    port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(15)])
    port_3_initiation_task_3[8:11] = True
    
    port_3_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_r[8:11] =  True
    
    port_3_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_nr[8:11] =  True
    
    port_4_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_r[11:13] = True
    
    port_4_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_nr[11:13] = True
    
    port_5_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_r[13:15] =  True
    
    port_5_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_nr[13:15] =  True
    
    
    physical_rsa = np.vstack([port_a_choice_task_1_r,port_a_choice_task_1_nr,port_a_choice_task_2_r,port_a_choice_task_2_nr,\
                              port_a_choice_task_3_r,port_a_choice_task_3_nr,port_2_initiation_task_1,port_2_initiation_task_2,\
                              port_3_initiation_task_3,port_3_choice_task_2_r,port_3_choice_task_2_nr,port_4_choice_task_1_r,\
                              port_4_choice_task_1_nr,port_5_choice_task_3_r,port_5_choice_task_3_nr])

    return physical_rsa


def RSA_a_b_initiation_rdm():

    # RSA Physical Space coding
    
    port_a_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_r[0:6] = True
    
    port_a_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_nr[0:6] = True
    
    port_a_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_r[0:6] = True
    
    port_a_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_nr[0:6] = True
    
    port_a_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_r[0:6] = True
    
    port_a_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_nr[0:6] = True
    
    port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(15)])
    #port_2_initiation_task_1[6:9] = True
    
    port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(15)])
    #port_2_initiation_task_2[6:9] = True
    
    port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(15)])
    #port_3_initiation_task_3[6:9] = True
    
    port_3_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_r[9:15] =  True
    
    port_3_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_nr[9:15] =  True
    
    port_4_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_r[9:15] = True
    
    port_4_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_nr[9:15] = True
    
    port_5_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_r[9:15] =  True
    
    port_5_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_nr[9:15] =  True
    
    
    choice_ab_rsa = np.vstack([port_a_choice_task_1_r,port_a_choice_task_1_nr,port_a_choice_task_2_r,port_a_choice_task_2_nr,\
                              port_a_choice_task_3_r,port_a_choice_task_3_nr,port_2_initiation_task_1,port_2_initiation_task_2,\
                              port_3_initiation_task_3,port_3_choice_task_2_r,port_3_choice_task_2_nr,port_4_choice_task_1_r,\
                              port_4_choice_task_1_nr,port_5_choice_task_3_r,port_5_choice_task_3_nr])
    
    return choice_ab_rsa
   


def reward_rdm():  
    # RSA Physical Space coding
    port_a_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_r[0] = True
    port_a_choice_task_1_r[2] = True
    port_a_choice_task_1_r[4] = True
    
    port_a_choice_task_1_r[9] =  True
    port_a_choice_task_1_r[11] =  True
    port_a_choice_task_1_r[13] =  True
    

    port_a_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_nr[1] = True
    port_a_choice_task_1_nr[3] = True
    port_a_choice_task_1_nr[5] = True
    
    port_a_choice_task_1_nr[10] =  True
    port_a_choice_task_1_nr[12] =  True
    port_a_choice_task_1_nr[14] =  True
       
    port_a_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_r[0] = True
    port_a_choice_task_2_r[2] = True
    port_a_choice_task_2_r[4] = True   
     
    port_a_choice_task_2_r[9] =  True
    port_a_choice_task_2_r[11] =  True
    port_a_choice_task_2_r[13] =  True
    
    
    port_a_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_nr[1] = True
    port_a_choice_task_2_nr[3] = True
    port_a_choice_task_2_nr[5] = True
    
    port_a_choice_task_2_nr[10] =  True
    port_a_choice_task_2_nr[12] =  True
    port_a_choice_task_2_nr[14] =  True
      
    
    port_a_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_r[0] = True
    port_a_choice_task_3_r[2] = True
    port_a_choice_task_3_r[4] = True   
        
    port_a_choice_task_3_r[9] =  True
    port_a_choice_task_3_r[11] =  True
    port_a_choice_task_3_r[13] =  True
  
    port_a_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_nr[1] = True
    port_a_choice_task_3_nr[3] = True
    port_a_choice_task_3_nr[5] = True
        
    port_a_choice_task_3_nr[10] =  True
    port_a_choice_task_3_nr[12] =  True
    port_a_choice_task_3_nr[14] =  True
     
    port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(15)])
    
    port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(15)])
    
    port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(15)])
    
    port_3_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_r[0] = True
    port_3_choice_task_2_r[2] = True
    port_3_choice_task_2_r[4] = True

    port_3_choice_task_2_r[9] =  True
    port_3_choice_task_2_r[11] =  True
    port_3_choice_task_2_r[13] =  True

    port_3_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    
    port_3_choice_task_2_nr[1] = True
    port_3_choice_task_2_nr[3] = True
    port_3_choice_task_2_nr[5] = True

    port_3_choice_task_2_nr[10] =  True
    port_3_choice_task_2_nr[12] =  True
    port_3_choice_task_2_nr[14] =  True

    
    port_4_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_r[0] = True
    port_4_choice_task_1_r[2] = True
    port_4_choice_task_1_r[4] = True
    
    port_4_choice_task_1_r[9] =  True
    port_4_choice_task_1_r[11] =  True
    port_4_choice_task_1_r[13] =  True
    
    port_4_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_nr[1] = True
    port_4_choice_task_1_nr[3] = True
    port_4_choice_task_1_nr[5] = True
    
    port_4_choice_task_1_nr[10] =  True
    port_4_choice_task_1_nr[12] =  True
    port_4_choice_task_1_nr[14] =  True
    
    port_5_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_r[0] = True
    port_5_choice_task_3_r[2] = True
    port_5_choice_task_3_r[4] = True
        
    port_5_choice_task_3_r[9] =  True
    port_5_choice_task_3_r[11] =  True
    port_5_choice_task_3_r[13] =  True
        
    port_5_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_nr[1] = True
    port_5_choice_task_3_nr[3] = True
    port_5_choice_task_3_nr[5] = True

    port_5_choice_task_3_nr[10] =  True
    port_5_choice_task_3_nr[12] =  True
    port_5_choice_task_3_nr[14] =  True
    
    reward_no_reward = np.vstack([port_a_choice_task_1_r,port_a_choice_task_1_nr,port_a_choice_task_2_r,port_a_choice_task_2_nr,\
                              port_a_choice_task_3_r,port_a_choice_task_3_nr,port_2_initiation_task_1,port_2_initiation_task_2,\
                              port_3_initiation_task_3,port_3_choice_task_2_r,port_3_choice_task_2_nr,port_4_choice_task_1_r,\
                              port_4_choice_task_1_nr,port_5_choice_task_3_r,port_5_choice_task_3_nr])

    return reward_no_reward

def reward_choice_space():
    reward_no_reward = reward_rdm()
    choice_ab_rsa = RSA_a_b_initiation_rdm()
    reward_at_choices = reward_no_reward & choice_ab_rsa 
 
    return reward_at_choices
  
def reward_choice_space_specific():
    reward_no_reward = reward_rdm()
    space_rdm = RSA_physical_rdm()
    reward_at_space = reward_no_reward & space_rdm 
 
    return reward_at_space

def remapping_a_to_b():
    
    
    port_a_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_r[0:2] = True
    port_a_choice_task_1_r[11:15] =  True

    port_a_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_nr[0:2] = True
    port_a_choice_task_1_nr[11:15] =  True

    port_a_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    
    port_a_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    
    port_a_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    
    port_a_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    
    port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(15)])
    
    port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(15)])
    
    port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(15)])
    
    port_3_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    
    port_3_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    
    port_4_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_r[0:2] =  True
    port_4_choice_task_1_r[11:15] =  True

    port_4_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_nr[0:2] =  True
    port_4_choice_task_1_nr[11:15] =  True

    port_5_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_r[0:2] =  True
    port_5_choice_task_3_r[11:15] =  True
    
    port_5_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_nr[0:2] =  True
    port_5_choice_task_3_nr[11:15] =  True
        
    remapping_a_to_b = np.vstack([port_a_choice_task_1_r,port_a_choice_task_1_nr,port_a_choice_task_2_r,port_a_choice_task_2_nr,\
                              port_a_choice_task_3_r,port_a_choice_task_3_nr,port_2_initiation_task_1,port_2_initiation_task_2,\
                              port_3_initiation_task_3,port_3_choice_task_2_r,port_3_choice_task_2_nr,port_4_choice_task_1_r,\
                              port_4_choice_task_1_nr,port_5_choice_task_3_r,port_5_choice_task_3_nr])
   
    return remapping_a_to_b

def choice_vs_initiation():
    
    # RSA Physical Space coding
    
    port_a_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_r[0:6] = True
    port_a_choice_task_1_r[9:15] =  True
    
    port_a_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_nr[0:6] = True
    port_a_choice_task_1_nr[9:15] =  True

    port_a_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_r[0:6] = True
    port_a_choice_task_2_r[9:15] =  True

    port_a_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_nr[0:6] = True
    port_a_choice_task_2_nr[9:15] =  True

    port_a_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_r[0:6] = True
    port_a_choice_task_3_r[9:15] =  True

    port_a_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_nr[0:6] = True
    port_a_choice_task_3_nr[9:15] =  True

    port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(15)])
    port_2_initiation_task_1[6:9] = True
    
    port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(15)])
    port_2_initiation_task_2[6:9] = True
    
    port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(15)])
    port_3_initiation_task_3[6:9] = True
    
    port_3_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_r[0:6] = True
    port_3_choice_task_2_r[9:15] =  True
    
    port_3_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_nr[0:6] = True
    port_3_choice_task_2_nr[9:15] =  True
    
    port_4_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_r[0:6] = True
    port_4_choice_task_1_r[9:15] = True
    
    port_4_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_nr[0:6] = True
    port_4_choice_task_1_nr[9:15] = True
    
    port_5_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_r[0:6] = True
    port_5_choice_task_3_r[9:15] =  True
    
    port_5_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_nr[0:6] = True
    port_5_choice_task_3_nr[9:15] =  True
    
    
    choice_initiation_rsa = np.vstack([port_a_choice_task_1_r,port_a_choice_task_1_nr,port_a_choice_task_2_r,port_a_choice_task_2_nr,\
                              port_a_choice_task_3_r,port_a_choice_task_3_nr,port_2_initiation_task_1,port_2_initiation_task_2,\
                              port_3_initiation_task_3,port_3_choice_task_2_r,port_3_choice_task_2_nr,port_4_choice_task_1_r,\
                              port_4_choice_task_1_nr,port_5_choice_task_3_r,port_5_choice_task_3_nr])
     
    return choice_initiation_rsa


def a_bs_task_specific():
    
    # RSA Physical Space coding
    
    port_a_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_r[0:2] = True
    
    port_a_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_nr[0:2] = True

    port_a_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_r[2:4] = True

    port_a_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_nr[2:4] = True

    port_a_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_r[4:6] = True

    port_a_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_nr[4:6] = True

    port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(15)])
    
    port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(15)])
    
    port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(15)])
    
    port_3_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_r[9:11] = True

    port_3_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_nr[9:11] = True

    port_4_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_r[11:13] = True

    port_4_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_nr[11:13] = True

    port_5_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_r[13:15] = True

    port_5_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_nr[13:15] = True

    
    a_bs_task_specific_rsa = np.vstack([port_a_choice_task_1_r,port_a_choice_task_1_nr,port_a_choice_task_2_r,port_a_choice_task_2_nr,\
                              port_a_choice_task_3_r,port_a_choice_task_3_nr,port_2_initiation_task_1,port_2_initiation_task_2,\
                              port_3_initiation_task_3,port_3_choice_task_2_r,port_3_choice_task_2_nr,port_4_choice_task_1_r,\
                              port_4_choice_task_1_nr,port_5_choice_task_3_r,port_5_choice_task_3_nr])
   
    return a_bs_task_specific_rsa


    