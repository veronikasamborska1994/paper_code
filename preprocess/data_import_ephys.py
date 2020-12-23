#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:37:21 2018

@author: behrenslab
"""
# =============================================================================
# Create data objects with ephys and behaviour together, some utility funcs 
# =============================================================================

import os
import numpy as np
import data_import as di
import re
import datetime
import copy 
from datetime import datetime
import scipy 
import align_activity as aa
from collections import OrderedDict


def import_code(ephys_path,beh_path):
    subjects_ephys = os.listdir(ephys_path)
    subjects_ephys = [subject for subject in subjects_ephys if not subject.startswith('.')] #Exclude .DS store from subject list
    subjects_beh = os.listdir(beh_path)
    
    m480 = []
    m483 = []
    m479 = []
    m486 = []
    m478 = []
    m481 = []
    m484 = []
    
    for subject_ephys in subjects_ephys: 
        # Go through each animal
        subject_subfolder = ephys_path + '/' + subject_ephys
        subject_sessions = os.listdir(subject_subfolder)
        # List all ephys_sessions
        subject_sessions = [session for session in subject_sessions if not session.startswith('.')] #Exclude .DS store from subject list
        subject_sessions = [session for session in subject_sessions if not session.startswith('LFP')] #Exclude LFP from subject list
        subject_sessions = [session for session in subject_sessions if not session.startswith('MUA')] #Exclude MUA from subject list
        

        for session in subject_sessions:
            match_ephys = re.search(r'\d{4}-\d{2}-\d{2}', session)
            date_ephys = datetime.strptime(match_ephys.group(), '%Y-%m-%d').date()
            date_ephys = match_ephys.group()
            
            for subject in subjects_beh:
                if subject == subject_ephys:
                    subject_beh_subfolder = beh_path + '/' + subject
                    subject_beh_sessions = os.listdir(subject_beh_subfolder)
                    subject_beh_sessions = [session for session in subject_beh_sessions if not session.startswith('.')] #Exclude .DS store from subject list
                    for beh_session in subject_beh_sessions:
                        match_behaviour = re.search(r'\d{4}-\d{2}-\d{2}', beh_session)
                        date_behaviour = datetime.strptime(match_behaviour.group(), '%Y-%m-%d').date()
                        date_behaviour = match_behaviour.group()
                        if date_ephys == date_behaviour:
                            behaviour_path = subject_beh_subfolder +'/'+beh_session
                            behaviour_session = di.Session(behaviour_path)
                            neurons_path = subject_subfolder+'/'+session 
                            neurons = np.load(neurons_path)
                            neurons = neurons[:,~np.isnan(neurons[1,:])]
                            behaviour_session.ephys = neurons
                            
                                                   
                            # if behaviour_session.file_name != 'm479-2018-08-12-150904.txt'  and behaviour_session.file_name != 'm484-2018-08-12-150904.txt'\
                            # and behaviour_session.file_name !='m483-2018-07-27-164242.txt'  and  behaviour_session.file_name != 'm480-2018-08-22-111012.txt'\
                            # and behaviour_session.file_name != 'm479-2018-08-22-111012.txt' and  behaviour_session.file_name !='m480-2018-09-11-163452.txt'\
                            # and behaviour_session.file_name !='m484-2018-09-11-163452.txt'  and behaviour_session.file_name !='m483-2018-06-21-173958.txt'\
                            # and behaviour_session.file_name !='m483-2018-06-20-172510.txt':
                            if behaviour_session.file_name != 'm479-2018-08-12-150904.txt' and behaviour_session.file_name != 'm484-2018-08-12-150904.txt'\
                            and behaviour_session.file_name !='m483-2018-07-27-164242.txt' and behaviour_session.file_name != 'm480-2018-08-22-111012.txt':
                       
                                if subject_ephys == 'm480':
                                    m480.append(behaviour_session)
                                elif subject_ephys == 'm483':
                                    m483.append(behaviour_session)
                                elif subject_ephys == 'm479':
                                    m479.append(behaviour_session)
                                elif subject_ephys == 'm486':
                                    m486.append(behaviour_session)
                                elif subject_ephys == 'm478':
                                    m478.append(behaviour_session)
                                elif subject_ephys == 'm481':
                                    m481.append(behaviour_session)
                                elif subject_ephys == 'm484':
                                    m484.append(behaviour_session)
                           
                                    
    HP = m484 + m479 + m483
    PFC = m478 + m486 + m480 + m481
    all_sessions = m484  + m479 + m483 + m478 + m486 + m480 + m481
    return HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions

def target_times_f(all_experiments):
    # Trial times is array of reference point times for each trial. Shape: [n_trials, n_ref_points]
    # Here we are using [init-1000, init, choice, choice+1000]    
    # target_times is the reference times to warp all trials to. Shape: [n_ref_points]
    # Here we are finding the median timings for a whole experiment 
    trial_times_all_trials  = []
    for session in all_experiments:
        init_times = np.asarray([ev.time for ev in session.events if ev.name in ['choice_state', 'a_forced_state', 'b_forced_state']])
        inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['choice_state', 'a_forced_state', 'b_forced_state','sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']]
        choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             (i>0 and inits_and_choices[i-1].name == 'choice_state') or (i>0  and inits_and_choices[i-1].name == 'a_forced_state')\
                                 or (i>0  and inits_and_choices[i-1].name == 'b_forced_state')])
        if len(choice_times) != len(init_times):
            init_times  =(init_times[:len(choice_times)])
            
        trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T
        trial_times_all_trials.append(trial_times)

    trial_times_all_trials  =np.asarray(trial_times_all_trials)
    target_times = np.hstack(([0], np.cumsum(np.median(np.diff(trial_times_all_trials[0],1),0))))    
        
    return target_times


def all_sessions_aligment(experiment, all_experiments,  fs=25):
    target_times  = target_times_f(all_experiments)
    experiment_aligned = []
    for session in experiment:
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        init_times = np.asarray([ev.time for ev in session.events if ev.name in ['choice_state', 'a_forced_state', 'b_forced_state']])
        inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['choice_state', 'a_forced_state', 'b_forced_state','sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']]


        choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             (i>0 and inits_and_choices[i-1].name == 'choice_state') or (i>0  and inits_and_choices[i-1].name == 'a_forced_state')\
                                 or (i>0  and inits_and_choices[i-1].name == 'b_forced_state')])
      
        
        if len(choice_times) != len(init_times):
            init_times  = (init_times[:len(choice_times)])
            
        trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T
        aligned_rates, t_out, min_max_stretch = aa.align_activity(trial_times, target_times, spikes, fs = fs)    
        session.aligned_rates = aligned_rates
        session.t_out = t_out
        session.target_times = target_times
        experiment_aligned.append(session)
        
    return experiment_aligned 

def create_mat(experiment, title):
    
    all_sessions_list = []
    firing_rates = []
    for s,session in enumerate(experiment):
        
        index_non_forced = np.where(session.trial_data['forced_trial'] == 0)[0]
        index_forced = np.where(session.trial_data['forced_trial'] == 1)[0]

        firing_rate_non_forced = session.aligned_rates[index_non_forced]
        firing_rate_forced = session.aligned_rates[index_forced]
        
        choices = session.trial_data['choices']
        trials, neurons, time = firing_rate_non_forced.shape
        firing_rate = np.zeros((len(choices), neurons, time))
        
  
        index_non_forced = np.where(session.trial_data['forced_trial'] == 0)[0]
        index_forced = np.where(session.trial_data['forced_trial'] == 1)[0]

        
        task = session.trial_data['task']
        forced_trials = session.trial_data['forced_trial']
        block = session.trial_data['block']
        non_forced_array = np.where(forced_trials == 0)[0]  
        non_forced_choices = choices[non_forced_array]            
        
        # Getting out task indicies and choices
        task = session.trial_data['task']
        forced_trials = session.trial_data['forced_trial']
        non_forced_array = np.where(forced_trials == 0)[0]
        task_non_forced = task[non_forced_array]
        
        task_1 = np.where(task == 1)[0]
        task_2 = np.where(task == 2)[0] 
        task_3 = np.where(task == 3)[0]
        
        task_2_non_forced = np.where(task_non_forced == 2)[0]
        task_3_non_forced = np.where(task_non_forced == 3)[0]

        forced_trials = session.trial_data['forced_trial']
        outcomes = session.trial_data['outcomes']
    
        predictor_A_Task_1_forced, predictor_A_Task_2_forced, predictor_A_Task_3_forced,\
        predictor_B_Task_1_forced, predictor_B_Task_2_forced, predictor_B_Task_3_forced, reward_forced,\
        predictor_a_good_task_1_forced, predictor_a_good_task_2_forced, predictor_a_good_task_3_forced = predictors_forced(session)
        
        predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
        predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
        predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3,\
        reward_previous,previous_trial_task_1,previous_trial_task_2,previous_trial_task_3,\
        same_outcome_task_1, same_outcome_task_2, same_outcome_task_3,different_outcome_task_1, different_outcome_task_2, different_outcome_task_3, switch = predictors_include_previous_trial(session)     
           
        non_forced_choices = predictor_A_Task_1 + predictor_A_Task_2 + predictor_A_Task_3
        forced_choices = predictor_A_Task_1_forced + predictor_A_Task_2_forced + predictor_A_Task_3_forced
    
        choices_forced_unforced = np.zeros(len(choices))
        choices_forced_unforced[index_forced] = forced_choices[:len(index_forced)]
        choices_forced_unforced[index_non_forced] = non_forced_choices
    
        state = np.zeros(len(choices))
        forced_state = predictor_a_good_task_1_forced + predictor_a_good_task_2_forced + predictor_a_good_task_3_forced
        non_forced_state = np.zeros(len(non_forced_array))
        non_forced_state[predictor_a_good_task_1] = 1
        non_forced_state[predictor_a_good_task_2+task_2_non_forced[0]] = 1
        non_forced_state[predictor_a_good_task_3+task_3_non_forced[0]] = 1
        
        state[index_forced] = forced_state[:len(index_forced)]
        state[index_non_forced] = non_forced_state
        
        choices_forced_unforced[index_forced] = forced_choices[:len(index_forced)]
        choices_forced_unforced[index_non_forced] = non_forced_choices
       
        ones = np.ones(len(choices))

        firing_rate[index_forced] = firing_rate_forced[:len(index_forced)]
        firing_rate[index_non_forced] = firing_rate_non_forced[:len(index_non_forced)]

# =============================================================================
#           Extracting identity of pokes in each task                  
# =============================================================================

        poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
        poke_A_task_2 = 'poke_'+str(session.trial_data['poke_A'][task_2[0]])
        poke_A_task_3 = 'poke_'+str(session.trial_data['poke_A'][task_3[0]])
        poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
        poke_B_task_2  = 'poke_'+str(session.trial_data['poke_B'][task_2[0]])
        poke_B_task_3 = 'poke_'+str(session.trial_data['poke_B'][task_3[0]])
        configuration = session.trial_data['configuration_i']
    
        i_pokes = np.unique(configuration)
        i_poke_task_1 = configuration[0]
        i_poke_task_2 = configuration[task_2[0]]
        i_poke_task_3 = configuration[task_3[0]]
        
        poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3,poke_B1_B2_A3 = poke_A_B_make_consistent(session)
        
        if poke_A1_A2_A3 == True:
            constant_poke_a = poke_A
            poke_b_1 = poke_B
            poke_b_2 = poke_B_task_2
            poke_b_3 = poke_B_task_3
            
        if poke_A1_B2_B3 == True:
            constant_poke_a = poke_A
            poke_b_1 = poke_B
            poke_b_2 = poke_A_task_2
            poke_b_3 = poke_A_task_3
            
        if poke_A1_B2_A3 == True:
            constant_poke_a = poke_A
            poke_b_1 = poke_B
            poke_b_2 = poke_A_task_2
            poke_b_3 = poke_B_task_3  
            
        if poke_A1_A2_B3 == True:
            constant_poke_a = poke_A
            poke_b_1 = poke_B
            poke_b_2 = poke_B_task_2
            poke_b_3 = poke_A_task_3
            
        if poke_B1_B2_B3 == True:
            constant_poke_a = poke_B
            poke_b_1 = poke_A
            poke_b_2 = poke_A_task_2
            poke_b_3 = poke_A_task_3
            
        if poke_B1_A2_A3 == True:
            constant_poke_a = poke_B
            poke_b_1 = poke_A
            poke_b_2 = poke_B_task_2
            poke_b_3 = poke_B_task_3
            
        if poke_B1_A2_B3 == True:
            constant_poke_a = poke_B
            poke_b_1 = poke_A
            poke_b_2 = poke_B_task_2
            poke_b_3 = poke_A_task_3
            
        if poke_B1_B2_A3 == True:
            constant_poke_a = poke_B
            poke_b_1 = poke_A
            poke_b_2 = poke_A_task_2
            poke_b_3 = poke_B_task_3
        
        a_pokes = np.zeros(len(choices))
        a_pokes[:] = constant_poke_a[-1]
    
        b_pokes = np.zeros(len(choices))
        b_pokes[:task_1[-1]+1] = poke_b_1[-1]
        b_pokes[task_1[-1]+1:task_2[-1]+1] = poke_b_2[-1]
        b_pokes[task_2[-1]+1:] = poke_b_3[-1]
        
        i_pokes = np.zeros(len(choices))
        i_pokes[:task_1[-1]+1] = i_poke_task_1
        i_pokes[task_1[-1]+1:task_2[-1]+1] = i_poke_task_2
        i_pokes[task_2[-1]+1:] = i_poke_task_3
               
        predictors_all = OrderedDict([
                          ('latent_state',state),
                          ('choice',choices_forced_unforced ),
                          ('reward', outcomes),
                          ('forced_trials',forced_trials),
                          ('block', block),
                          ('task',task),
                          ('A', a_pokes),
                          ('B', b_pokes),
                          ('Initiation', i_pokes),
                          ('ones', ones)])
            
        X = np.vstack(predictors_all.values()).T[:len(choices),:].astype(float)
        
        # Save all sessions
        all_sessions_list.append(X)
        firing_rates.append(firing_rate)
   
    scipy.io.savemat('/Users/veronikasamborska/Desktop/'+ title + '.mat',{'Data': firing_rates, 'DM': all_sessions_list})
    data = {'Data': firing_rates, 'DM': all_sessions_list}
    
    return data



def extract_choice_pokes(session):
    task = session.trial_data['task']
    task_2_change = np.where(task ==2)[0]
    task_3_change = np.where(task ==3)[0]
    poke_I = 'poke_'+ str(session.trial_data['configuration_i'][0])
    poke_I_task_2 = 'poke_'+ str(session.trial_data['configuration_i'][task_2_change[0]])
    poke_I_task_3 = 'poke_'+ str(session.trial_data['configuration_i'][task_3_change[0]])  
    poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
    poke_A_task_2 = 'poke_'+str(session.trial_data['poke_A'][task_2_change[0]])
    poke_A_task_3 = 'poke_'+str(session.trial_data['poke_A'][task_3_change[0]])
    poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
    poke_B_task_2  = 'poke_'+str(session.trial_data['poke_B'][task_2_change[0]])
    poke_B_task_3 = 'poke_'+str(session.trial_data['poke_B'][task_3_change[0]])    
    
    return poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3

def poke_A_B_make_consistent(session):

    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3  = extract_choice_pokes(session)

    poke_A1_A2_A3 = False
    poke_A1_B2_B3 = False
    poke_A1_B2_A3 = False
    poke_A1_A2_B3 = False 
    poke_B1_B2_B3 = False
    poke_B1_A2_A3 = False
    poke_B1_A2_B3 = False
    poke_B1_B2_A3 = False
    
    if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
        poke_A1_A2_A3 = True 
    elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_A:
        poke_A1_B2_B3 = True
    elif poke_A == poke_A_task_3 and poke_B_task_2 == poke_A:
        poke_A1_B2_A3 = True
    elif poke_A == poke_A_task_2 and poke_A == poke_B_task_3:
        poke_A1_A2_B3 = True 
    elif poke_B == poke_B_task_2 and poke_B == poke_B_task_3:
        poke_B1_B2_B3 = True 
    elif poke_B == poke_A_task_2 and poke_B == poke_A_task_3:
        poke_B1_A2_A3 = True 
    elif poke_B == poke_A_task_2 and poke_B == poke_B_task_3:
        poke_B1_A2_B3 = True
    elif poke_B == poke_B_task_2 and poke_B == poke_A_task_3:
        poke_B1_B2_A3 = True
    return  poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3, poke_B1_B2_A3 

def predictors_pokes(session):

    choices = session.trial_data['choices']
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]   
    outcomes_all = session.trial_data['outcomes'] 
    reward = outcomes_all[non_forced_array]
    choice_non_forced = choices[non_forced_array]
    
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    poke_A = session.trial_data['poke_A']
    poke_B = session.trial_data['poke_B']
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3  = extract_choice_pokes(session)
    n_trials = len(choice_non_forced)

    #Task 1 
    choices_a = np.where(choice_non_forced == 1)
    choices_b = np.where(choice_non_forced == 0)
    
    predictor_a = np.zeros([1,n_trials])
    predictor_a[0][choices_a[0]] = 1
    predictor_b = np.zeros([1,n_trials])
    predictor_b[0][choices_b[0]] = 1
    if len(reward)!= len(predictor_a[0]):
        reward = np.append(reward,0)
     
    poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3,poke_B1_B2_A3 = poke_A_B_make_consistent(session)
    
    
    predictor_a_1 = copy.copy(predictor_a)
    predictor_a_1[0][len(task_1):] = 0
    predictor_a_2 =  copy.copy(predictor_a)
    predictor_a_2[0][:len(task_1)] = 0
    predictor_a_2[0][len(task_1)+len(task_2):] = 0 
    predictor_a_3 =  copy.copy(predictor_a)
    predictor_a_3[0][:len(task_1)+len(task_2)] = 0 
    
    predictor_b_1 =  copy.copy(predictor_b)
    predictor_b_1[0][len(task_1):] = 0
    predictor_b_2 = copy.copy(predictor_b)
    predictor_b_2[0][:len(task_1)] = 0
    predictor_b_2[0][len(task_1)+len(task_2):] = 0 
    predictor_b_3 = copy.copy(predictor_b)
    predictor_b_3[0][:len(task_1)+len(task_2)] = 0
    
    state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good = state_indices(session)

    if poke_A1_A2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(state_a_good)
        predictor_a_good_task_2 = copy.copy(state_t2_a_good)
        predictor_a_good_task_3 = copy.copy(state_t3_a_good)

    elif poke_A1_B2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(state_a_good)
        predictor_a_good_task_2 = copy.copy(state_t2_b_good)
        predictor_a_good_task_3 = copy.copy(state_t3_b_good)

    elif poke_A1_B2_A3 == True: 
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(state_a_good)
        predictor_a_good_task_2 = copy.copy(state_t2_b_good)
        predictor_a_good_task_3 = copy.copy(state_t3_a_good)

    elif poke_A1_A2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(state_a_good)
        predictor_a_good_task_2 = copy.copy(state_t2_a_good)
        predictor_a_good_task_3 = copy.copy(state_t3_b_good)

    elif poke_B1_B2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(state_b_good)
        predictor_a_good_task_2 = copy.copy(state_t2_b_good)
        predictor_a_good_task_3 = copy.copy(state_t3_b_good)

    elif poke_B1_A2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(state_b_good)
        predictor_a_good_task_2 = copy.copy(state_t2_a_good)
        predictor_a_good_task_3 = copy.copy(state_t3_a_good)
        
    elif poke_B1_A2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(state_b_good)
        predictor_a_good_task_2 = copy.copy(state_t2_a_good)
        predictor_a_good_task_3 = copy.copy(state_t3_b_good)

    elif poke_B1_B2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(state_b_good)
        predictor_a_good_task_2 = copy.copy(state_t2_b_good)
        predictor_a_good_task_3 = copy.copy(state_t3_a_good)
    
    return predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3
        
  
def extract_correct_forced_states(session):
    events = session.events
    forced_trials = []
    for event in events:
        if 'a_forced_state' in event:
            forced_trials.append(1)
        elif 'b_forced_state' in event:
            forced_trials.append(0)
    forced_trials = np.asarray(forced_trials)
    
    return forced_trials
                        

                             
def predictors_forced(session):
    forced_trials = session.trial_data['forced_trial']
    forced_array = np.where(forced_trials == 1)[0]
    
    task = session.trial_data['task']
    task_forced = task[forced_array]   
    outcomes_all = session.trial_data['outcomes'] 
    reward = outcomes_all[forced_array]
    choice_forced = extract_correct_forced_states(session)
    n_trials = len(choice_forced)
    task_1 = np.where(task_forced == 1)[0]
    task_2 = np.where(task_forced == 2)[0] 
    poke_A = session.trial_data['poke_A']
    poke_B = session.trial_data['poke_B']
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3  = extract_choice_pokes(session)
    
    #Task 1 
    choices_a = np.where(choice_forced == 1)
    choices_b = np.where(choice_forced == 0)
    
    predictor_a = np.zeros([1,n_trials])
    predictor_a[0][choices_a[0]] = 1
    predictor_b = np.zeros([1,n_trials])
    predictor_b[0][choices_b[0]] = 1
    if len(reward)!= len(predictor_a[0]):
        reward = np.append(reward,0)
        
    poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3,poke_B1_B2_A3 = poke_A_B_make_consistent(session)
    
    predictor_a_1 = copy.copy(predictor_a)
    predictor_a_1[0][len(task_1):] = 0
    predictor_a_2 =  copy.copy(predictor_a)
    predictor_a_2[0][:len(task_1)] = 0
    predictor_a_2[0][len(task_1)+len(task_2):] = 0 
    predictor_a_3 =  copy.copy(predictor_a)
    predictor_a_3[0][:len(task_1)+len(task_2)] = 0 
    
    predictor_b_1 =  copy.copy(predictor_b)
    predictor_b_1[0][len(task_1):] = 0
    predictor_b_2 = copy.copy(predictor_b)
    predictor_b_2[0][:len(task_1)] = 0
    predictor_b_2[0][len(task_1)+len(task_2):] = 0 
    predictor_b_3 = copy.copy(predictor_b)
    predictor_b_3[0][:len(task_1)+len(task_2)] = 0
    
    state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good = state_indices_forced(session)
    
    predictor_state_a = np.zeros([n_trials])
    predictor_state_b = np.zeros([n_trials])
    predictor_state_a_1_good = copy.copy(predictor_state_a)
    predictor_state_a_2_good = copy.copy(predictor_state_a)
    predictor_state_a_3_good = copy.copy(predictor_state_a)
    
    predictor_state_b_1_good = copy.copy(predictor_state_b)
    predictor_state_b_2_good = copy.copy(predictor_state_b)
    predictor_state_b_3_good = copy.copy(predictor_state_b)

    predictor_state_a_1_good[state_a_good] = 1
    predictor_state_b_1_good[state_b_good] = 1
    predictor_state_a_2_good[state_t2_a_good] = 1
    predictor_state_b_2_good[state_t2_b_good] = 1
    predictor_state_a_3_good[state_t3_a_good] = 1
    predictor_state_b_3_good[state_t3_b_good] = 1
    
    if poke_A1_A2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_a_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_a_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_a_3_good)

    elif poke_A1_B2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_a_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_b_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_b_3_good)

    elif poke_A1_B2_A3 == True: 
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_a_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_b_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_a_3_good)


    elif poke_A1_A2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_a_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_a_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_b_3_good)

    elif poke_B1_B2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_b_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_b_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_b_3_good)

    elif poke_B1_A2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_b_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_a_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_a_3_good)
        
    elif poke_B1_A2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_b_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_a_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_b_3_good)

    elif poke_B1_B2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_b_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_b_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_a_3_good)
    
    return predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3
    
    
    
def predictors_include_previous_trial(session): 
    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = predictors_pokes(session)    
    same_task_1 = []
    same_task_2 = []
    same_task_3 = []
    reward_previous = []
    previous_trial_task_1 = []
    previous_trial_task_2 = []
    previous_trial_task_3 = []
    
    task = session.trial_data['task']
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    
    task_non_forced = task[non_forced_array]
    task_1 = np.where(task_non_forced == 1)[0]
    task_1_len   = len(task_1) 
                
    task_2 = np.where(task_non_forced == 2)[0]        
    task_2_len  = len(task_2)
    
    predictor_A = predictor_A_Task_1+predictor_A_Task_2+predictor_A_Task_3
    switch = []
    for i,predictor in enumerate(predictor_A):
        if i > 0:
            if predictor_A[i-1] == 1 and predictor_A[i] == 1:
                switch.append(1)
            elif predictor_A[i-1] == 0 and predictor_A[i] == 0:
                switch.append(1)
            else:
                switch.append(0)
                
                
    for i,predictor in enumerate(predictor_A_Task_1):
        if i > 0:
            if predictor_A_Task_1[i-1] == 1 and predictor_A_Task_1[i] == 1:
                same_task_1.append(1)
            elif predictor_A_Task_1[i-1] == 0 and predictor_A_Task_1[i] == 0:
                same_task_1.append(1)
            else:
                same_task_1.append(0)
                
    for i,predictor in enumerate(predictor_A_Task_2):
        if i > 0:
            if predictor_A_Task_2[i-1] == 1 and predictor_A_Task_2[i] == 1 :
                same_task_2.append(1)
            elif predictor_A_Task_2[i-1] == 0 and predictor_A_Task_2[i] == 0:
                same_task_2.append(1)
            else:
                same_task_2.append(0)
                
    for i,predictor in enumerate(predictor_A_Task_3):
        if i > 0:
            if predictor_A_Task_3[i-1] == 1 and predictor_A_Task_3[i] == 1 :
                same_task_3.append(1)
            elif predictor_A_Task_3[i-1] == 0 and predictor_A_Task_3[i] == 0:
                same_task_3.append(1)
            else:
                same_task_3.append(0)
                
    for i,predictor in enumerate(reward):
        if i > 0:
            if reward[i-1] == 1 and reward[i] == 1 :
                reward_previous.append(1)
            else:
                reward_previous.append(0)      
                
    for i,predictor in enumerate(predictor_A_Task_1):
        if i > 0:
            if predictor_A_Task_1[i-1] == 1:
                trial = 1
            else:
                trial = 0
        else:
            trial = 0     
        previous_trial_task_1.append(trial)
        
    for i,predictor in enumerate(predictor_A_Task_2):
        if i > 0:
            if predictor_A_Task_2[i-1] == 1:
                trial = 1
            else:
                trial = 0
        else:
            trial = 0     
        previous_trial_task_2.append(trial)
        
    for i,predictor in enumerate(predictor_A_Task_3):
        if i > 0:
            if predictor_A_Task_3[i-1] == 1:
                trial = 1
            else:
                trial = 0
        else:
            trial = 0     
        previous_trial_task_3.append(trial)
                
    same_task_1 = np.asarray(same_task_1)
    same_task_2 = np.asarray(same_task_2)
    same_task_3 = np.asarray(same_task_3)
    
    reward_previous = np.asarray(reward_previous)    
    previous_trial_task_1 = np.asarray(previous_trial_task_1)
    previous_trial_task_2 = np.asarray(previous_trial_task_2)
    previous_trial_task_3 = np.asarray(previous_trial_task_3)
    
    same_outcome_task_1 = []
    for same, r in zip(same_task_1,reward_previous):
        if same ==1 and r ==1:
            same_outcome_task_1.append(0.5)
        elif same ==1 and r ==0:
            same_outcome_task_1.append(-0.5)
        else:
            same_outcome_task_1.append(0)
            
    same_outcome_task_2 = []
    for same, r in zip(same_task_2,reward_previous):
        if same ==1 and r ==1:
            same_outcome_task_2.append(0.5)
        elif same ==1 and r ==0:
            same_outcome_task_2.append(-0.5)
        else:
            same_outcome_task_2.append(0)
            
    same_outcome_task_3 = []
    for same, r in zip(same_task_3,reward_previous):
        if same ==1 and r ==1:
            same_outcome_task_3.append(0.5)
        elif same ==1 and r ==0:
            same_outcome_task_3.append(-0.5)
        else:
            same_outcome_task_3.append(0)

    
    same_outcome_task_1 = np.asarray(same_outcome_task_1)
    same_outcome_task_2 = np.asarray(same_outcome_task_2)
    same_outcome_task_3 = np.asarray(same_outcome_task_3)
   

    same_task_1 = same_task_1[:task_1_len-1]
    same_task_2 = same_task_2[task_1_len:task_1_len+task_2_len]
    same_task_3 = same_task_3[task_1_len+task_2_len:]
    
    reward_previous_task_1 = reward_previous[:task_1_len-1]
    reward_previous_task_2 = reward_previous[task_1_len:task_1_len+task_2_len]
    reward_previous_task_3 = reward_previous[task_1_len+task_2_len:]


    different_outcome_task_1 = []
    for same, r in zip(same_task_1,reward_previous_task_1):
        if same == 0 and r ==1:
            different_outcome_task_1.append(0.5)
        elif same == 0 and r ==0:
            different_outcome_task_1.append(-0.5)
        else:
            different_outcome_task_1.append(0)
            
    different_outcome_task_2 = []
    for same, r in zip(same_task_2,reward_previous_task_2):
        if same == 0 and r ==1:
            different_outcome_task_2.append(0.5)
        elif same == 0 and r ==0:
            different_outcome_task_2.append(-0.5)
        else:
            different_outcome_task_2.append(0)
            
    different_outcome_task_3 = []
    for same, r in zip(same_task_3,reward_previous_task_3):
        if same == 0 and r ==1:
            different_outcome_task_3.append(0.5)
        elif same == 0 and r ==0:
            different_outcome_task_3.append(-0.5)
        else:
            different_outcome_task_3.append(0)
    
    different_outcome_task_1 = np.asarray(different_outcome_task_1)
    different_outcome_task_2 = np.asarray(different_outcome_task_2)
    different_outcome_task_3 = np.asarray(different_outcome_task_3)
    
    return predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3,\
    reward_previous,previous_trial_task_1,previous_trial_task_2,previous_trial_task_3,\
    same_outcome_task_1, same_outcome_task_2, same_outcome_task_3,different_outcome_task_1, different_outcome_task_2, different_outcome_task_3, switch
        
def state_indices(session):
    trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3,ITI_task_1, ITI_task_2,ITI_task_3 = initiation_and_trial_end_timestamps(session)
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    state = session.trial_data['state']
    state_non_forced = state[non_forced_array]
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]
    
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    
    #Task 1 
    state_1 = state_non_forced[:len(task_1)]
    state_a_good = np.where(state_1 == 1)[0]
    state_b_good = np.where(state_1 == 0)[0]
    
    # Task 2
    state_2 = state_non_forced[len(task_1): (len(task_1) +len(task_2))]
    state_t2_a_good = np.where(state_2 == 1)[0]
    state_t2_b_good = np.where(state_2 == 0)[0]

    #Task 3 
    state_3 = state_non_forced[len(task_1) + len(task_2):]
    state_t3_a_good = np.where(state_3 == 1)[0]
    state_t3_b_good = np.where(state_3 == 0)[0]
    return state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good

# Extracts poke identities of poke A and B (1-9) for each task
def extract_choice_pokes(session):
    task = session.trial_data['task']
    task_2_change = np.where(task ==2)[0]
    task_3_change = np.where(task ==3)[0]
    poke_I = 'poke_'+ str(session.trial_data['configuration_i'][0])
    poke_I_task_2 = 'poke_'+ str(session.trial_data['configuration_i'][task_2_change[0]])
    poke_I_task_3 = 'poke_'+ str(session.trial_data['configuration_i'][task_3_change[0]])  
    poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
    poke_A_task_2 = 'poke_'+str(session.trial_data['poke_A'][task_2_change[0]])
    poke_A_task_3 = 'poke_'+str(session.trial_data['poke_A'][task_3_change[0]])
    poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
    poke_B_task_2  = 'poke_'+str(session.trial_data['poke_B'][task_2_change[0]])
    poke_B_task_3 = 'poke_'+str(session.trial_data['poke_B'][task_3_change[0]])    
    
    return poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3

# Extracts trial initiation timestamps and ITI timestamps
def extract_times_of_initiation_and_ITIs(session):
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3 = extract_choice_pokes(session)

    pyControl_choice = [event.time for event in session.events if event.name in ['choice_state']]
    pyControl_choice = np.array(pyControl_choice)
     
    #Poke A and Poke B Timestamps 
    pyControl_a_poke_entry = [event.time for event in session.events if event.name in [poke_A,poke_A_task_2,poke_A_task_3]]
    pyControl_b_poke_entry = [event.time for event in session.events if event.name in [poke_B,poke_B_task_2,poke_B_task_3 ]]

    #ITI Timestamps 
    pyControl_end_trial = [event.time for event in session.events if event.name in ['inter_trial_interval']][2:] #first two ITIs are free rewards
    pyControl_end_trial = np.array(pyControl_end_trial)
    
    return pyControl_choice, pyControl_a_poke_entry, pyControl_b_poke_entry, pyControl_end_trial

def initiation_and_trial_end_timestamps(session):
    pyControl_choice, pyControl_a_poke_entry, pyControl_b_poke_entry, pyControl_end_trial = extract_times_of_initiation_and_ITIs(session)
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]
    
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 

    #For Choice State Calculations
    trial_сhoice_state_task_1 = pyControl_choice[:len(task_1)]
    trial_сhoice_state_task_2 = pyControl_choice[len(task_1):(len(task_1) +len(task_2))]
    trial_сhoice_state_task_3 = pyControl_choice[len(task_1) + len(task_2):]


    task_1_end_trial = np.where(task == 1)[0]
    task_2_end_trial = np.where(task == 2)[0]
    pyControl_end_trial_1 = pyControl_end_trial[:len(task_1_end_trial)]
    pyControl_end_trial_2 =pyControl_end_trial[len(task_1_end_trial)+2:(len(task_1_end_trial)+len(task_2_end_trial)+2)]
    pyControl_end_trial_3 = pyControl_end_trial[len(task_1_end_trial)+len(task_2_end_trial)+4:]
    pyControl_end_trial =  np.concatenate([pyControl_end_trial_1, pyControl_end_trial_2,pyControl_end_trial_3])

    #For ITI Calculations
    ITI_non_forced = pyControl_end_trial[non_forced_array]  
    ITI_task_1 = ITI_non_forced[:len(task_1)]#[2:]
    ITI_task_2 = ITI_non_forced[(len(task_1)):(len(task_1)+len(task_2))]
    ITI_task_3 = ITI_non_forced[len(task_1) + len(task_2):]
    
    return trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3, ITI_task_1, ITI_task_2,ITI_task_3
  
def state_indices_forced(session):
    forced_trials = session.trial_data['forced_trial']
    forced_array = np.where(forced_trials == 1)[0]
    state = session.trial_data['state']
    state_forced = state[forced_array]
    task = session.trial_data['task']
    task_forced = task[forced_array]
    task_1 = np.where(task_forced == 1)[0]
    task_2 = np.where(task_forced == 2)[0] 
    
    
    #Task 1 
    state_1 = state_forced[:len(task_1)]
    state_a_good = np.where(state_1 == 1)[0]
    state_b_good = np.where(state_1 == 0)[0]
    
    # Task 2
    state_2 = state_forced[len(task_1): (len(task_1) +len(task_2))]
    state_t2_a_good = np.where(state_2 == 1)[0]
    state_t2_a_good = state_t2_a_good+len(task_1)
    state_t2_b_good = np.where(state_2 == 0)[0]
    state_t2_b_good = state_t2_b_good+len(task_1)

    #Task 3 
    state_3 = state_forced[len(task_1) + len(task_2):]
    state_t3_a_good = np.where(state_3 == 1)[0]
    state_t3_b_good = np.where(state_3 == 0)[0]
    state_t3_a_good =state_t3_a_good + (len(task_1) + len(task_2))
    state_t3_b_good = state_t3_b_good + (len(task_1) + len(task_2))
    
    return state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good
