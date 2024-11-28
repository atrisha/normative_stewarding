'''
Created on 10 Mar 2023

@author: Atrisha
'''

from perception_gap_information_design import parallel_env, Institution
import numpy as np
import mdptoolbox, mdptoolbox.example
import utils
import math
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import os
from utils import load_from_json
import json
import pickle
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator

def normalize_2d(R):
    total_sum = np.sum(R)
    return R / total_sum

# Function to calculate the expectation of x given a list of y values using RegularGridInterpolator
def calculate_expectation_from_points(ret_image, target_y_list, num_samples=1000):
    x_values = np.linspace(np.min(target_y_list),np.max(target_y_list),num_samples)
    # Normalize the 2D array to create a probability distribution
    R = ret_image.get_array()
    #R_normalized = normalize_2d(R)
    
    # Create an interpolator for the normalized data on the original grid
    interp_func = RegularGridInterpolator((target_y_list, target_y_list), R, method='slinear', bounds_error=False, fill_value=-1)
    #X, Y = np.meshgrid(np.linspace(0,0.5,100), np.linspace(0,0.5,100), indexing='ij')
    #fig = plt.figure()

    #ax = fig.add_subplot(projection='3d')
    #ax.plot_wireframe(X, Y, interp_func((X, Y)), rstride=3, cstride=3, alpha=0.4, color='m', label='cubic interp')
    #ax.set_xlabel('X Axis Label')
    #ax.set_ylabel('Y Axis Label')
    #plt.show()
    # Create a dictionary to store the results
    expected_x_dict = {}
    
    # Loop over the target_y values and calculate expected x for each
    for target_y in target_y_list:
        # Interpolate to get the probabilities for the exact target_y
        
        target_y_array = np.repeat(target_y, len(x_values))
        input_array = np.array([target_y_array, x_values]).reshape(2,len(x_values))
        probabilities = interp_func(input_array.T)  # Get interpolated probabilities
        probabilities = probabilities + np.abs(np.min(probabilities))
        # Ensure probabilities are not all zero (this could happen in some edge cases)
        probabilities = np.array(probabilities).flatten()
        if np.sum(probabilities) == 0:
            expected_x_dict[target_y] = None
            continue
        
        # Normalize probabilities for the selected y
        probabilities /= np.sum(probabilities)  # Normalize to sum to 1
        
        # Sample x values based on the probability distribution
        #sampled_x = np.random.choice(x_values, size=num_samples, p=probabilities)
        
        # Calculate the expectation (mean) of the sampled x values
        #expected_x = np.mean(sampled_x)
        expected_x = x_values[np.argmax(probabilities)]
        # Store the result in the dictionary with target_y as the key
        expected_x_dict[target_y] = expected_x
    
    return expected_x_dict

def get_value(index, state_space):
    # Determine the range based on the maximum value in the state_space
    if  max(state_space) < 0.6:
        # Reverse the operation: index/10 to get the value back
        value = index / 10.0
    elif max(state_space) >0.6 :
        # Find the value in the state_space for the given index and reverse the adjusted scaling
        # Assuming the state_space is sorted or has a predictable pattern for reverse calculation
        # This is tricky without knowing how values are distributed in state_space. Assuming uniform distribution:
        adjusted_value = index / (len(state_space) - 1)
        value = (adjusted_value / 2) + 0.5  # Reverse the adjust and scale for the second half
    else:
        raise ValueError("State space out of allowed range [0,1]", max(state_space))
    
    # No need to adjust for bounds as we're converting index to value
    return value


def get_index(value, state_space):
    # Adjust the value based on its range
    try:
        if 0 <= max(state_space) < 0.5:
            return int(value*10)  # Use the full scale for the first half
        elif 0.5 <= max(state_space) <= 1:
            adjusted_value = (value - 0.5) * 2  # Adjust and scale the second half
            index = state_space.index(value)
        else:
            raise ValueError("Value out of allowed range [0,1]", max(state_space))
    except ValueError:
        print('value and state space',value,state_space)
        raise ValueError("Value out of allowed range [0,1]", max(state_space))
    # Calculate the index and ensure it falls within the array size
    #index = int(np.floor(adjusted_value * (len(state_space) - 0.00001)))
    return min(index, len(state_space) - 1)  # Ensure index is within bounds

#intensive_outgroup_optimal = {0.0: 0.0, 0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0}

def generate_transition_matrix(action_and_state_space,institution,attr_dict):
    
    transition_map = dict()
    signal_cluster = 'appr' if max(action_and_state_space) > 0.6 else 'disappr'
    transition_matrix_len = len(action_and_state_space)
    reward_matrix = np.full((transition_matrix_len, transition_matrix_len), -1.0)
    for action in tqdm(action_and_state_space,desc='Action Progress'):
        #print('\n')
        if action not in transition_map:
            transition_map[action] = dict()
        actlist_t, actlist_r = [],[]
        repeats = 50
        if institution.type == 'intensive' and abs(action-0.1) < 0.01:
            f=1
        for run_iter in np.arange(repeats):
            institution.constant_disappr_signal = 0.4 if institution.type == 'extensive' else 0.3
            institution.constant_appr_signal = 0.5 if institution.type == 'extensive' else 0.7
            env = parallel_env(render_mode='human', attr_dict=attr_dict)
            env.ts=2
            env.signal_cluster = signal_cluster
            env.rhetoric_estimation_model = attr_dict['rhetoric_estimation_model']
            env.min_op_appr = np.min([ag.opinion[ag.norm_context] for ag in env.possible_agents if ag.opinion[ag.norm_context] >= 0.5])
            ''' Check that every norm context has at least one agent '''
            if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
                raise Exception()
            env.single_institution_env = True
            number_of_iterations = 50000
            env.NUM_ITERS = number_of_iterations
            for state in action_and_state_space: 
                if (state == 0.8 and action == 0.9):
                    f=1
                if  (state == 0 and action == 0.5):
                    f=1
                if (state-0.5)*(action-0.5) < 0:
                    continue
                #print(action,':',state,':',run_iter)
                env.reset()
                if state >= 0.5:
                    env.common_prior_appr = utils.est_beta_from_mu_sigma(state, utils.beta_var(attr_dict['common_prior_appr'][0], attr_dict['common_prior_appr'][1]))
                    env.common_prior_disappr = utils.est_beta_from_mu_sigma(0.4, utils.beta_var(attr_dict['common_prior_disappr'][0], attr_dict['common_prior_disappr'][1]))
                else:
                    env.common_prior_disappr = utils.est_beta_from_mu_sigma(state,  utils.beta_var(attr_dict['common_prior_disappr'][0], attr_dict['common_prior_disappr'][1]))
                    env.common_prior_appr = utils.est_beta_from_mu_sigma(0.6,  utils.beta_var(attr_dict['common_prior_appr'][0], attr_dict['common_prior_appr'][1]))
                
                #env.prior_baseline = (env.common_prior_appr + env.common_prior_disappr)/2
                for ag in env.possible_agents:
                    ag.init_beliefs(env)

                if institution.type == 'intensive':
                    if action >= 0.5:
                        if abs(action-utils.beta_mean(env.common_prior_appr)) <= env.normal_constr_w:
                            valid_distr = True
                        else:
                            valid_distr = False
                    else:
                        valid_distr = True
                else:
                    valid_distr = True
                    check_params = env.common_prior_appr if signal_cluster == 'appr' else env.common_prior_disappr
                    if round(abs(action-utils.beta_mean(check_params)),1) > env.normal_constr_w:
                            valid_distr = False
                    
                

                if valid_distr:
                    appr_pos_for_ts,disappr_pos_for_ts, prop_for_ts = None, None, None
                    for agent in env.possible_agents:
                        agent.sampled_institution = institution
                        if math.isnan(agent.common_prior_outgroup[0]/np.sum(agent.common_prior_outgroup)) or math.isnan(agent.common_prior_ingroup[0]/np.sum(agent.common_prior_ingroup)):
                            continue
                        institution.opt_signals = {'disappr': {round(x,1):(institution.constant_disappr_signal,action) if agent.opinion[agent.norm_context] >= 0.5 else (action,institution.constant_disappr_signal) for x in [round(x,1) for x in np.arange(0,0.5,.1)]},
                                        'appr': {round(x,1):(institution.constant_appr_signal,action) if agent.opinion[agent.norm_context] >= 0.5 else (action,institution.constant_appr_signal) for x in [round(x,1) for x in np.arange(0.5,1.1,.1)]}
                                        }   
                        if env.homogenous_priors and appr_pos_for_ts is not None and disappr_pos_for_ts is not None:
                            ingroup_posterior = appr_pos_for_ts if agent.opinion[agent.norm_context] >= 0.5 else disappr_pos_for_ts
                            outgroup_posterior = disappr_pos_for_ts if agent.opinion[agent.norm_context] >= 0.5 else appr_pos_for_ts
                            agent.common_proportion_prior = prop_for_ts
                        else:
                            if signal_cluster == 'appr':
                                if agent.opinion[agent.norm_context] >= 0.5:
                                    ingroup_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,institution.opt_signals,agent.common_proportion_prior,'ingroup')
                                    outgroup_posterior = agent.common_prior_outgroup
                                else:
                                    outgroup_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,institution.opt_signals,agent.common_proportion_prior,'outgroup')
                                    ingroup_posterior = agent.common_prior_ingroup
                            else:
                                if agent.opinion[agent.norm_context] >= 0.5:
                                    outgroup_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,institution.opt_signals,agent.common_proportion_prior,'outgroup')                         
                                    ingroup_posterior = agent.common_prior_ingroup
                                else:
                                    ingroup_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,institution.opt_signals,agent.common_proportion_prior,'ingroup')  
                                    outgroup_posterior = agent.common_prior_outgroup
                                    
                        
                        if env.homogenous_priors:
                            if appr_pos_for_ts is None:
                                appr_pos_for_ts = ingroup_posterior if agent.opinion[agent.norm_context] >= 0.5 else outgroup_posterior
                                prop_for_ts = agent.common_proportion_posterior
                            if disappr_pos_for_ts is None:
                                disappr_pos_for_ts = ingroup_posterior if agent.opinion[agent.norm_context] < 0.5 else outgroup_posterior

                        agent.pseudo_update_posteriors = {institution.type:{'outgroup':outgroup_posterior,'ingroup':ingroup_posterior}}
                    
                    actions = {agent.id:agent.simple_act(env,run_type={'institutions':institution,'update_type':'common'},baseline=False) for agent in env.possible_agents}
                    _poi = [ag.common_posterior_ingroup for ag in env.possible_agents][0]
                    _poo = [ag.common_posterior_outgroup for ag in env.possible_agents][0]
                    _f = np.mean([ag.opinion[ag.norm_context] for ag in env.possible_agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] >= 0.5])
                    _p = {c:[ag.action_code for ag in env.possible_agents].count(c) for c in [-1,0,1]}    
                    '''
                    plt.figure()
                    plt.hist([ag.opinion[ag.norm_context] for ag in env.possible_agents if ag.action[0]!=-1])
                    plt.show()
                    '''
                    ''' common prior is updated based on the action observations '''
                    if env.print_log:
                        print(f'Running with signal cluster {signal_cluster} and action {action} and state {state}')
                    observations, reward, terminations, truncations, infos = env.step(actions,run_iter,'transition_generation')
                    
                    f=1
                else:
                    observations = {'appr':utils.beta_mean(env.common_prior_appr),'disappr':utils.beta_mean(env.common_prior_disappr)}
                    observations, reward, terminations, truncations, infos = observations, -1, {agent.id:False for agent in env.possible_agents}, {agent.id:False for agent in env.possible_agents}, {agent.id:{} for agent in env.possible_agents}
                #print(round(action,1),round(state,1),round(observations[0]/sum(observations),1))
                #print(round(action,1),round(state,1),round(reward,1))
                _grp_key = 'appr' if max(action_and_state_space) > 0.6 else 'disappr'
                next_state = np.clip(round(observations[_grp_key],1), min(action_and_state_space), max(action_and_state_space))
                if next_state is np.NaN or (next_state > 0.5 and max(action_and_state_space)<=0.5):
                    print(observations)
                try:
                    if action not in action_and_state_space or state not in action_and_state_space or next_state not in action_and_state_space:
                        f=1
                    a_idx, s_idx, s_prime_idx = get_index(action,action_and_state_space), get_index(state,action_and_state_space), get_index(next_state,action_and_state_space)
                except ValueError:
                    print(next_state,observations)
                    f=1
                    raise
                if s_idx == 4:
                    f=1
                if (s_idx,s_prime_idx) not in transition_map[action]:
                    transition_map[action][(s_idx,s_prime_idx)] = 1
                else:
                    transition_map[action][(s_idx,s_prime_idx)] += 1
                reward_matrix[s_idx,a_idx] += reward
                #print('----')
    reward_matrix = reward_matrix/repeats    
    '''
    transition_matrix = np.zeros(shape=(transition_matrix_len,transition_matrix_len,transition_matrix_len))
    for act,s_s_data in transition_map.items():
        a_idx = int(round(act,1)*10)-1
        for s_s_prime,ct in s_s_data.items():
            transition_matrix[a_idx,s_s_prime[0],s_s_prime[1]] = ct
    # Normalize each row to ensure the sum of each row is equal to 1
    row_sums = transition_matrix.sum(axis=2)
    transition_matrix = transition_matrix / row_sums[:, :, np.newaxis]
    '''
    transition_matrix = np.zeros(shape=(transition_matrix_len, transition_matrix_len, transition_matrix_len))
    
    for act, s_s_data in transition_map.items():
        a_idx = get_index(act,action_and_state_space)
        for s_s_prime, ct in s_s_data.items():
            transition_matrix[a_idx, s_s_prime[0], s_s_prime[1]] = ct
    
    # Normalize each row
    row_sums = transition_matrix.sum(axis=2)
    zero_rows = row_sums == 0
    transition_matrix[~zero_rows] = transition_matrix[~zero_rows] / row_sums[:, :, np.newaxis][~zero_rows]
    
    # Handle rows that sum to zero (if any)
    # One approach is to distribute the probabilities evenly across such rows
    #for idx in np.where(zero_rows):
     #   transition_matrix[idx[0], idx[1], :] = 1.0 / transition_matrix_len
    
    # Check the sums after normalization
    assert np.allclose(transition_matrix.sum(axis=2), 1), "Rows do not sum to 1"
    
    # Continue with your code...
    # reward_matrix = ...

    return transition_matrix, reward_matrix

def generate_reward(institution,attr_dict,state,action):
    signal_cluster = 'appr' # for now, we assume the signal cluster is 'appr'
    env = parallel_env(render_mode='human', attr_dict=attr_dict)
    env.signal_cluster = signal_cluster
    env.rhetoric_estimation_model = attr_dict['rhetoric_estimation_model']
    env.min_op_appr = np.min([ag.opinion[ag.norm_context] for ag in env.possible_agents if ag.opinion[ag.norm_context] >= 0.5])
    number_of_iterations = 50000
    env.NUM_ITERS = number_of_iterations
    ''' Check that every norm context has at least one agent '''
    if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
        raise Exception()
    env.single_institution_env = True
    env.reset()
    if state >= 0.5:
        env.common_prior_appr = utils.est_beta_from_mu_sigma(state, utils.beta_var(attr_dict['common_prior_appr'][0], attr_dict['common_prior_appr'][1]))
        env.common_prior_disappr = utils.est_beta_from_mu_sigma(0.4, utils.beta_var(attr_dict['common_prior_disappr'][0], attr_dict['common_prior_disappr'][1]))
    else:
        env.common_prior_disappr = utils.est_beta_from_mu_sigma(state,  utils.beta_var(attr_dict['common_prior_disappr'][0], attr_dict['common_prior_disappr'][1]))
        env.common_prior_appr = utils.est_beta_from_mu_sigma(0.6,  utils.beta_var(attr_dict['common_prior_appr'][0], attr_dict['common_prior_appr'][1]))
    
    #env.prior_baseline = (env.common_prior_appr + env.common_prior_disappr)/2
    for ag in env.possible_agents:
        ag.init_beliefs(env)
    institution.constant_disappr_signal = 0.4 if institution.type == 'extensive' else 0.3
    institution.constant_appr_signal = 0.5 if institution.type == 'extensive' else 0.7
    if institution.type == 'intensive':
        if action >= 0.5:
            if abs(action-utils.beta_mean(env.common_prior_appr)) <= env.normal_constr_w:
                valid_distr = True
            else:
                valid_distr = False
        else:
            valid_distr = True
    else:
        valid_distr = True
        check_params = env.common_prior_appr if signal_cluster == 'appr' else env.common_prior_disappr
        if abs(action-utils.beta_mean(check_params)) > env.normal_constr_w:
                valid_distr = False
        
    

    if valid_distr:
        appr_pos_for_ts,disappr_pos_for_ts, prop_for_ts = None, None, None
        for agent in env.possible_agents:
            agent.sampled_institution = institution
            if math.isnan(agent.common_prior_outgroup[0]/np.sum(agent.common_prior_outgroup)) or math.isnan(agent.common_prior_ingroup[0]/np.sum(agent.common_prior_ingroup)):
                continue
            institution.opt_signals = {'disappr': {round(x,1):(institution.constant_disappr_signal,action) if agent.opinion[agent.norm_context] >= 0.5 else (action,institution.constant_disappr_signal) for x in [round(x,1) for x in np.arange(0,0.5,.1)]},
                            'appr': {round(x,1):(institution.constant_appr_signal,action) if agent.opinion[agent.norm_context] >= 0.5 else (action,institution.constant_appr_signal) for x in [round(x,1) for x in np.arange(0.5,1.1,.1)]}
                            }   
            if env.homogenous_priors and appr_pos_for_ts is not None and disappr_pos_for_ts is not None:
                ingroup_posterior = appr_pos_for_ts if agent.opinion[agent.norm_context] >= 0.5 else disappr_pos_for_ts
                outgroup_posterior = disappr_pos_for_ts if agent.opinion[agent.norm_context] >= 0.5 else appr_pos_for_ts
                agent.common_proportion_prior = prop_for_ts
            else:
                if signal_cluster == 'appr':
                    if agent.opinion[agent.norm_context] >= 0.5:
                        ingroup_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,institution.opt_signals,agent.common_proportion_prior,'ingroup')
                        outgroup_posterior = agent.common_prior_outgroup
                    else:
                        outgroup_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,institution.opt_signals,agent.common_proportion_prior,'outgroup')
                        ingroup_posterior = agent.common_prior_ingroup
                else:
                    if agent.opinion[agent.norm_context] >= 0.5:
                        outgroup_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,institution.opt_signals,agent.common_proportion_prior,'outgroup')                         
                        ingroup_posterior = agent.common_prior_ingroup
                    else:
                        ingroup_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,institution.opt_signals,agent.common_proportion_prior,'ingroup')  
                        outgroup_posterior = agent.common_prior_outgroup
                        
            
            if env.homogenous_priors:
                if appr_pos_for_ts is None:
                    appr_pos_for_ts = ingroup_posterior if agent.opinion[agent.norm_context] >= 0.5 else outgroup_posterior
                    prop_for_ts = agent.common_proportion_posterior
                if disappr_pos_for_ts is None:
                    disappr_pos_for_ts = ingroup_posterior if agent.opinion[agent.norm_context] < 0.5 else outgroup_posterior

            agent.pseudo_update_posteriors = {institution.type:{'outgroup':outgroup_posterior,'ingroup':ingroup_posterior}}
        
        actions = {agent.id:agent.simple_act(env,run_type={'institutions':institution,'update_type':'common'},baseline=False) for agent in env.possible_agents}
        _poi = [ag.common_posterior_ingroup for ag in env.possible_agents][0]
        _poo = [ag.common_posterior_outgroup for ag in env.possible_agents][0]
        _f = np.mean([ag.opinion[ag.norm_context] for ag in env.possible_agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] >= 0.5])
        _p = {c:[ag.action_code for ag in env.possible_agents].count(c) for c in [-1,0,1]}    
        '''
        plt.figure()
        plt.hist([ag.opinion[ag.norm_context] for ag in env.possible_agents if ag.action[0]!=-1])
        plt.show()
        '''
        ''' common prior is updated based on the action observations '''
        if env.print_log:
            print(f'Running with signal cluster {signal_cluster} and action {action} and state {state}')
        observations, reward, terminations, truncations, infos = env.step(actions,None,'transition_generation')
        
        f=1
    else:
        observations = {'appr':utils.beta_mean(env.common_prior_appr),'disappr':utils.beta_mean(env.common_prior_disappr)}
        observations, reward, terminations, truncations, infos = observations, -1, {agent.id:False for agent in env.possible_agents}, {agent.id:False for agent in env.possible_agents}, {agent.id:{} for agent in env.possible_agents}
    #print(round(action,1),round(state,1),round(observations[0]/sum(observations),1))
    #print(round(action,1),round(state,1),round(reward,1))
    _grp_key = 'appr' #for now, we assume the signal cluster is 'appr'
    next_state = observations[_grp_key]
    return next_state, reward


def run_simulation(institution, signal_type, attr_dict={}, show_plots=False):
    if institution == "extensive":
        institution = Institution('extensive')
    elif institution == "intensive":
        institution = Institution('intensive')
    if signal_type == "outgroup":
        action_and_state_space = [round(x, 1) for x in np.linspace(0, 0.5, 6)]
    elif signal_type == "ingroup":
        action_and_state_space = [round(x, 1) for x in np.linspace(0.5, 1, 6)]
    P, R = generate_transition_matrix(action_and_state_space,institution,attr_dict)

    fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.5, 1000)
    #fh = mdptoolbox.mdp.QLearning(P, R, 0.9)
    fh.run()
    #print([np.round((x+1)/10,1) for x in list(fh.policy)])3.
    print(fh.policy[:99,-1])
    
    ret_image = plt.imshow(R, cmap='viridis', interpolation='bicubic')
    if show_plots:
        plt.colorbar()
        plt.title(institution.type + ': ' + signal_type)
        plt.xticks(ticks=range(len(action_and_state_space)), labels=action_and_state_space)
        plt.yticks(ticks=range(len(action_and_state_space)), labels=action_and_state_space)
        plt.xlabel("Action Index")
        plt.ylabel("State Index")
        plt.show()
    
    expected_policy_dict = calculate_expectation_from_points(ret_image, action_and_state_space, num_samples=1000)
    
    policy =  {s:get_value(x,action_and_state_space) for s,x in zip(action_and_state_space,fh.policy[:99,-1])}
    print(expected_policy_dict)
    return expected_policy_dict, np.mean(R)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
from gym import spaces

# Define custom environment with state and action space between [0.5, 1]
class SimpleEnv(gym.Env):
    def __init__(self, attr_dict):
        super(SimpleEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([0.5]), high=np.array([1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0.5]), high=np.array([1.0]), dtype=np.float32)
        self.state = np.random.uniform(0.5, 1.0)
        self.institution = Institution('intensive')
        self.attr_dict = attr_dict

    def reset(self):
        self.state = np.random.uniform(0.5, 1.0)
        return np.array([self.state])

    def step(self, state, action):
        next_state, reward = generate_reward(self.institution, self.attr_dict, state, action)
        self.state = next_state
        done = False
        return np.array([self.state]), reward, done, {}

# Define the neural network for the policy (actor) and value function (critic)
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc_policy = nn.Linear(64, 1)
        self.fc_value = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        action_mean = torch.sigmoid(self.fc_policy(x)) * 0.5 + 0.5  # Scale output to [0.5, 1]
        value = self.fc_value(x)
        return action_mean, value

# VPG Agent
class VPGAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.policy = ActorCritic()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mean, _ = self.policy(state)
        dist = Normal(action_mean, torch.tensor([0.1]))  # Set small variance for exploration
        action = dist.sample()
        action = torch.clamp(action, 0.5, 1.0)  # Ensure action is within [0.5, 1.0]
        return action.detach().numpy(), dist.log_prob(action)

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        states = torch.FloatTensor(memory.states)
        actions = torch.FloatTensor(memory.actions)
        logprobs = torch.FloatTensor(memory.logprobs)

        # Get value estimates from critic
        _, values = self.policy(states)

        # Calculate advantages (rewards - baseline from value function)
        advantages = rewards - values.detach()

        # Compute the policy gradient loss (negative log-probability * advantage)
        policy_loss = -(logprobs * advantages).mean()

        # Optionally include value loss
        value_loss = self.MseLoss(values, rewards)

        # Total loss: policy loss + value loss
        loss = policy_loss + value_loss

        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# PPO Agent
class PPOAgent:
    def __init__(self, env, use_gpu=False, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic().to(self.device)  # Move model to device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Move state to device
        action_mean, _ = self.policy(state)
        dist = Normal(action_mean, torch.tensor([0.1], device=self.device))  # Set tensor on correct device
        action = dist.sample()
        action = torch.clamp(action, 0.5, 1.0)
        return action.detach().cpu().numpy(), dist.log_prob(action)

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)  # Move rewards to device
        states = torch.FloatTensor(memory.states).to(self.device)  # Move states to device
        actions = torch.FloatTensor(memory.actions).to(self.device)  # Move actions to device
        old_logprobs = torch.FloatTensor(memory.logprobs).to(self.device)  # Move logprobs to device
        total_loss = 0
        
        for _ in range(self.k_epochs):
            action_means, values = self.policy(states)
            dist = Normal(action_means, torch.tensor([0.1], device=self.device))  # Move tensor to device
            logprobs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Calculate the ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantage estimation
            advantages = rewards - values.detach()

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards) - 0.01 * entropy

            # Take a gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            total_loss += loss.mean().item()
        
        return total_loss

# Memory to store trajectories
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def add(self, state, action, logprob, reward):
        self.actions.append(action)
        self.states.append(state)
        self.logprobs.append(logprob)
        self.rewards.append(reward)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

def setup_attrs(inst_type):
    common_prior_appr_input = (5,3)
    common_prior_appr = (5,3)
    common_prior_disappr = (3,5)
    common_proportion_prior = (5,5)
    inst_opt_policy, inst_sampling_ratios = {}, {}
    data_directory = os.path.join(os.getcwd(), 'data')
    filename = os.path.join(data_directory, 'inst_opt_policy.json')
    filename_sampling_ratios = os.path.join(data_directory, 'inst_sampling_ratios.json')
    if os.path.isfile(filename):
        inst_opt_policy = load_from_json(filename)
    if os.path.isfile(filename_sampling_ratios):
        inst_sampling_ratios = load_from_json(filename_sampling_ratios)
    attr_dict = {'distr_params':{'mean_op_degree_apr':0.6,'mean_op_degree_disapr':0.4,'apr_weight':0.5,'SD':0.2},
                                                                    'distr_shape':'U',
                                                                      'extensive': False,
                                                                      'common_prior_appr': common_prior_appr,
                                                                      'common_prior_disappr': common_prior_disappr,
                                                                      'common_proportion_prior': common_proportion_prior,
                                                                      'common_prior_appr_input': common_prior_appr_input,
                                                                      'only_intensive': False,
                                                                      'homogenous_priors': True,
                                                                      'num_players':100,
                                                                      'alpha':0.5,
                                                                      'tailored_alpha':False,
                                                                      'lambda_outgroup':0.5,
                                                                      'lambda_ingroup':1.5,
                                                                      'normal_constr_w':0.2,
                                                                      'rhet_thresh_mean':0.5,
                                                                      'update_rate':0.2,
                                                                      'inst_update_rate':0.2,
                                                                      'num_batches':10,
                                                                      'num_timesteps':100,
                                                                      'print_log':False,
                                                                      'verbose':False,}
    file_path = os.path.join(os.getcwd(),'pickles','rhet_eq_estimation.pkl')
    run_param ={'common_prior_appr_input':common_prior_appr_input,
                        'common_prior_appr':common_prior_appr,
                        'common_prior_disappr':common_prior_disappr,
                        'common_proportion_prior':common_proportion_prior,
                        'normal_constr_w':attr_dict['normal_constr_w'],
                        'credible':True}
    run_param['attr_dict'] = attr_dict
    if os.path.exists(file_path):
        attr_dict['rhetoric_estimation_model'] = pickle.load(open(file_path, "rb"))
    else:
        print('Generating Rhetoric Equilibrium Estimation Model')
        model = utils.generate_rhetoric_equilibrium_estimation_model(run_param)
        attr_dict['rhetoric_estimation_model'] = model
        pickle.dump(model, open(file_path, "wb"))
    attr_dict['extensive'] = True if inst_type=='extensive' else False
    return attr_dict
    
# Main training loop
def main_rl():
    attr_dict = {'distr_params': {'mean_op_degree_apr': 0.6, 'mean_op_degree_disapr': 0.4, 'apr_weight': 0.5, 'SD': 0.2},
                 'distr_shape': 'U',
                 'extensive': False,
                 'common_prior_appr': (5, 3),
                 'common_prior_disappr': (3, 5),
                 'common_proportion_prior': (5, 5),
                 'only_intensive': False,
                 'homogenous_priors': True,
                 'num_players': 100,
                 'alpha': 0.5,
                 'lambda_outgroup': 0.5,
                 'lambda_ingroup': 1.5,
                 'normal_constr_w': 0.2,
                 'rhet_thresh_mean': 0.5,
                 'update_rate': 0.2,
                 'inst_update_rate': 0.2,
                 'num_batches': 10,
                 'num_timesteps': 100,
                 'print_log': False,
                 'verbose': False}
    common_prior_appr_input = (5,3)
    common_prior_appr = (5,3)
    common_prior_disappr = (3,5)
    common_proportion_prior = (5,5)
    inst_opt_policy, inst_sampling_ratios = {}, {}
    data_directory = os.path.join(os.getcwd(), 'data')
    filename = os.path.join(data_directory, 'inst_opt_policy.json')
    filename_sampling_ratios = os.path.join(data_directory, 'inst_sampling_ratios.json')
    if os.path.isfile(filename):
        inst_opt_policy = load_from_json(filename)
    if os.path.isfile(filename_sampling_ratios):
        inst_sampling_ratios = load_from_json(filename_sampling_ratios)
    attr_dict = {'distr_params':{'mean_op_degree_apr':0.6,'mean_op_degree_disapr':0.4,'apr_weight':0.5,'SD':0.2},
                                                                    'distr_shape':'U',
                                                                      'extensive': False,
                                                                      'common_prior_appr': common_prior_appr,
                                                                      'common_prior_disappr': common_prior_disappr,
                                                                      'common_proportion_prior': common_proportion_prior,
                                                                      'common_prior_appr_input': common_prior_appr_input,
                                                                      'only_intensive': False,
                                                                      'homogenous_priors': True,
                                                                      'num_players':100,
                                                                      'alpha':0.5,
                                                                      'tailored_alpha':False,
                                                                      'lambda_outgroup':0.5,
                                                                      'lambda_ingroup':1.5,
                                                                      'normal_constr_w':0.2,
                                                                      'rhet_thresh_mean':0.5,
                                                                      'update_rate':0.2,
                                                                      'inst_update_rate':0.2,
                                                                      'num_batches':10,
                                                                      'num_timesteps':100,
                                                                      'print_log':False,
                                                                      'verbose':False,}
    run_param ={'common_prior_appr_input':common_prior_appr_input,
                        'common_prior_appr':common_prior_appr,
                        'common_prior_disappr':common_prior_disappr,
                        'common_proportion_prior':common_proportion_prior,
                        'normal_constr_w':attr_dict['normal_constr_w'],
                        'credible':True}
    run_param['attr_dict'] = attr_dict
    ''' This is a regression model constructed to estimate the equilibrium rhetoric. This is done to make the simulation more efficient.'''
    file_path = os.path.join(os.getcwd(),'pickles','rhet_eq_estimation.pkl')
    if os.path.exists(file_path):
        run_param['rhetoric_estimation_model'] = pickle.load(open(file_path, "rb"))
    else:
        print('Generating Rhetoric Equilibrium Estimation Model')
        model = utils.generate_rhetoric_equilibrium_estimation_model(run_param)
        run_param['rhetoric_estimation_model'] = model
        pickle.dump(model, open(file_path, "wb"))
    inst_type = 'intensive'
    attr_dict['extensive'] = True if inst_type=='extensive' else False
    attr_dict['rhetoric_estimation_model'] = run_param['rhetoric_estimation_model']
    # Set use_gpu flag
    use_gpu = True  # Set to True to use GPU, False to use CPU

    env = SimpleEnv(attr_dict)
    #agent = PPOAgent(env, use_gpu=use_gpu)
    agent = VPGAgent(env)
    memory = Memory()
    
    num_episodes = 500
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for t in range(100):
            action, logprob = agent.select_action(state)
            next_state, reward, done, _ = env.step(state, action)
            memory.add(state, action, logprob, reward)
            state = next_state
            total_reward += reward

            if done:
                break

        total_loss = agent.update(memory)
        memory.clear()

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Total Loss: {total_loss}")



def main():
    if len(sys.argv) == 3:
        print("Usage: python solving_tools.py <institution> <signal_type>")
        return
        institution = sys.argv[1]
        signal_type = sys.argv[2]
    else:
        institution = "intensive"
        signal_type = "ingroup"
    run_simulation(institution, signal_type)

def test_run():
    inst_type = 'intensive'
    attr_dict = setup_attrs(inst_type)
    run_simulation(inst_type, 'outgroup',attr_dict, show_plots=True)

if __name__ == "__main__":
    test_run()