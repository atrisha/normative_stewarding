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
                        if round(abs(action-utils.beta_mean(env.common_prior_appr)),1) <= env.normal_constr_w:
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

def run_simulation(institution, signal_type, attr_dict, show_plots=False):
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
    policy =  {s:get_value(x,action_and_state_space) for s,x in zip(action_and_state_space,fh.policy[:99,-1])}
    print(policy)
    if show_plots:
        plt.imshow(R, cmap='viridis', interpolation='bicubic')
        plt.colorbar()
        plt.title("Heatmap of R Matrix")
        plt.xticks(ticks=range(len(action_and_state_space)), labels=action_and_state_space)
        plt.yticks(ticks=range(len(action_and_state_space)), labels=action_and_state_space)
        plt.xlabel("Action Index")
        plt.ylabel("State Index")
        plt.show()

    return policy, np.mean(R)

def main():
    if len(sys.argv) != 3:
        print("Usage: python solving_tools.py <institution> <signal_type>")
        return
    institution = sys.argv[1]
    signal_type = sys.argv[2]
    run_simulation(institution, signal_type)

def test_run():
    run_simulation('intensive', 'ingroup')

if __name__ == "__main__":
    test_run()