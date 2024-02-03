'''
Created on 11 Jan 2024

@author: Atrisha
'''
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

alpha = 0.1

op_in, op_out = 0.6, 0.4
op_hat_in_init, op_hat_out_init = 0.7, 0.3
op_to_rhet_intensity = lambda o,op_hat_in,op_hat_out: (-1.6425*o**2 + 3.6693*o + -1.3048 if o >= 0.5 else -1.6425*(1-o)**2 + 3.6693*(1-o) + -1.3048) if o > (alpha+0.4)/ (op_hat_in + op_hat_out) else 0
rhet_intensity_out = op_to_rhet_intensity(op_out,1-op_hat_in_init, 1-op_hat_out_init)
inst_signals_extensive = {0:0.1,0.1:0.1, 0.2:0.5, 0.3:0.5, 0.4:0.5, 0.5:0.5, 0.6:0.5, 0.7:0.5, 0.8:0.5, 0.9:0.6,1:0.7}
inst_signals_intensive = {0:0.1,0.1:0.1, 0.2:0.1, 0.3:0.1, 0.4:0.1, 0.5:0.2, 0.6:0.3, 0.7:0.5, 0.8:0.6, 0.9:0.6,1:0.7}
rhet_thresh = 0

class Agent:
    def __init__(self, opinion, op_in_belief, op_out_belief, rhet_intensity_out, rhet_intensity):
        self.opinion = opinion
        self.op_in_belief = op_in_belief
        self.op_out_belief = op_out_belief
        self.rhet_intensity_out = rhet_intensity_out
        self.rhet_intensity = rhet_intensity
        self.subscribed_to = []
        self.subscribed_proportions = {'extensive':0.5,'intensive':0.5}

    def update_from_inst_signal(self, signal, signal_type):
        if signal_type == 'in_group':
            return np.mean([self.op_in_belief,signal])
        else:
            return np.mean([self.op_out_belief,signal])
        

    def act(self,op_in_bel,op_out_bel):
        opt_rhet = op_to_rhet_intensity(self.opinion,op_in_bel,op_out_bel)
        return 'E' if opt_rhet > rhet_thresh else 'S' 
            

    def update_from_interaction_signal(self, post_interaction_signal):
        # Placeholder for implementation
        pass

class Institution:
    def __init__(self, type):
        self.type=type
        self.subscriber_list = []
        self.signals = {'in_group':[],'out_group':[]}
        self.institution_community_opinion = None
        self.institution_community_ingroup_belief = None
        self.institution_community_outgroup_belief = None
        
    def generate_signal(self, op_state):
        if self.type == 'intensive':
            return inst_signals_intensive[round(op_state,1)]
        else:
            return inst_signals_extensive[round(op_state,1)]

# Function to sample 100 agents
def sample_agents():
    agents = []
    for _ in range(50):  # First half of the distribution
        opinion = np.random.normal(op_out, 0.1)
        agents.append(Agent(opinion, op_hat_in_init, op_hat_out_init, None, None))

    for op in np.linspace(0.5,1,50):  # Second half of the distribution
        opinion = op
        agents.append(Agent(opinion, op_hat_in_init, op_hat_out_init, None, None))

    return agents

# Sample 100 agents
sampled_agents = sample_agents()
sampled_agents = [a for a in sampled_agents if a.opinion >= 0.5]
sampled_agents.sort(key=lambda x: x.opinion)
# Displaying the opinions of the first 10 sampled agents for verification
extensive_institution, intensive_institution = None, None
for iter_ts in np.arange(10):
    if iter_ts == 0:
        extensive_institution = Institution('extensive')
        intensive_institution = Institution('intensive')

    ''' Institutions generate signals and keep a track of all the signals they have generated'''

    if len(extensive_institution.subscriber_list) == 0:
        extensive_institution_signal_ingroup = extensive_institution.generate_signal(op_hat_in_init)
        extensive_institution_signal_outgroup = extensive_institution.generate_signal(op_hat_out_init)
    else:
        extensive_institution_signal_ingroup = extensive_institution.generate_signal(extensive_institution.institution_community_opinion)
        extensive_institution_signal_outgroup = extensive_institution.generate_signal(extensive_institution.institution_community_outgroup_belief)
    
    if len(intensive_institution.subscriber_list) == 0:
        intensive_institution_signal_ingroup = intensive_institution.generate_signal(op_hat_in_init)
        intensive_institution_signal_outgroup = intensive_institution.generate_signal(op_hat_out_init)
    else:
        intensive_institution_signal_ingroup = intensive_institution.generate_signal(intensive_institution.institution_community_opinion)
        intensive_institution_signal_outgroup = intensive_institution.generate_signal(intensive_institution.institution_community_outgroup_belief)
        
    extensive_institution.signals['in_group'].append(extensive_institution_signal_ingroup)
    intensive_institution.signals['in_group'].append(intensive_institution_signal_ingroup)
    extensive_institution.signals['out_group'].append(extensive_institution_signal_outgroup)
    intensive_institution.signals['out_group'].append(intensive_institution_signal_outgroup)
    
    ''' Agents pseudo-update their beliefs based on the signals they receive from the institutions'''
    for ag in sampled_agents:
        op_rhet_intensive_marg = ag.act( np.mean([ag.op_in_belief,intensive_institution_signal_ingroup]), np.mean([ag.op_out_belief,intensive_institution_signal_outgroup]))
        op_rhet_extensive_marg = ag.act( np.mean([ag.op_in_belief,extensive_institution_signal_ingroup]), np.mean([ag.op_out_belief,extensive_institution_signal_outgroup]))
        if op_rhet_intensive_marg == 'E' and op_rhet_extensive_marg == 'E':
            ag.op_in_belief = np.mean([ag.op_in_belief,ag.subscribed_proportions['intensive']*intensive_institution_signal_ingroup+ag.subscribed_proportions['extensive']*extensive_institution_signal_ingroup])
            ag.op_out_belief = np.mean([ag.op_out_belief,ag.subscribed_proportions['intensive']*intensive_institution_signal_outgroup+ag.subscribed_proportions['extensive']*extensive_institution_signal_outgroup])
            ag.reacted_to = 'EE'
        elif op_rhet_intensive_marg == 'E' and op_rhet_extensive_marg == 'S':
            ag.op_in_belief = np.mean([ag.op_in_belief,ag.subscribed_proportions['intensive']*intensive_institution_signal_ingroup + (1-ag.subscribed_proportions['intensive'])*ag.op_in_belief])
            ag.op_out_belief = np.mean([ag.op_out_belief,ag.subscribed_proportions['intensive']*intensive_institution_signal_outgroup + (1-ag.subscribed_proportions['intensive'])*ag.op_out_belief])
            ag.reacted_to = 'ES'
        elif op_rhet_intensive_marg == 'S' and op_rhet_extensive_marg == 'E':
            ag.op_in_belief = np.mean([ag.op_in_belief,ag.subscribed_proportions['extensive']*extensive_institution_signal_ingroup + (1-ag.subscribed_proportions['extensive'])*ag.op_in_belief])
            ag.op_out_belief = np.mean([ag.op_out_belief,ag.subscribed_proportions['extensive']*extensive_institution_signal_outgroup + (1-ag.subscribed_proportions['extensive'])*ag.op_out_belief])
            ag.reacted_to = 'SE'
        else:
            ag.reacted_to = 'SS'
            pass
    
    ''' Calculate community opinions in this timestep'''
    community_opinion_intensive = np.mean([ag.opinion for ag in sampled_agents if ag.reacted_to == 'EE' or ag.reacted_to == 'ES'])
    community_opinion_extensive = np.mean([ag.opinion for ag in sampled_agents if ag.reacted_to == 'EE' or ag.reacted_to == 'SE'])

    ''' Agents update their beliefs based on the community opinion'''
    for ag in sampled_agents:
        if ag.reacted_to == 'EE':
            ag.op_in_belief = np.mean([ag.op_in_belief,ag.subscribed_proportions['intensive']*community_opinion_intensive + ag.subscribed_proportions['extensive']*community_opinion_extensive])
        elif ag.reacted_to == 'ES':
            ag.op_in_belief = np.mean([ag.op_in_belief,ag.subscribed_proportions['intensive']*community_opinion_intensive + (1-ag.subscribed_proportions['intensive'])*ag.op_in_belief])
        elif ag.reacted_to == 'SE':
            ag.op_in_belief = np.mean([ag.op_in_belief,ag.subscribed_proportions['extensive']*community_opinion_extensive + (1-ag.subscribed_proportions['extensive'])*ag.op_in_belief])
        else:
            pass
    
    ''' Institutions update their subscriber lists and beliefs based on the community opinion'''
    extensive_institution.subscriber_list.append([ag for ag in sampled_agents if ag.reacted_to == 'EE' or ag.reacted_to == 'SE'])
    extensive_institution.institution_community_opinion = np.mean([ag.opinion for ag in sampled_agents if ag.reacted_to == 'EE' or ag.reacted_to == 'SE'])
    extensive_institution.institution_community_ingroup_belief = np.mean([ag.op_in_belief for ag in sampled_agents if ag.reacted_to == 'EE' or ag.reacted_to == 'SE'])
    extensive_institution.institution_community_outgroup_belief = np.mean([ag.op_out_belief for ag in sampled_agents if ag.reacted_to == 'EE' or ag.reacted_to == 'SE'])

    intensive_institution.subscriber_list.append([ag for ag in sampled_agents if ag.reacted_to == 'EE' or ag.reacted_to == 'ES'])
    intensive_institution.institution_community_opinion = np.mean([ag.opinion for ag in sampled_agents if ag.reacted_to == 'EE' or ag.reacted_to == 'ES'])
    intensive_institution.institution_community_ingroup_belief = np.mean([ag.op_in_belief for ag in sampled_agents if ag.reacted_to == 'EE' or ag.reacted_to == 'ES'])
    intensive_institution.institution_community_outgroup_belief = np.mean([ag.op_out_belief for ag in sampled_agents if ag.reacted_to == 'EE' or ag.reacted_to == 'ES'])



    print(iter_ts, 'Institution community opinions:', extensive_institution.institution_community_opinion, intensive_institution.institution_community_opinion)
    print(iter_ts, 'Institution community ingroup beliefs:', extensive_institution.institution_community_ingroup_belief, intensive_institution.institution_community_ingroup_belief)
    print(iter_ts, 'Institution community outgroup beliefs:', extensive_institution.institution_community_outgroup_belief, intensive_institution.institution_community_outgroup_belief)

        
        

