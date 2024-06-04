'''
Created on 15 Jan 2023

@author: Atrisha
'''

from collections import Counter
import copy
import csv
import functools
from math import isnan
import math
from multiprocessing import Process
import operator
import os
from pathlib import Path
import re

from conda.common._logic import TRUE
import gymnasium
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from scipy import optimize
import scipy
from scipy.special import softmax
from scipy.stats import beta, norm, halfnorm, pearsonr
from sympy import Symbol
from sympy import simplify
from sympy.solvers import solve
from sympy.stats import P, E, variance, Beta, Normal
from tabulate import tabulate
import torch

import constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils
from tqdm import tqdm








class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "sim_s1"}

    def create_group(self, agents, listening_proportions_type, listened_to_type , opinion_threshold, compare_op):
        """Creates a group of agents based on listening type and opinion.

        Args:
        agents (list): List of agents.
        listened_to_type (str): The type of listening ('intensive', 'extensive', or 'both').
        opinion_threshold (float): The threshold for opinion comparison.
        compare_op (callable): Comparison operator (e.g., operator.lt for less than).

        Returns:
        list: List of tuples (opinion, listening proportion) for the group.
        """
        return [(ag.opinion[ag.norm_context], ag.listening_proportions[listening_proportions_type])
                for ag in agents if ag.listened_to == listened_to_type and compare_op(ag.opinion[ag.norm_context], opinion_threshold)]

    def calculate_update(self, group, update_rate):
        """Calculates the update values for a group.

        Args:
        group (list): List of tuples (opinion, listening proportion).
        update_rate (float): The update rate.

        Returns:
        tuple: Updated values (a_prime, b_prime).
        """
        O, W = np.array([x[0] for x in group]), np.array([x[1] for x in group])
        W_normalized = W / np.sum(W)
        signal = np.dot(O, W_normalized)
        a_prime = signal * update_rate
        b_prime = update_rate - a_prime
        return a_prime, b_prime
    
    def update_simple(self, agent, agents, update_rate):
        theta_prime_rate_appr = None if len([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] >=0.5]) == 0 else np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] >=0.5])
        theta_prime_rate_disappr = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] <0.5]) if len([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] <0.5]) > 0 else None
        if theta_prime_rate_appr is None:
            a_prime = theta_prime_rate_appr * update_rate
            b_prime = update_rate - a_prime
            if agent.opinion[agent.norm_context] >= 0.5:
                agent.common_posterior_ingroup = (agent.common_posterior_ingroup[0] + a_prime, agent.common_posterior_ingroup[1] + b_prime)
            else:
                agent.common_posterior_outgroup = (agent.common_posterior_ingroup[0] + a_prime, agent.common_posterior_ingroup[1] + b_prime)
        if theta_prime_rate_disappr is None:
            a_prime = theta_prime_rate_disappr * update_rate
            b_prime = update_rate - a_prime
            if agent.opinion[agent.norm_context] < 0.5:
                agent.common_posterior_outgroup = (agent.common_posterior_outgroup[0] + a_prime, agent.common_posterior_outgroup[1] + b_prime)    
            else:
                agent.common_posterior_ingroup = (agent.common_posterior_ingroup[0] + a_prime, agent.common_posterior_ingroup[1] + b_prime)


    def update_agent(self, agent, agents, update_rate):
        """Updates the agent based on its listening type and opinion.

        Args:
        agent (Agent): The agent to be updated.
        agents (list): List of all agents.
        update_rate (float): The update rate.
        """
        # Determine comparison operator based on agent's opinion
        compare_op = operator.lt if agent.opinion[agent.norm_context] < 0.5 else operator.ge

        # Update logic for 'intensive' listening type
        if agent.listened_to == 'intensive':
            ingroup_intensives = self.create_group(agents, 'intensive', 'intensive', 0.5, compare_op) 
            ingroup_boths = self.create_group(agents, 'intensive', 'both', 0.5, compare_op)
            ingroup = ingroup_intensives + ingroup_boths
            a_prime, b_prime = self.calculate_update(ingroup, update_rate)
            agent.common_prior_ingroup = (agent.common_prior_ingroup[0] + a_prime, agent.common_prior_ingroup[1] + b_prime)
        elif agent.listened_to == 'extensive':
            # Update logic for 'extensive' listening type
            outgroup_extensives = self.create_group(agents, 'extensive', 'extensive', 0.5, operator.lt if compare_op==operator.ge else operator.ge)
            outgroup_boths = self.create_group(agents, 'extensive', 'both', 0.5, operator.lt if compare_op==operator.ge else operator.ge)
            update_group_outgroup = outgroup_extensives + outgroup_boths
            
            a_prime, b_prime = self.calculate_update(update_group_outgroup, update_rate)
            agent.common_prior_outgroup = (agent.common_prior_outgroup[0] + a_prime, agent.common_prior_outgroup[1] + b_prime)

            ingroup_extensives = self.create_group(agents, 'extensive', 'extensive', 0.5, compare_op)
            ingroup_boths = self.create_group(agents, 'extensive', 'both', 0.5, compare_op)
            update_group_ingroup = ingroup_extensives + ingroup_boths
            
            a_prime, b_prime = self.calculate_update(update_group_ingroup, update_rate)
            agent.common_prior_ingroup = (agent.common_prior_ingroup[0] + a_prime, agent.common_prior_ingroup[1] + b_prime)
            
        elif agent.listened_to == 'both':
            ''' agent listened to both '''
            ''' First update when agent listens to intensive '''
            ingroup_boths_intensives = self.create_group(agents, 'intensive', 'both', 0.5, compare_op) 
            ingroup_intensives = self.create_group(agents, 'intensive', 'intensive', 0.5, compare_op)
            ingroup_from_intensive_listening = ingroup_boths_intensives + ingroup_intensives
            a_prime, b_prime = self.calculate_update(ingroup_from_intensive_listening, agent.listening_proportions['intensive']*update_rate)
            agent.common_prior_ingroup = (agent.common_prior_ingroup[0] + a_prime, agent.common_prior_ingroup[1] + b_prime)

            ''' Then update again when agent listens to extensive. Scaling the update rate accordingly  '''
            ingroup_boths_extensives = self.create_group(agents, 'extensive', 'both', 0.5, compare_op)
            ingroup_extensives = self.create_group(agents, 'extensive', 'extensive', 0.5, compare_op)
            ingroup_from_extensive_listening = ingroup_boths_extensives + ingroup_extensives
            a_prime, b_prime = self.calculate_update(ingroup_from_extensive_listening, agent.listening_proportions['extensive']*update_rate)
            agent.common_prior_ingroup = (agent.common_prior_ingroup[0] + a_prime, agent.common_prior_ingroup[1] + b_prime)

            ''' Update outgroup '''
            outgroup_extensives = self.create_group(agents, 'extensive', 'extensive', 0.5, operator.lt if compare_op==operator.ge else operator.ge)
            outgroup_boths = self.create_group(agents, 'extensive', 'both', 0.5, operator.lt if compare_op==operator.ge else operator.ge)
            update_group_outgroup = outgroup_extensives + outgroup_boths
            a_prime, b_prime = self.calculate_update(update_group_outgroup, agent.listening_proportions['extensive']*update_rate)
            agent.common_prior_outgroup = (agent.common_prior_outgroup[0] + a_prime, agent.common_prior_outgroup[1] + b_prime)
        else:
            ''' Agent listened to none '''
            pass
        if isinstance(agent.common_prior_ingroup,float) or isinstance(agent.common_prior_outgroup,float):
            f=1



        # Add similar logic for 'extensive' and 'both' cases

    
    def generate_opinions(self,distr_shape,distr_params):
        if distr_shape == 'uniform':
            ops = [o for o in np.random.uniform(low=0.5,high=1,size=int(self.num_players*distr_params['mean_op_degree_apr']))]
            ops.extend([o for o in np.random.uniform(low=0,high=0.5,size=int(self.num_players*(1-distr_params['mean_op_degree_apr'])))])
            np.random.shuffle(ops)
            if len(ops) < self.num_players:
                ops.extend([np.random.random()])
            return ops
        elif distr_shape == 'gaussian':
            ops = np.random.normal(distr_params['mean_op_degree_apr'], distr_params['SD'], self.num_players)
            ops = np.clip(ops, 0, 1)
            np.random.shuffle(ops)
            if len(ops) < self.num_players:
                ops.extend([np.random.random()])
            return ops
        elif distr_shape == 'U':
            mu1, std1 = distr_params['mean_op_degree_apr'], distr_params['SD']  # First Gaussian distribution
            mu2, std2 = distr_params['mean_op_degree_disapr'], distr_params['SD']   # Second Gaussian distribution
            
            # Generate data points from the two Gaussian distributions
            data1 = np.random.normal(mu1, std1, int(self.num_players*distr_params['apr_weight']))
            data2 = np.random.normal(mu2, std2, int(self.num_players*(1-distr_params['apr_weight'])))
            
            # Create a U-shaped distribution by combining the two datasets
            ops = np.concatenate((data1, data2))
            ops = list(np.clip(ops, 0, 1))
            np.random.shuffle(ops)
            if len(ops) < self.num_players:
                ops.extend([np.random.random()])
            return ops
            

    def __init__(self, render_mode=None, attr_dict = None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        if attr_dict is not None:
            for key in attr_dict:
                setattr(self, key, attr_dict[key])
        self.num_players = 100 if not hasattr(self, 'num_players') else self.num_players
        self.update_rate = 2
        self.sanc_marginal_target = 0.2
        #self.norm_context_list = ['n1','n2','n3','n4']
        self.norm_context_list = ['n1']
        self.security_util = 0.1
        sanctioning_vals = np.random.normal(0.5, 0.1, self.num_players)
        self.mean_sanction, self.mean_sanction_baseline = 0.5,0.5
        sanctioning_vals = np.clip(sanctioning_vals, 0, 1)
        self.possible_agents = [Player(r,self) for r in range(self.num_players)]
        self.results_map = dict()
        self.observations = None
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        
        ''' Define list of normative contexts and initial distributions '''
        #self.norm_contexts_distr = {x:0.25 for x in self.norm_context_list}
        if not hasattr(self, 'norm_contexts_distr'):
            #self.norm_contexts_distr = {'n1':0.4,'n2':0.2,'n3':0.3,'n4':0.1} 
            self.norm_contexts_distr = {'n1':1} 
        #self.norm_contexts_distr = {k:v/np.sum(list(self.norm_contexts_distr.values())) for k,v in self.norm_contexts_distr.items()}
        ''' Sample the player opinions based on their private contexts sampled from the context distribution '''
        try_ct = 0
        if not hasattr(self, 'players_private_contexts'):
            players_private_contexts = np.random.choice(a=list(self.norm_contexts_distr.keys()),size=self.num_players,p=list(self.norm_contexts_distr.values()))
            while(set(players_private_contexts) != set(self.norm_context_list)):
                try_ct+=1
                print('trying...',try_ct)
                players_private_contexts = np.random.choice(a=list(self.norm_contexts_distr.keys()),size=self.num_players,p=list(self.norm_contexts_distr.values()))
            self.players_private_contexts = players_private_contexts
        players_private_contexts  = self.players_private_contexts
        for idx,op in enumerate(players_private_contexts): self.possible_agents[idx].norm_context = players_private_contexts[idx]
        
        distr_params = {'mean_op_degree_apr':0.7,'mean_op_degree_disapr':0.4,'apr_weight':0.5,'SD':0.05} if self.distr_params is None else self.distr_params
        distr_shape = 'U' if self.distr_shape is None else self.distr_shape
        ops = self.generate_opinions(distr_shape,distr_params)
        self.num_appr = len([op for op in ops if op >= 0.5])
        self.num_disappr = len([op for op in ops if op < 0.5])
        self.mean_appr_opinion = np.mean([op for op in ops if op >= 0.5])
        self.mean_disappr_opinion = np.mean([op for op in ops if op < 0.5])
        
        '''
        opinions = np.random.choice([1,0],size=self.num_players,p=[norm_context_appr_rate['n1'], 1-norm_context_appr_rate['n1']])
        opinions = opinions.reshape((self.num_players,len(self.norm_context_list)))
        
        self.opinion_marginals = dict()
        for n_idx,norm_context in enumerate(self.norm_context_list):
            ops = opinions[:,n_idx]
            for idx,op in enumerate(ops): 
                self.possible_agents[idx].opinion[norm_context] = np.random.uniform(0.5,1) if op == 1 else np.random.uniform(0,0.5)
        '''
        for idx,op in enumerate(ops): 
            self.possible_agents[idx].opinion['n1'] = op
        for ag in self.possible_agents:
            ag.init_beliefs(self)
        
        for idx,s in enumerate(sanctioning_vals):
            agent_op = self.possible_agents[idx].opinion['n1']
            self.possible_agents[idx].sanction_capacity = np.random.normal(0.5,.1)#np.random.normal(agent_op, 0.1)
        ''' Define the marginal approval means'''
        
        self.agents = self.possible_agents
        
    
    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=0, high=1.0, shape=(1, 2), dtype=np.float16)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)
    
    def state(self,agent=None):
        pass
        

    def render(self,msg):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        ''' Maybe just display the mean beliefs of approval and payoffs stratified by each norm context.'''
        

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent.id: None for agent in self.agents}

        if not return_info:
            return observations
        else:
            infos = {agent: {} for agent in self.agents}
            return observations, infos

    def step(self, actions, iter_no, run_type):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # rewards for all agents are placed in the rewards dictionary to be returned
        ''' Since only the sender agent is learning, we do not need a full dict '''
        observed_action_values = np.array([ag.action[1] for ag in self.agents if ag.action[0]!=-1])
        num_observation = len(observed_action_values)
        num_participation = len([ag for ag in self.agents if ag.action[0]!=-1])/self.num_players
        run_type = 'execution' if isinstance(run_type,bool) and run_type==True else 'baseline' if isinstance(run_type,bool) and run_type==False else run_type
        if run_type == 'execution' or run_type == 'transition_genration':
            ''' Let the reward be inversely proportional to the opinion value extremity'''
            baseline_op_mean = np.mean([ag.opinion[ag.norm_context] for ag in self.agents])
            terminations = {agent.id: False for agent in self.agents}
            self.num_moves += 1
            env_truncation = self.num_moves >= self.NUM_ITERS
            truncations = {agent.id: env_truncation for agent in self.agents}
    
            ''' Observation is the next state, or the common prior change '''
            num_appr = len([ag.action[0] for ag in self.agents if ag.action[0]==1 and ag.action[0]!=-1])
            num_disappr = len([ag.action[0] for ag in self.agents if ag.action[0]==0 and ag.action[0]!=-1])
            
            if num_observation > 0:
                
                theta_prime_rate_appr = 1 if len([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] >=0.5]) == 0 else np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] >=0.5])
                theta_prime_rate_disappr = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] <0.5]) if len([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] <0.5]) > 0 else 0
                for agent in self.agents:
                    """ Posteriors become priors for the next round in this function """
                    agent.common_prior_outgroup = utils.distributionalize(agent.common_prior_outgroup,agent.common_posterior_outgroup)
                    agent.common_prior_ingroup = utils.distributionalize(agent.common_prior_ingroup,agent.common_posterior_ingroup)
                    if run_type == 'transition_genration':
                        self.update_simple(agent,self.agents,self.update_rate)
                    else:
                        self.update_agent(agent,self.agents,self.update_rate)
            else:
                theta_prime_rate_appr = 1
                theta_prime_rate_disappr = 0
                for agent in self.agents:
                    agent.common_prior_outgroup = utils.distributionalize(agent.common_prior_outgroup,agent.common_posterior_outgroup)
                    agent.common_prior_ingroup = utils.distributionalize(agent.common_prior_ingroup,agent.common_posterior_ingroup)
            for agent in self.agents:
                if (agent.common_prior_outgroup[0]/np.sum(agent.common_prior_outgroup) - 0.5)*(agent.opinion[agent.norm_context]-0.5) > 0 \
                        or (agent.common_prior_ingroup[0]/np.sum(agent.common_prior_ingroup) - 0.5)*(agent.opinion[agent.norm_context]-0.5) < 0:
                        f=1
                    
            ''' Assign (institutional) rewards'''
            if self.extensive:
                rewards = (np.mean([num_appr/self.num_appr,num_disappr/self.num_disappr])-0.5)*2
            else:
                try:
                    if num_appr > 0:
                        mean_appr_degree = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] >= 0.5])
                        mean_sanctioning_capacity = np.mean([ag.action[3] for ag in self.agents if ag.action[0]!=-1])
                        rewards = 4*mean_appr_degree - 3
                    else:
                        rewards = -1
                except ValueError as e:
                    f=1
                    raise e
            
            observations = {'appr':np.mean([utils.beta_mean(agent.common_prior_ingroup) if agent.opinion[agent.norm_context] >= 0.5 else utils.beta_mean(agent.common_prior_outgroup) for agent in self.agents]),
                            'disappr':np.mean([utils.beta_mean(agent.common_prior_ingroup) if agent.opinion[agent.norm_context] < 0.5 else utils.beta_mean(agent.common_prior_outgroup) for agent in self.agents])}
            if observations['appr'] < 0.5:
                f=1
            if self.only_intensive:
                x = [(ag.opinion[ag.norm_context],ag.listened_to,ag.action[0],ag.action[5],ag.action[6]) for ag in self.agents if ag.opinion[ag.norm_context] >=0.5]
                x.sort(key=lambda x: x[0])
            else:
                x = [(ag.opinion[ag.norm_context],ag.listened_to,ag.action[0],ag.action[5],ag.action[6]) for ag in self.agents if ag.opinion[ag.norm_context] >=0.5]
                x.sort(key=lambda x: x[0])
            # typically there won't be any information in the infos, but there must
            # still be an entry for each agent
            
            infos = x
    
            if env_truncation:
                self.agents = []
    
            if self.render_mode == "human":
                self.render(iter_no)
            return observations, rewards, terminations, truncations, infos
        elif run_type == 'baseline':
            num_appr = len([ag.action[0] for ag in self.agents if ag.action[0]==1 and ag.action[0]!=-1])
            num_disappr = len([ag.action[0] for ag in self.agents if ag.action[0]==0 and ag.action[0]!=-1])
            if num_observation > 0:
                theta_prime_rate = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1])
                #theta_prime_by_nums = num_appr /(num_appr+num_disappr)
                #theta_prime_rate = theta_prime_by_nums
                a_prime = theta_prime_rate*self.update_rate
                b_prime =  self.update_rate-a_prime
                self.prior_baseline = (self.prior_baseline[0]+a_prime, self.prior_baseline[1]+b_prime)
                
                a_prime_prop = num_participation*self.update_rate
                b_prime_prop =  self.update_rate-a_prime_prop
                self.prior_prop_baseline = (self.prior_prop_baseline[0]+a_prime, self.prior_prop_baseline[1]+b_prime)
                
                mean_appr_degree = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1])
                mean_sanctioning_capacity = np.mean([ag.action[3] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] >= 0.5])
                #print('------>',self.common_proportion_prior[0]/np.sum(self.common_proportion_prior), self.common_prior[0]/np.sum(self.common_prior), '||', mean_appr_degree,mean_sanctioning_capacity,)
                self.mean_sanction_baseline = mean_sanctioning_capacity*mean_appr_degree
        else:
            raise ValueError(f"Invalid run type: {run_type}")

    
    
    @property
    def common_prior_mean(self):   
        return self.common_prior[0]/sum(self.common_prior)
    
    def generate_row_entry(self, group_type, calc_participation=True):
        group = [ag for ag in self.agents if ag.opinion[ag.norm_context] >= 0.5] if group_type == 'appr' else [ag for ag in self.agents if ag.opinion[ag.norm_context] < 0.5]
        mean_out_belief = np.mean([ag.common_prior_outgroup[0]/np.sum(ag.common_prior_outgroup) for ag in group])
        mean_opinion = self.mean_appr_opinion if group_type == 'appr' else self.mean_disappr_opinion
        mean_in_belief = np.mean([ag.common_prior_ingroup[0]/np.sum(ag.common_prior_ingroup) for ag in group])
        participation = len([ag for ag in group if ag.action_code != -1])/len(group) if calc_participation else None
        return mean_opinion, mean_in_belief, mean_out_belief, participation
    
        
class Player():
    
    def __init__(self,id, env):
        self.id = id
        self.payoff_tol = constants.payoff_tol
        self.opinion = dict()
        self.opinion_val = dict()
        self.norm_context = 'n1'
        '''
        if real_p:
            self.shadow_player = Player(-id,False)
        '''
        
        self.total_reward = 0
        self.total_participation = 0
        self.historical_listened_to = []
        self.env = env
        self.rhet_thresh = np.random.beta(0.3,3)
    
    def init_beliefs(self,env):
        if not env.homogenous_priors:
            
            if self.opinion['n1'] >= 0.5:
                _sum = np.sum(env.common_prior_appr_input)
                sample = np.random.randint(1,math.ceil(_sum/2))
                self.common_prior_ingroup = (_sum-sample,sample)
                self.common_prior_outgroup = (self.common_prior_ingroup[1],self.common_prior_ingroup[0])
                assert self.common_prior_ingroup[0] >= self.common_prior_ingroup[1], f"Ingroup prior first element should be greater or equal to second. _sum: {_sum}, sample: {sample}"
                assert self.common_prior_outgroup[0] < self.common_prior_outgroup[1], f"Outgroup prior first element should be less than second. _sum: {_sum}, sample: {sample}"
            else:
                _sum = np.sum(env.common_prior_appr_input)
                sample = np.random.randint(1,math.ceil(_sum/2))
                self.common_prior_outgroup = (_sum-sample,sample)
                self.common_prior_ingroup = (self.common_prior_outgroup[1],self.common_prior_outgroup[0])
                assert self.common_prior_ingroup[0] < self.common_prior_ingroup[1], f"Ingroup prior first element should be less than to second. _sum: {_sum}, sample: {sample}"
                assert self.common_prior_outgroup[0] >= self.common_prior_outgroup[1], f"Outgroup prior first element should be greater than equal second. _sum: {_sum}, sample: {sample}"
            self.common_prior_outgroup_init = self.common_prior_outgroup[0]/np.sum(self.common_prior_outgroup)
            self.common_posterior_ingroup = self.common_prior_ingroup
            self.common_posterior_outgroup = self.common_prior_outgroup
            '''
            self.common_prior_ingroup = env.common_prior_appr if self.opinion['n1'] >= 0.5 else env.common_prior_disappr
            self.common_prior_outgroup = env.common_prior_disappr if self.opinion['n1'] >= 0.5 else env.common_prior_appr
            self.common_prior_outgroup_init = self.common_prior_outgroup[0]/np.sum(self.common_prior_outgroup)
            self.common_posterior_ingroup = env.common_prior_appr if self.opinion['n1'] >= 0.5 else env.common_prior_disappr
            self.common_posterior_outgroup = env.common_prior_disappr if self.opinion['n1'] >= 0.5 else env.common_prior_appr
            '''
        else:
            self.common_prior_ingroup = env.common_prior_appr if self.opinion['n1'] >= 0.5 else env.common_prior_disappr
            self.common_prior_outgroup = env.common_prior_disappr if self.opinion['n1'] >= 0.5 else env.common_prior_appr
            self.common_prior_outgroup_init = self.common_prior_outgroup[0]/np.sum(self.common_prior_outgroup)
            self.common_posterior_ingroup = env.common_prior_appr if self.opinion['n1'] >= 0.5 else env.common_prior_disappr
            self.common_posterior_outgroup = env.common_prior_disappr if self.opinion['n1'] >= 0.5 else env.common_prior_appr
        self.common_proportion_prior = env.common_proportion_prior
        self.common_proportion_posterior = env.common_proportion_prior
        self.group = 'appr' if self.opinion['n1'] >= 0.5 else 'disappr'
    
    def act(self, env, run_type, baseline):
        return self.act_with_sanction_cap(env,run_type,baseline)
    
    def outgroup_rhetoric_estimate(self,op_hat_out):
        o = op_hat_out
        rhet =  -1.6425*o**2 + 3.6693*o - 1.3048 
        return min(max(0,rhet),1)
        
    def opt_rhetoric_intensity_func(self, env, o,op_hat_in,op_hat_out):
        op_hat_in = op_hat_in[0]/np.sum(op_hat_in) if isinstance(op_hat_in,tuple) else op_hat_in
        op_hat_out = op_hat_out[0]/np.sum(op_hat_out) if isinstance(op_hat_out,tuple) else op_hat_out
        """ This comes from the estimated function of the optimal rhetoric intensity"""
        model = env.rhetoric_estimation_model
        '''
        _diff_list = []
        for n_o in np.linspace(0,1,100):
            n_i = model.predict(np.asarray([op_hat_in, o, env.alpha, n_o, op_hat_out]).reshape(1,-1))
            n_o_resp = model.predict(np.asarray([1-op_hat_out, 1-op_hat_out, env.alpha, n_i[0], 1-op_hat_in]).reshape(1,-1))
            _diff_list.append((abs(n_o-n_o_resp[0]),n_i[0],n_o_resp[0]))
        _diff_list.sort(key=lambda x: x[0])
        rhet_eq = max(min(_diff_list[0][1],1),0)
        '''
        rhet_eq = model.predict(np.asarray([op_hat_in, o, env.alpha, 1-op_hat_out, op_hat_out]).reshape(1,-1))
        rhet_eq = max(min(rhet_eq,1),0)
        r_comp_thresh = (self.env.alpha+self.outgroup_rhetoric_estimate(op_hat_out))/ (op_hat_in + op_hat_out)
        '''if o > r_comp_thresh:
            return rhet_eq
        else:
            return 0'''
        return rhet_eq
    
    def simple_act(self, env, run_type, baseline):
        rhet_thresh = self.rhet_thresh
        u_bar = env.security_util
        op = self.opinion[self.norm_context]
        mean_from_params = lambda params: params[0]/(params[0]+params[1]) if isinstance(params,tuple) else params
        n_p = self.common_proportion_posterior if isinstance(self.common_proportion_posterior,float) else self.common_proportion_posterior[0]/np.sum(self.common_proportion_posterior)
        op_degree = op if op >= 0.5 else (1-op)
        conc_prop = n_p if op >= 0.5 else (1-n_p)
        opt_rhetoric = dict()
        institution = run_type['institutions']

        theta_ingroup = self.pseudo_update_posteriors[institution.type]['ingroup']
        theta_outgroup = self.pseudo_update_posteriors[institution.type]['outgroup']
        theta_ingroup = theta_ingroup[0]/np.sum(theta_ingroup) if isinstance(theta_ingroup,tuple) else theta_ingroup
        theta_outgroup = theta_outgroup[0]/np.sum(theta_outgroup) if isinstance(theta_outgroup,tuple) else theta_outgroup
        common_rhetoric = min(self.opt_rhetoric_intensity_func(env,op_degree,theta_ingroup if self.group=='appr' else 1-theta_ingroup,theta_outgroup if self.group=='appr' else 1-theta_outgroup),1)
        
        if common_rhetoric > rhet_thresh:
            self.action_code = 1 if op >= 0.5 else 0
            self.listened_to = institution.type
            self.rhetoric_intensity = common_rhetoric
        else:
            self.action_code = -1
            self.listened_to = 'none'
            self.rhetoric_intensity = 0

        if self.listened_to != 'none':
            self.common_posterior_ingroup = self.pseudo_update_posteriors[self.listened_to]['ingroup']
            self.common_posterior_outgroup = self.pseudo_update_posteriors[self.listened_to]['outgroup']
        else:
            self.common_posterior_ingroup = self.common_prior_ingroup
            self.common_posterior_outgroup = self.common_prior_outgroup    
        self.common_posteriors = {'ingroup':self.common_posterior_ingroup,'outgroup':self.common_posterior_outgroup}
        self.action =(self.action_code,None,self.opinion[self.norm_context],self.rhetoric_intensity,self.listened_to,None,
                        self.common_posteriors)
        return self.action


    def act_with_sanction_cap(self, env, run_type, baseline):
        rhet_thresh = self.rhet_thresh
        u_bar = env.security_util
        op = self.opinion[self.norm_context]
        if op < 0.5:
            f=1
        
        mean_from_params = lambda params: params[0]/(params[0]+params[1]) if isinstance(params,tuple) else params
        n_p = self.common_proportion_posterior if isinstance(self.common_proportion_posterior,float) else self.common_proportion_posterior[0]/np.sum(self.common_proportion_posterior)
        op_degree = op if op >= 0.5 else (1-op)
        conc_prop = n_p if op >= 0.5 else (1-n_p)
        single_institution_env = True if not all(run_type['institutions'].values()) else False
        
        if not baseline:
            opt_rhetoric = dict()
            for inst in ['extensive','intensive']:
                if run_type['institutions'][inst] is not None:
                    institution = run_type['institutions'][inst]
                    theta_ingroup = self.pseudo_update_posteriors[institution.type]['ingroup']
                    theta_outgroup = self.pseudo_update_posteriors[institution.type]['outgroup']
                    theta_ingroup = theta_ingroup[0]/np.sum(theta_ingroup) if isinstance(theta_ingroup,tuple) else theta_ingroup
                    theta_outgroup = theta_outgroup[0]/np.sum(theta_outgroup) if isinstance(theta_outgroup,tuple) else theta_outgroup
                    opt_rhetoric[institution.type] = min(self.opt_rhetoric_intensity_func(env,op_degree,theta_ingroup if self.group=='appr' else 1-theta_ingroup,theta_outgroup if self.group=='appr' else 1-theta_outgroup),1)
            
            if single_institution_env:
                common_rhetoric = next(iter(opt_rhetoric.values()))
                if common_rhetoric > rhet_thresh:
                    self.action_code = 1 if op >= 0.5 else 0
                    self.listened_to = next(iter(opt_rhetoric))
                    self.rhetoric_intensity = common_rhetoric
                else:
                    self.action_code = -1
                    self.listened_to = 'none'
                    self.rhetoric_intensity = 0

                if self.listened_to != 'none':
                    self.common_posterior_ingroup = self.pseudo_update_posteriors[self.listened_to]['ingroup']
                    self.common_posterior_outgroup = self.pseudo_update_posteriors[self.listened_to]['outgroup']
                else:
                    self.common_posterior_ingroup = self.common_prior_ingroup
                    self.common_posterior_outgroup = self.common_prior_outgroup    
                self.common_posteriors = {'ingroup':self.common_posterior_ingroup,'outgroup':self.common_posterior_outgroup}
                self.action =(self.action_code,None,self.opinion[self.norm_context],self.rhetoric_intensity,self.listened_to,None,
                              self.common_posteriors)
                
                self.historical_listened_to.append(self.listened_to)
                inst_weights = [self.historical_listened_to.count('extensive')+self.historical_listened_to.count('both'),self.historical_listened_to.count('intensive')+self.historical_listened_to.count('both')]
                inst_weights = [w/sum(inst_weights) for w in inst_weights] if sum(inst_weights) > 0 else [0,0]
                self.listening_proportions = {'extensive':inst_weights[0],'intensive':inst_weights[1]}
            else:
                self.opt_rhetoric_extensive = opt_rhetoric['extensive']
                self.opt_rhetoric_intensive = opt_rhetoric['intensive']
                if self.opt_rhetoric_extensive > rhet_thresh and self.opt_rhetoric_intensive > rhet_thresh:
                    self.action_code = 1 if op >= 0.5 else 0
                    self.listened_to = 'both'
                elif self.opt_rhetoric_extensive > rhet_thresh and self.opt_rhetoric_intensive <= rhet_thresh:
                    self.action_code = 1 if op >= 0.5 else 0
                    self.listened_to = 'extensive'
                    self.rhetoric_intensity = self.opt_rhetoric_extensive
                elif self.opt_rhetoric_extensive <= rhet_thresh and self.opt_rhetoric_intensive > rhet_thresh:
                    self.action_code = 1 if op >= 0.5 else 0
                    self.listened_to = 'intensive'
                    self.rhetoric_intensity = self.opt_rhetoric_intensive
                else:
                    self.action_code = -1
                    self.listened_to = 'none'
                    self.rhetoric_intensity = 0
                
                self.historical_listened_to.append(self.listened_to)
                
                inst_weights = [self.historical_listened_to.count('extensive')+self.historical_listened_to.count('both'),self.historical_listened_to.count('intensive')+self.historical_listened_to.count('both')]
                if sum(inst_weights) > 0:
                    inst_weights = [w/sum(inst_weights) for w in inst_weights] if sum(inst_weights) > 0 else [0.5,0.5]
                self.listening_proportions = {'extensive':inst_weights[0],'intensive':inst_weights[1]}
                
                if self.listened_to == 'both':
                    self.common_posterior_outgroup = inst_weights[0]*mean_from_params(self.pseudo_update_posteriors['extensive']['outgroup']) + inst_weights[1]*mean_from_params(self.pseudo_update_posteriors['intensive']['outgroup'])
                    self.common_posterior_ingroup = inst_weights[0]*mean_from_params(self.pseudo_update_posteriors['extensive']['ingroup']) + inst_weights[1]*mean_from_params(self.pseudo_update_posteriors['intensive']['ingroup'])
                                       
                    self.rhetoric_intensity = inst_weights[0]*self.opt_rhetoric_extensive + inst_weights[1]*self.opt_rhetoric_intensive
                    
                else:
                    if self.listened_to == 'intensive':
                        self.common_posterior_outgroup = self.pseudo_update_posteriors['intensive']['outgroup']
                        self.common_posterior_ingroup = self.pseudo_update_posteriors['intensive']['ingroup']
                    elif self.listened_to == 'extensive':
                        self.common_posterior_outgroup = self.pseudo_update_posteriors['extensive']['outgroup']
                        self.common_posterior_ingroup = self.pseudo_update_posteriors['extensive']['ingroup']
                    else:
                        self.common_posterior_ingroup = self.common_prior_ingroup
                        self.common_posterior_outgroup = self.common_prior_outgroup
                        
                self.common_posteriors = {'ingroup':self.common_posterior_ingroup,'outgroup':self.common_posterior_outgroup}
                self.action =(self.action_code,None,self.opinion[self.norm_context],self.rhetoric_intensity,self.listened_to,None,
                              self.common_posteriors)
        else:
            theta_baseline = env.prior_baseline[0]/sum(env.prior_baseline)
            prop_baseline = env.prior_prop_baseline[0]/sum(env.prior_prop_baseline)
            op_degree = op if op >= 0.5 else (1-op)
            conc_prop = prop_baseline if op >= 0.5 else (1-prop_baseline)
            conc_deg = theta_baseline if op >= 0.5 else (1-theta_baseline)
            if op_degree*conc_prop*(1-conc_deg)*1**(-conc_deg) > env.sanc_marginal_target:
                opt_sanc = 1
            else:
                opt_sanc = math.pow(env.sanc_marginal_target/(op_degree*conc_prop*(1-conc_deg)),-1/conc_deg)
            self.sanction_intensity = opt_sanc
            
            util_baseline = lambda op : op*(self.sanction_intensity**(1-theta_baseline))*prop_baseline if op >= 0.5 else (1-op)*(self.sanction_intensity**theta_baseline)*(1-prop_baseline)
            
            if util_baseline(op) < u_bar:
                self.action_code_baseline = -1
                self.action_util_baseline = u_bar
            else:
                self.action_code_baseline = 1 if op >= 0.5 else 0
                self.action_util_baseline = util_baseline(op)
            
            
            
            self.action =(self.action_code_baseline,self.action_util_baseline,self.opinion[self.norm_context],self.sanction_intensity)
        
        return self.action
    
        
    def generate_posteriors_deprecated(self,env,institution,common_proportion_prior,update_type):
        opt_signals = institution.opt_signals if institution is not None else institution.opt_signals
        if update_type == 'ingroup':
            group_type = 'appr' if self.opinion[self.norm_context] >= 0.5 else 'disappr'
        elif update_type == 'outgroup':
            group_type = 'appr' if self.opinion[self.norm_context] < 0.5 else 'disappr'
        else:
            raise ValueError('Invalid update type')
        opt_signals = opt_signals[group_type]
        common_prior = self.common_prior_ingroup if update_type=='ingroup' else self.common_prior_outgroup
        common_prior_mean = common_prior[0]/np.sum(common_prior)
        curr_state = common_prior_mean
        valid_dist = True
        try:
            _curr_state = round(curr_state,1)
            signal_distribution = opt_signals[_curr_state]
        except KeyError:
            print('Info:')
            print(self.opinion[self.norm_context])
            print(update_type)
            print(curr_state)
            print(opt_signals)
            print(common_prior)
            raise ValueError('State not in distribution:'+str(curr_state))
        ''' Since we are running with only one intensive instituion, the signals are a tuple - (the intensive signal for approval, the intensive signal for disapproval)'''
        if update_type == 'ingroup':
            signal_distribution = signal_distribution[1] if self.opinion[self.norm_context] >= 0.5 else signal_distribution[0]
        else:
            signal_distribution = signal_distribution[1] if self.opinion[self.norm_context] >= 0.5 else signal_distribution[0]

        if institution.type == 'extensive' or (institution.type == 'intensive' and update_type == 'ingroup'):
            try:
                if round(abs(signal_distribution-common_prior_mean),1) > env.normal_constr_w:
                    common_posterior,common_proportion_posterior =  common_prior, common_proportion_prior
                    valid_dist = False
                     
            except TypeError:
                print('update type',update_type)
                print(self.opinion[self.norm_context],common_prior,signal_distribution,self.pseudo_update_posteriors if hasattr(self, 'pseudo_update_posteriors') else 'first run')
                raise
        '''
            This method updates the posterior for the population (posterior over the rate of approval) based on the signal dristribution.
            Since signal distribution is a Bernoulli, we can get individual realizations of 0 and 1 separately, and then take the expectation.
        '''
        if valid_dist:
            if not hasattr(env,'posterior_prediction_model'):
                def _post(x,priors_rescaled,likelihood_rescaled):
                    prior_x = priors_rescaled[x]
                    
                    ''' This evaluates the likelohood as conditioned on the state value x
                        Find prob of signal distr. (this connects to the 'state', i.e., have state information)
                        Then calc liklihood of the signal realization
                    '''
                    signal_param_prob = likelihood_rescaled[x]
                    lik = lambda x: signal_param_prob*signal_distribution if x == 1 else signal_param_prob*(1-signal_distribution)
                    post = (prior_x*lik(1),prior_x*lik(0))
                    return post
                all_posteriors = []
                if update_type == 'ingroup':
                    posterior_space = [x for x in np.linspace(0.01, 0.99, 50) if x < 0.5] if self.opinion[self.norm_context] < 0.5 else [x for x in np.linspace(0.01, 0.99, 50) if x >= 0.5]
                else:
                    posterior_space = [x for x in np.linspace(0.01, 0.99, 50) if x < 0.5] if self.opinion[self.norm_context] >= 0.5 else [x for x in np.linspace(0.01, 0.99, 50) if x >= 0.5]
                
                priors_rescaled, likelihood_rescaled = dict(), dict()
                for x in np.linspace(0.01,0.99,50):
                    priors_rescaled[x] = utils.beta_pdf(x, common_prior[0], common_prior[1],np.linspace(0.01,0.99,50))
                    _constr_distr = utils.Gaussian_plateu_distribution(signal_distribution,.01,env.normal_constr_w)
                    likelihood_rescaled[x] = _constr_distr.pdf(x)
                    #_constr_distr = utils.Gaussian_plateu_distribution(0,.01,self.normal_constr_sd)
                    #likelihood_rescaled[x] = _constr_distr.pdf(abs(x-signal_distribution))
                priors_rescaled = {k:v/sum(list(priors_rescaled.values())) for k,v in priors_rescaled.items()}
                likelihood_rescaled = {k:v/sum(list(likelihood_rescaled.values())) for k,v in likelihood_rescaled.items()}
                for x in posterior_space:
                    posteriors = _post(x,priors_rescaled,likelihood_rescaled)
                    ''' Since the signal realization will be based on the signal distribution, we can take the expectation of the posterior w.r.t each realization.'''
                    expected_posterior_for_state_x = (signal_distribution*posteriors[0]) + ((1-signal_distribution)*posteriors[1])
                    all_posteriors.append(expected_posterior_for_state_x)
                all_posteriors = [x/np.sum(all_posteriors) for x in all_posteriors]
                exp_x = np.sum([x*prob_x for x,prob_x in zip(posterior_space,all_posteriors)])
                var_x = _constr_distr.var()
            else:
                model = env.posterior_prediction_model[group_type]
                exp_x, var_x = utils.predict_posterior(model, common_prior[0], common_prior[1], signal_distribution)
                f=1
            '''
            print(exp_x)
            plt.figure()
            plt.plot(list(priors_rescaled.keys()),list(priors_rescaled.values()))
            plt.plot(list(likelihood_rescaled.keys()),list(likelihood_rescaled.values()))
            plt.plot(np.linspace(0.01,0.99,100),all_posteriors)
            plt.title('likelihood:'+str(signal_distribution)+','+str(self.common_prior[0]/sum(self.common_prior)))
            plt.show()
            '''
            common_posterior = utils.est_beta_from_mu_sigma(exp_x, var_x)
        ''' Sanity check '''
        _param_min = np.min(common_posterior)
        if _param_min < 1:
            _diff = 1-_param_min
            common_posterior = (common_posterior[0]+_diff,common_posterior[1]+_diff)
        common_proportion_posterior = common_proportion_prior[0]/np.sum(common_proportion_prior)
        if update_type == 'outgroup':
            _c = common_posterior[0]/np.sum(common_posterior)
            if (_c-0.5)*(self.opinion[self.norm_context]-0.5) > 0:
                f=1
        else:
            _c = common_posterior[0]/np.sum(common_posterior)
            if (_c-0.5)*(self.opinion[self.norm_context]-0.5) <= 0:
                f=1
        return common_posterior, common_proportion_posterior
        ''' Generate posteriors for norm support '''
        '''
        self.st_signal = (1,0) if signal_distribution > 0.5 else (0,1) if signal_distribution < 0.5 else (0.5,0.5)
        update_rate = 2
        self.common_proportion_posterior = (self.common_proportion_prior[0]+(update_rate*self.st_signal[0]),self.common_proportion_prior[1]+(update_rate*self.st_signal[1]))
        self.common_proportion_posterior = self.common_proportion_posterior[0]/np.sum(self.common_proportion_posterior).
        '''
        #self.common_proportion_posterior = exp_x

    def generate_posteriors(self,env,institution,common_proportion_prior,update_type):
        opt_signals = institution.opt_signals if institution is not None else institution.opt_signals
        if update_type == 'ingroup':
            group_type = 'appr' if self.opinion[self.norm_context] >= 0.5 else 'disappr'
        elif update_type == 'outgroup':
            group_type = 'appr' if self.opinion[self.norm_context] < 0.5 else 'disappr'
        else:
            raise ValueError('Invalid update type')
        opt_signals = opt_signals[group_type]
        common_prior = self.common_prior_ingroup if update_type=='ingroup' else self.common_prior_outgroup
        common_prior_mean = common_prior[0]/np.sum(common_prior)
        curr_state = common_prior_mean
        valid_dist = True
        try:
            _curr_state = round(curr_state,1)
            signal_distribution = opt_signals[_curr_state]
        except KeyError:
            print('Info:')
            print(self.opinion[self.norm_context])
            print(update_type)
            print(curr_state)
            print(opt_signals)
            print(common_prior)
            raise ValueError('State not in distribution:'+str(curr_state))
        ''' Since we are running with only one intensive instituion, the signals are a tuple - (the intensive signal for approval, the intensive signal for disapproval)'''
        if update_type == 'ingroup':
            signal_distribution = signal_distribution[1] if env.signal_cluster == 'appr' else signal_distribution[0]
        else:
            signal_distribution = signal_distribution[1] if env.signal_cluster == 'appr' else signal_distribution[0]

        if institution.type == 'extensive' or (institution.type == 'intensive' and update_type == 'ingroup'):
            try:
                if round(abs(signal_distribution-common_prior_mean),1) > env.normal_constr_w:
                    common_posterior,common_proportion_posterior =  common_prior, common_proportion_prior
                    valid_dist = False
                     
            except TypeError:
                print('update type',update_type)
                print(self.opinion[self.norm_context],common_prior,signal_distribution,self.pseudo_update_posteriors if hasattr(self, 'pseudo_update_posteriors') else 'first run')
                raise
        '''
            This method updates the posterior for the population (posterior over the rate of approval) based on the signal dristribution.
            Since signal distribution is a Bernoulli, we can get individual realizations of 0 and 1 separately, and then take the expectation.
        '''
        if valid_dist:
            if not hasattr(env,'posterior_prediction_model'):
                # Precompute distributions
                x_range = np.linspace(0.01, 0.99, 500)
                priors_rescaled = utils.beta_pdf(x_range, common_prior[0], common_prior[1], x_range)
                gaussian_distribution = utils.Gaussian_plateu_distribution(signal_distribution, 0.01, env.normal_constr_w)
                likelihood_rescaled = gaussian_distribution.pdf(x_range)

                # Normalize distributions
                priors_rescaled /= priors_rescaled.sum()
                likelihood_rescaled /= likelihood_rescaled.sum()

                # Calculate posterior
                all_posteriors = priors_rescaled * likelihood_rescaled
                all_posteriors /= np.sum(all_posteriors)
                #expected_posterior_for_state_x = (signal_distribution * all_posteriors + (1 - signal_distribution) * all_posteriors)
                expected_posterior_for_state_x = all_posteriors
                exp_x = (x_range * expected_posterior_for_state_x).sum()
                var_x = gaussian_distribution.var()
            else:
                model = env.posterior_prediction_model[group_type]
                exp_x, var_x = utils.predict_posterior(model, common_prior[0], common_prior[1], signal_distribution)
                f=1
            '''
            print(exp_x)
            plt.figure()
            plt.plot(list(priors_rescaled.keys()),list(priors_rescaled.values()))
            plt.plot(list(likelihood_rescaled.keys()),list(likelihood_rescaled.values()))
            plt.plot(np.linspace(0.01,0.99,100),all_posteriors)
            plt.title('likelihood:'+str(signal_distribution)+','+str(self.common_prior[0]/sum(self.common_prior)))
            plt.show()
            '''
            common_posterior = utils.est_beta_from_mu_sigma(exp_x, var_x)
            if self.opinion[self.norm_context] >= 0.5:
                if (update_type=='ingroup' and common_posterior[0]<common_posterior[1]) or (update_type=='outgroup' and common_posterior[0]>common_posterior[1]):
                    f=1
            else:    
                if (update_type=='ingroup' and common_posterior[0]>common_posterior[1]) or (update_type=='outgroup' and common_posterior[0]<common_posterior[1]):
                    f=1   
        ''' Sanity check '''
        _param_min = np.min(common_posterior)
        if _param_min < 1:
            _diff = 1-_param_min
            common_posterior = (common_posterior[0]+_diff,common_posterior[1]+_diff)
        common_proportion_posterior = common_proportion_prior[0]/np.sum(common_proportion_prior)
        if update_type == 'outgroup':
            _c = common_posterior[0]/np.sum(common_posterior)
            if (_c-0.5)*(self.opinion[self.norm_context]-0.5) > 0:
                f=1
        else:
            _c = common_posterior[0]/np.sum(common_posterior)
            if (_c-0.5)*(self.opinion[self.norm_context]-0.5) <= 0:
                f=1
        return common_posterior, common_proportion_posterior
        ''' Generate posteriors for norm support '''
        '''
        self.st_signal = (1,0) if signal_distribution > 0.5 else (0,1) if signal_distribution < 0.5 else (0.5,0.5)
        update_rate = 2
        self.common_proportion_posterior = (self.common_proportion_prior[0]+(update_rate*self.st_signal[0]),self.common_proportion_prior[1]+(update_rate*self.st_signal[1]))
        self.common_proportion_posterior = self.common_proportion_posterior[0]/np.sum(self.common_proportion_posterior).
        '''
        #self.common_proportion_posterior = exp_x
           

    
class StewardAgent():
    
    def __init__(self,qnetwork):
        self.qnetwork = qnetwork       
        
class Institution:
    def __init__(self, type, opt_signals=None):
        self.type=type
        self.subscriber_list = []
        self.signals = {'in_group':[],'out_group':[]}
        self.institution_community_opinion = None
        self.institution_community_ingroup_belief = None
        self.institution_community_outgroup_belief = None
        self.opt_signals = opt_signals
        
    def generate_signal(self, op_state):
        if self.type == 'intensive':
            return self.opt_signals[round(op_state,1)]
        else:
            return self.opt_signals[round(op_state,1)]
            
class RunInfo():
    
    def __init__(self,iter):
        self.iter = iter

def run_sim_single_institution(run_param):
    """ ENV SETUP """
    common_prior_appr_input, common_prior_appr, common_prior_disappr, common_proportion_prior = run_param['common_prior_appr_input'],run_param['common_prior_appr'],run_param['common_prior_disappr'],run_param['common_proportion_prior']
    normal_constr_w = run_param['normal_constr_w']
    common_prior_mean = common_prior_appr_input[0]/sum(common_prior_appr_input)
    state_evolution = []
    state_evolution_baseline = []
    both_baseline_and_extensive = True
    lst = []
    cols = ['run_id', 'time_step', 'opinion', 'out_belief', 'in_belief', 'group_type','tau','alpha','participation']
    lst_df = pd.DataFrame(lst, columns=cols)
    inst_type = 'extensive' if run_param['attr_dict']['extensive'] else 'intensive'
    institution = Institution(inst_type)
    if institution.type == 'extensive':
        institution.opt_signals = {'appr': run_param['extensive_optimal'], 'disappr': run_param['extensive_optimal']}
    else:
        '''TODO: Need to make the name consistent with the intensive case'''
        institution.opt_signals = {'appr': run_param['intensive_optimal'], 'disappr': run_param['intensive_optimal']}

    for group_type in ['control','treatment']:
        
        for batch_num in tqdm(np.arange(1, run_param['attr_dict']['num_batches']), desc='Batch Progress'):
            env = parallel_env(render_mode='human', attr_dict=run_param['attr_dict'])
            
            
            env.reset()
            env.NUM_ITERS = 100
            if batch_num == 1:
                mean_opinion_appr, mean_in_belief_appr, mean_out_belief_appr, participation_appr = env.generate_row_entry('appr',calc_participation=False)
                mean_opinion_disappr, mean_in_belief_disappr, mean_out_belief_disappr, participation_disappr = env.generate_row_entry('disappr',calc_participation=False)
                if group_type == 'control':
                    state_evolution_baseline.append([0,0,mean_opinion_appr,mean_out_belief_appr,mean_in_belief_appr,group_type,env.normal_constr_w,env.alpha,0.0])
                    state_evolution_baseline.append([0,0,mean_opinion_disappr,mean_out_belief_disappr,mean_in_belief_disappr,group_type,env.normal_constr_w,env.alpha,0.0])
                else:
                    state_evolution.append([0,0,mean_opinion_appr,mean_out_belief_appr,mean_in_belief_appr,group_type,env.normal_constr_w,env.alpha,0.0])
                    state_evolution.append([0,0,mean_opinion_disappr,mean_out_belief_disappr,mean_in_belief_disappr,group_type,env.normal_constr_w,env.alpha,0.0])
            for ts in np.arange(1, 100):
                                                               
                #print('Progress: batch_num:', batch_num, 'ts:', ts, 'out of 100')
                mean_common_prior_ingroup_var = np.mean([utils.beta_var(agent.common_prior_ingroup[0],agent.common_prior_ingroup[1]) for agent in env.possible_agents])
                mean_common_prior_outgroup_var = np.mean([utils.beta_var(agent.common_prior_outgroup[0],agent.common_prior_outgroup[1]) for agent in env.possible_agents])
                
                if max(mean_common_prior_ingroup_var,mean_common_prior_outgroup_var) < 0.001:
                    break
                appr_pos_for_ts,disappr_pos_for_ts, prop_for_ts = None, None, None
                for agent in env.possible_agents:
                    if math.isnan(agent.common_prior_outgroup[0]/np.sum(agent.common_prior_outgroup)) or math.isnan(agent.common_prior_ingroup[0]/np.sum(agent.common_prior_ingroup)):
                        raise Exception('Nan in common prior')
                    if group_type == 'treatment':
                        if env.homogenous_priors and appr_pos_for_ts is not None and disappr_pos_for_ts is not None:
                            ingroup_posterior = appr_pos_for_ts if agent.opinion[agent.norm_context] >= 0.5 else disappr_pos_for_ts
                            outgroup_posterior = disappr_pos_for_ts if agent.opinion[agent.norm_context] >= 0.5 else appr_pos_for_ts
                            agent.common_proportion_prior = prop_for_ts
                        else:
                            outgroup_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,institution,agent.common_proportion_prior,'outgroup')
                            ingroup_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,institution,agent.common_proportion_prior,'ingroup')
                        if env.homogenous_priors:
                            if appr_pos_for_ts is None:
                                appr_pos_for_ts = ingroup_posterior if agent.opinion[agent.norm_context] >= 0.5 else outgroup_posterior
                                prop_for_ts = agent.common_proportion_posterior
                            if disappr_pos_for_ts is None:
                                disappr_pos_for_ts = ingroup_posterior if agent.opinion[agent.norm_context] < 0.5 else outgroup_posterior

                        agent.pseudo_update_posteriors = {institution.type:{'outgroup':outgroup_posterior,'ingroup':ingroup_posterior}}
                    else:
                        agent.pseudo_update_posteriors = {institution.type:{'outgroup':agent.common_prior_outgroup,'ingroup':agent.common_prior_ingroup}}
                actions = {agent.id:agent.act(env,run_type={'institutions':{'extensive':institution if institution.type == 'extensive' else None,
                                                                            'intensive':institution if institution.type == 'intensive' else None},'update_type':'common'},baseline=False) for agent in env.possible_agents}
                observations, rewards, terminations, truncations, infos = env.step(actions,ts,baseline=False)
                mean_opinion_appr, mean_in_belief_appr, mean_out_belief_appr, participation_appr = env.generate_row_entry('appr')
                mean_opinion_disappr, mean_in_belief_disappr, mean_out_belief_disappr, participation_disappr = env.generate_row_entry('disappr')
                if group_type == 'treatment':
                    state_evolution.append([batch_num,ts,mean_opinion_appr,mean_out_belief_appr,mean_in_belief_appr,group_type,env.normal_constr_w,env.alpha,participation_appr])
                    state_evolution.append([batch_num,ts,mean_opinion_disappr,mean_out_belief_disappr,mean_in_belief_disappr,group_type,env.normal_constr_w,env.alpha,participation_disappr])
                else:
                    state_evolution_baseline.append([batch_num,ts,mean_opinion_appr,mean_out_belief_appr,mean_in_belief_appr,group_type,env.normal_constr_w,env.alpha,participation_appr])
                    state_evolution_baseline.append([batch_num,ts,mean_opinion_disappr,mean_out_belief_disappr,mean_in_belief_disappr,group_type,env.normal_constr_w,env.alpha,participation_disappr])
        
        df = pd.DataFrame(state_evolution if group_type=='treatment' else state_evolution_baseline, columns=cols)
        if run_param['credible']:
            file_path = 'data\\single_'+institution.type+'_'+str(group_type)+'_'+run_param['attr_dict']['distr_shape']+'.csv'
        else:
            file_path = 'data\\single_'+institution.type+'_'+str(group_type)+'_'+run_param['attr_dict']['distr_shape']+'_incredible.csv'
        if os.path.exists(file_path):
            # Append without header if file exists
            df.to_csv(file_path, mode='a', header=False, index=True)
        else:
            # Write with header if file does not exist
            df.to_csv(file_path, mode='w', header=True, index=True)
    print('Done')


    
    
def run_sim_multiple_institution(run_param):
    """ ENV SETUP """
    common_prior_appr_input, common_prior_appr, common_prior_disappr, common_proportion_prior = run_param['common_prior_appr_input'],run_param['common_prior_appr'],run_param['common_prior_disappr'],run_param['common_proportion_prior']
    normal_constr_w = run_param['normal_constr_w']
    common_prior_mean = common_prior_appr_input[0]/sum(common_prior_appr_input)
    state_evolution = []
    state_evolution_baseline = []
    both_baseline_and_extensive = True
    lst = []
    cols = ['run_id', 'time_step', 'opinion', 'out_belief', 'in_belief', 'group_type','tau','alpha','participation', 'listened_to']
    lst_df = pd.DataFrame(lst, columns=cols)
    
    institution_extensive = Institution('extensive')
    institution_intensive = Institution('intensive')

    institution_extensive.opt_signals = run_param['extensive_optimal']
    institution_intensive.opt_signals = run_param['intensive_optimal']
    
    for batch_num in tqdm(np.arange(1, run_param['attr_dict']['num_batches']), desc='Batch Progress', position=0, leave=True):
            env = parallel_env(render_mode='human', attr_dict=run_param['attr_dict'])
            env.posterior_prediction_model = run_param['posterior_prediction_model']
            env.rhetoric_estimation_model = run_param['rhetoric_estimation_model']
            env.reset()
            env.NUM_ITERS = 100
            if batch_num == 1:
                for ag in env.possible_agents:
                    state_evolution.append([0,0,ag.opinion[ag.norm_context],ag.common_posterior_outgroup[0]/np.sum(ag.common_posterior_outgroup),ag.common_posterior_ingroup[0]/np.sum(ag.common_posterior_ingroup),'treatment',env.normal_constr_w,env.alpha,-1,'none'])
            for ts in (np.arange(1, run_param['attr_dict']['num_timesteps'])):
                                                               
                #print('Progress: batch_num:', batch_num, 'ts:', ts, 'out of 100')
                mean_common_prior_ingroup_var = np.mean([utils.beta_var(agent.common_prior_ingroup[0],agent.common_prior_ingroup[1]) for agent in env.possible_agents])
                mean_common_prior_outgroup_var = np.mean([utils.beta_var(agent.common_prior_outgroup[0],agent.common_prior_outgroup[1]) for agent in env.possible_agents])
                
                if max(mean_common_prior_ingroup_var,mean_common_prior_outgroup_var) < 0.001:
                    break
                appr_pos_for_ts,disappr_pos_for_ts, prop_for_ts = None, None, None
                for agent in env.possible_agents:
                    if math.isnan(agent.common_prior_outgroup[0]/np.sum(agent.common_prior_outgroup)) or math.isnan(agent.common_prior_ingroup[0]/np.sum(agent.common_prior_ingroup)):
                        raise Exception('Nan in common prior')
                    if (agent.common_prior_outgroup[0]/np.sum(agent.common_prior_outgroup) - 0.5)*(agent.opinion[agent.norm_context]-0.5) > 0 \
                        or (agent.common_prior_ingroup[0]/np.sum(agent.common_prior_ingroup) - 0.5)*(agent.opinion[agent.norm_context]-0.5) < 0:
                        f=1
                    if env.homogenous_priors and appr_pos_for_ts is not None and disappr_pos_for_ts is not None:
                        ingroup_posterior = appr_pos_for_ts if agent.opinion[agent.norm_context] >= 0.5 else disappr_pos_for_ts
                        outgroup_posterior = disappr_pos_for_ts if agent.opinion[agent.norm_context] >= 0.5 else appr_pos_for_ts
                        agent.common_proportion_prior = prop_for_ts
                    else:
                        outgroup_posterior_extensive, agent.common_proportion_posterior = agent.generate_posteriors(env,institution_extensive,agent.common_proportion_prior,'outgroup')
                        ingroup_posterior_extensive, agent.common_proportion_posterior = agent.generate_posteriors(env,institution_extensive,agent.common_proportion_prior,'ingroup')

                        outgroup_posterior_intensive, agent.common_proportion_posterior = agent.generate_posteriors(env,institution_intensive,agent.common_proportion_prior,'outgroup')
                        ingroup_posterior_intensive, agent.common_proportion_posterior = agent.generate_posteriors(env,institution_intensive,agent.common_proportion_prior,'ingroup')

                    if env.homogenous_priors:
                        if appr_pos_for_ts is None:
                            appr_pos_for_ts = ingroup_posterior if agent.opinion[agent.norm_context] >= 0.5 else outgroup_posterior
                            prop_for_ts = agent.common_proportion_posterior
                        if disappr_pos_for_ts is None:
                            disappr_pos_for_ts = ingroup_posterior if agent.opinion[agent.norm_context] < 0.5 else outgroup_posterior

                    agent.pseudo_update_posteriors = {'extensive':{'outgroup':outgroup_posterior_extensive,'ingroup':ingroup_posterior_extensive},
                                                      'intensive':{'outgroup':outgroup_posterior_intensive,'ingroup':ingroup_posterior_intensive}}
                    
                actions = {agent.id:agent.act(env,run_type={'institutions':{'extensive':institution_extensive,'intensive':institution_intensive}},baseline=False) for agent in env.possible_agents}
                observations, rewards, terminations, truncations, infos = env.step(actions,ts,baseline=False)
                
                for ag in env.possible_agents:
                    try:
                        state_evolution.append([batch_num,ts,ag.opinion[ag.norm_context],ag.common_prior_outgroup[0]/np.sum(ag.common_prior_outgroup),ag.common_prior_ingroup[0]/np.sum(ag.common_prior_ingroup),'treatment',env.normal_constr_w,env.alpha,ag.action_code,ag.listened_to])
                    except IndexError:
                        f=1
                        raise
    df = pd.DataFrame(state_evolution, columns=cols)
    file_path = 'data\\multiple_homo='+str(run_param['attr_dict']['homogenous_priors'])+'_'+run_param['attr_dict']['distr_shape']+'.csv'
    df.to_csv(file_path, mode='w', header=True, index=True)
    print('Done') 
                

        
def old_run_sim_multiple_institution(run_param):
    """ ENV SETUP """
    common_prior, common_prior_ingroup, common_prior_outgroup = run_param['common_prior'],run_param['common_prior_ingroup'],run_param['common_prior_outgroup']
    common_proportion_prior = run_param['common_proportion_prior']
    normal_constr_w = run_param['normal_constr_w']
    common_prior_mean = common_prior[0]/sum(common_prior)
    state_evolution,state_evolution_baseline = dict(), dict()
    lst = []
    cols = ['run_id', 'time_step', 'listened', 'opinion', 'out_belief']
    lst_df = pd.DataFrame(lst, columns=cols)
    #for signal_distr_theta_idx, signal_distr_theta in enumerate([common_prior_mean-(normal_constr_w+0.05),common_prior_mean-(normal_constr_w-0.05),common_prior_mean+(normal_constr_w+0.05),common_prior_mean+(normal_constr_w-0.05)]):
    '''
        opt_signals acquired from running solving_tools.py separately
    '''
    opt_signals, opt_signals_ingroup, opt_signals_outgroup = run_param['opt_signals'], run_param['opt_signals_ingroup'], run_param['opt_signals_outgroup']
    for batch_num in np.arange(10):
        extensive_institution = Institution('extensive')
        intensive_institution = Institution('intensive')
        env = parallel_env(render_mode='human',attr_dict={'distr_params':{'mean_op_degree_apr':0.7,'mean_op_degree_disapr':0.4,'apr_weight':0.5,'SD':0.05},
                                                           'distr_shape':'U',
                                                            'extensive':False,
                                                            'common_prior' : common_prior,
                                                            'common_prior_ingroup' : common_prior_ingroup,
                                                            'common_prior_outgroup' : common_prior_outgroup,
                                                            'common_proportion_prior' : common_proportion_prior,
                                                            'common_prior_appr_input':run_param['common_prior_appr_input'],
                                                            'only_intensive':run_param['only_intensive']})
        ''' Check that every norm context has at least one agent '''
        if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
            raise Exception()
        env.reset()
        env.no_print = True
        env.NUM_ITERS = 100
        
        env.prior_baseline = env.common_prior
        env.prior_prop_baseline = common_proportion_prior
        env.normal_constr_w = normal_constr_w
        #env.constraining_distribution = utils.Gaussian_plateu_distribution(env.common_prior[0]/sum(env.common_prior),.01,.3)
        #env.constraining_distribution = utils.Gaussian_plateu_distribution(.3,.01,.3)
        dataset = []
        history = [[(ag.opinion[ag.norm_context],'intensive' if env.only_intensive else 'both',1,0,ag.common_posterior_outgroup[0]/np.sum(ag.common_posterior_outgroup)) for ag in env.possible_agents if ag.opinion[ag.norm_context]>=0.5]]
        #history = []
        '''
        plt.figure()
        plt.hist([ag.opinion[ag.norm_context] for ag in env.possible_agents])
        plt.show()
        '''
        for i in np.arange(100):
            mean_common_prior_var = np.mean([utils.beta_var(agent.common_prior[0],agent.common_prior[1]) for agent in env.possible_agents])
            mean_common_prior_ingroup_var = np.mean([utils.beta_var(agent.common_prior_ingroup[0],agent.common_prior_ingroup[1]) for agent in env.possible_agents])
            mean_common_prior_outgroup_var = np.mean([utils.beta_var(agent.common_prior_outgroup[0],agent.common_prior_outgroup[1]) for agent in env.possible_agents])
            
            if min(mean_common_prior_var,mean_common_prior_ingroup_var,mean_common_prior_outgroup_var) < 0.001:
                break
            #print(min(mean_common_prior_var,mean_common_prior_ingroup_var,mean_common_prior_outgroup_var))
            
            print(common_prior,batch_num,i)
            #curr_state = np.mean([agent.common_prior[0]/sum(agent.common_prior) for agent in env.possible_agents])
            #curr_state_ingroup = np.mean([agent.common_prior_ingroup[0]/sum(agent.common_prior_ingroup) for agent in env.possible_agents])
            #curr_state_outgroup = np.mean([agent.common_prior_outgroup[0]/sum(agent.common_prior_outgroup) for agent in env.possible_agents])
            #signal_distr_theta = curr_state - 0.3
            
            #signal_distr_theta = opt_signals[round(curr_state,1)]
            #signal_distr_theta_ingroup = opt_signals_ingroup[round(curr_state_ingroup,1)]
            #signal_distr_theta_outgroup = opt_signals_outgroup[round(curr_state_outgroup,1)]
            
            
            if i not in  state_evolution:
                state_evolution[i] = []
            state_evolution[i].append((env.common_prior[0]/sum(env.common_prior),env.mean_sanction))
            if i not in  state_evolution_baseline:
                state_evolution_baseline[i] = []
            state_evolution_baseline[i].append((env.prior_baseline[0]/sum(env.prior_baseline),env.mean_sanction_baseline))
            
            ''' break if the mean beliefs (common or any of ingroup and outgroup) variance is very low. Because then information is stable '''
            
            ''' act is based on the new posterior acting as prior '''
            for agent in env.possible_agents:
                if math.isnan(agent.common_prior[0]/np.sum(agent.common_prior)) or math.isnan(agent.common_prior_outgroup[0]/np.sum(agent.common_prior_outgroup)) or math.isnan(agent.common_prior_ingroup[0]/np.sum(agent.common_prior_ingroup)):
                    continue
                # Change this to generate for both institutions and reverse the signals for intensive institutions for disapp opinions
                agent.common_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,(extensive_institution, intensive_institution),agent.common_proportion_prior,'common')
                agent.common_posterior_ingroup, agent.common_proportion_posterior = agent.generate_posteriors(env,(extensive_institution, intensive_institution),agent.common_proportion_prior,'ingroup')
                agent.common_posterior_outgroup, agent.common_proportion_posterior = agent.generate_posteriors(env,(extensive_institution, intensive_institution),agent.common_proportion_prior,'outgroup')
            
                
            actions = {agent.id:agent.act(env,run_type='self-ref',baseline=False) for agent in env.possible_agents}
            '''
            plt.figure()
            plt.hist([ag.opinion[ag.norm_context] for ag in env.possible_agents if ag.action[0]!=-1])
            plt.show()
            '''
            ''' common prior is updated based on the action observations '''
            observations, rewards, terminations, truncations, infos = env.step(actions,i,baseline=False)
            history.append(infos)
            
            #actions = {agent.id:agent.act(env,run_type='self-ref',baseline=True) for agent in env.possible_agents}
            #env.step(actions,i,baseline=True)
        '''
        plt.figure()
        plt.plot([ag.common_prior_outgroup_init for ag in env.possible_agents if ag.opinion[ag.norm_context]>=0.5],[ag.common_posterior_outgroup for ag in env.possible_agents if ag.opinion[ag.norm_context]>=0.5],'.')
        plt.show()
        '''
        data = {'run_id':[batch_num]*len(history[0]), 'time_step':[1]*len(history[0]), 
                'listened': [d[1] for d in history[0]],
                'opinion': [d[0] for d in history[0]], 'out_belief': [d[4] for d in history[0]] }
        df = pd.DataFrame(data)
        data = df.dropna()
        lst_df = lst_df.append(data)
        
        data = {'run_id':[batch_num]*len(history[-1]), 'time_step':[len(history)+1]*len(history[-1]), 
                'listened': [d[1] for d in history[-1]],
                'opinion': [d[0] for d in history[-1]], 'out_belief': [d[4] for d in history[-1]] }
        df = pd.DataFrame(data)
        data = df.dropna()
        lst_df = lst_df.append(data)
        '''
        sns.lmplot(x="x", y="y", data=df, ci=95, hue='listened')  
        subset_data = data[data['listened'] == 'both']
        r, p = pearsonr(subset_data['x'], subset_data['y'])
        ax = plt.gca()
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),transform=ax.transAxes)
        
        data = {'x': [d[0] for d in history[-1]], 'y': [d[4] for d in history[-1]], 'listened': [d[1] for d in history[-1]]}
        df = pd.DataFrame(data)
        data = df.dropna()
        
        sns.lmplot(x="x", y="y", data=df, ci=95, hue='listened')  
        subset_data = data[data['listened'] == 'both']
        r, p = pearsonr(subset_data['x'], subset_data['y'])
        ax = plt.gca()
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),transform=ax.transAxes)
        
        plt.show()  
        
            
            #env.common_prior = (np.random.randint(low=1,high=4),np.random.randint(low=1,high=4))
        cols = ['run_id', 'time_step', 'listened', 'opinion', 'out_belief']
        only_baseline_plot = False
        
        
        if not only_baseline_plot:
            for k,v in state_evolution.items():
                for _v in v:
                    lst.append([k,_v[0],'signal',_v[1]])
            
        
        for k,v in state_evolution_baseline.items():
            for _v in v:
                lst.append([k,_v[0],'no signal',_v[1]])
        '''
    
    return lst_df

def multiple_inst_run(attr_dict=None,
                    run_param=None):
    
    run_sim_multiple_institution(run_param)
    
    
def single_inst_run(extensive_outgroup_optimal=None,
                    extensive_ingroup_optimal=None,
                    attr_dict=None,
                    run_param=None):
    intensive_outgroup_optimal = {'type':'disappr', 'opt_signals': {0.0: 0.0, 0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0}}
    if extensive_outgroup_optimal is None:
        extensive_outgroup_optimal = {0.2:{'type':'disappr', 'opt_signals': {0.0: 0.2, 0.1: 0.3, 0.2: 0.3, 0.3: 0.3, 0.4: 0.3}},
                                0.3:{'type':'disappr', 'opt_signals': {0.0: 0.0, 0.1: 0.4, 0.2: 0.5, 0.3: 0.5, 0.4: 0.5}},
                                0.1:{'type':'disappr', 'opt_signals': {0.0: 0.0, 0.1: 0.0, 0.2: 0.3, 0.3: 0.4, 0.4: 0.5}}}
    intensive_ingroup_optimal = {'type':'appr', 'opt_signals': {0.5: 0.6, 0.6: 0.6, 0.7: 0.6, 0.8: 0.6, 0.9: 0.8, 1.0: 0.8}}
    if extensive_ingroup_optimal is None:
        extensive_ingroup_optimal = {0.2:{'type':'appr', 'opt_signals': {0.5: 0.5, 0.6: 0.5, 0.7: 0.5, 0.8: 0.6, 0.9: 0.8, 1.0: 0.8}},
                                0.3:{'type':'appr', 'opt_signals': {0.5: 0.5, 0.6: 0.5, 0.7: 0.5, 0.8: 0.5, 0.9: 0.6, 1.0: 1.0}},
                                0.1:{'type':'appr', 'opt_signals': {0.5: 0.5, 0.6: 0.5, 0.7: 0.6, 0.8: 0.7, 0.9: 1.0, 1.0: 1.0}}
                                }
    extensive_optimal = None
    if run_param is None:
        run_param ={'common_prior_appr_input':(5,2),
                        'common_prior_appr':(5,2),
                        'common_prior_disappr':(2,5),
                        'common_proportion_prior':(5,5),
                        'normal_constr_w':0.3,
                        'only_intensive':False,
                        'credible':True}
    if attr_dict is None:
        institution = Institution('extensive')
        attr_dict = { 'distr_params':{'mean_op_degree_apr':0.55,'mean_op_degree_disapr':0.45,'apr_weight':0.5,'SD':0.05},
                                            'distr_shape':'uniform',
                                            'extensive': False if institution.type == 'intensive' else True,
                                            'common_prior_appr': run_param['common_prior_appr'],
                                            'common_prior_disappr': run_param['common_prior_disappr'],
                                            'common_proportion_prior': run_param['common_proportion_prior'],
                                            'common_prior_appr_input': run_param['common_prior_appr_input'],
                                            'only_intensive': False if institution.type == 'extensive' else True,
                                            'homogenous_priors': True,
                                            'num_players':500,
                                            'alpha':0.3,
                                            'outgroup_rhetoric_intensity':0.3,
                                            'normal_constr_w':run_param['normal_constr_w']
                                }
        run_param['attr_dict'] = attr_dict
    extensive_optimal = extensive_outgroup_optimal[run_param['normal_constr_w'] if run_param['credible'] else 0.3]['opt_signals']
    extensive_optimal.update(extensive_ingroup_optimal[run_param['normal_constr_w'] if run_param['credible'] else 0.3]['opt_signals'])
    extensive_optimal = {k:(v,v) for k,v in extensive_optimal.items()}
    int_type = 'extensive' if run_param['attr_dict']['extensive'] else 'intensive'
    run_param[int_type+'_optimal'] = extensive_optimal
    run_sim_single_institution(run_param)

if __name__ == "__main__":
    single_inst_run()
