'''
Created on 6 Oct 2022

@author: Atrisha
'''

import numpy as np
import itertools
from lp_utils import solve_lp_multivar

class CorrelatedEquilibria:
    
    def __init__(self,additional_constraints_callback=None,callback_args = None):
        self.additional_constraints_callback = additional_constraints_callback
        self.callback_args = callback_args
    
    def solve(self,payoff_dict,players):
        num_players = len(list(payoff_dict.keys())[0])
        player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
        all_indices = list(itertools.product(*[list(np.arange(len(x))) for x in player_actions]))
        all_vars = ['p'+''.join(list(x)) for x in payoff_dict.keys()]
        
        #all_agents = [p.id for p in players]
        all_agents =  players
        num_players = len(all_agents)
        high_constr_list = None
        high_bounds_transformed_payoffs = payoff_dict
            
        for idx,ag_id in enumerate(all_agents):
            act_tups = []
            replaced_p_a = list(player_actions) 
            replaced_p_a[idx] = [None]*len(replaced_p_a[idx])
            
            obj_strats = []
            for this_ag_strat in player_actions[idx]:
                other_agent_act_combinations = list(set(itertools.product(*[v for idx_2,v in enumerate(replaced_p_a)])))
                for idx_2,s in enumerate(other_agent_act_combinations):
                    _s = list(s)
                    _s[idx] = this_ag_strat
                    other_agent_act_combinations[idx_2] = tuple(_s)
                high_obj_utils = [high_bounds_transformed_payoffs[x][idx] for x in other_agent_act_combinations]
                var_vect = ['p'+''.join([str(y) for a_idx,y in enumerate(list(x))]) for x in other_agent_act_combinations]
                
                for this_ag_oth_strat in player_actions[idx]:
                    if this_ag_strat != this_ag_oth_strat:
                        other_agent_act_combinations = list(set(itertools.product(*[v for idx_2,v in enumerate(replaced_p_a)])))
                        for idx_2,s in enumerate(other_agent_act_combinations):
                            _s = list(s)
                            _s[idx] = this_ag_oth_strat
                            other_agent_act_combinations[idx_2] = tuple(_s)
                        constr_utils = [high_bounds_transformed_payoffs[x][idx] for x in other_agent_act_combinations]
                        constr_diff = [x1 - x2 for (x1, x2) in zip(high_obj_utils, constr_utils)]
                        if np.any(constr_diff):
                            if high_constr_list is None:
                                high_constr_list = [(var_vect,constr_diff)] 
                            else:
                                high_constr_list.append((var_vect,constr_diff))
                        
        solns_dict = dict()
        obj_vals = []
        for act_code in all_vars:
            strat_profile = tuple(list(act_code[1:]))
            ''' Utilitarian social welfare will just add the utilities '''
            val = sum(high_bounds_transformed_payoffs[strat_profile])
            obj_vals.append(val)
        if self.additional_constraints_callback is not None:
            high_constr_list = self.additional_constraints_callback((high_constr_list,self.callback_args))
        if high_constr_list is not None:
            solns = solve_lp_multivar(all_vars,obj_vals,high_constr_list) 
            solns_dict['high'] = solns
        print(solns)
        rec = solns.index(1)
        rec_strat_str = tuple(list(all_vars[rec][1:]))
        eq_dict =  {rec_strat_str:payoff_dict[rec_strat_str]}
        return eq_dict
                
class PureNashEquilibria():
    
    def solve(self,payoff_dict,all_eqs=True):
        num_players = len(list(payoff_dict.values())[0])
        eq = list(payoff_dict.keys())
        N = len(payoff_dict)
        ct = 0
        for i in np.arange(num_players):
            for k1,v1 in payoff_dict.items():
                #print(ct,'/',N)
                #ct += 1
                for k2,v2 in payoff_dict.items():
                    if k2==(124,27387,29316,31028) and k1 == (64,27387,28595,30446) and i==1:
                        brk=1
                        
                    ''' agent i's strategy changes'''
                    _v1 = v1[i]
                    _v2 = v2[i]
                    if v2[i] > v1[i]:
                        ''' all other agent's strategy remains same '''
                        oth_strategy_same = True
                        for j in np.arange(num_players):
                            if j!= i:
                                if k2[j] == k1[j]:
                                    oth_strategy_same = oth_strategy_same and True
                                else:
                                    oth_strategy_same = False
                                    break
                        ''' not an equilibrium '''
                        if k1 in eq and oth_strategy_same:
                            eq.remove(k1)
        eq_dict =  {k:payoff_dict[k] for k in eq}
        if not all_eqs:
            ''' if multiple nash equilibria found, then select the one that has highest cumulative payoff '''
            if len(eq_dict) > 1:
                max_sum = max([sum(v) for v in eq_dict.values()])
                for k,v in eq_dict.items():
                    if sum(v) == max_sum:
                        return {k:v}
            else:
                return eq_dict
        else:
            return eq_dict
'''
payoff_dict = {('s','s'):[2,2],
               ('s','h'):[-1,0],
               ('h','s'):[0,-1],
               ('h','h'):[1,1]}
agent_ids = [1,2]

ce = CorrelatedEquilibria()
res = ce.solve(payoff_dict, agent_ids)
print(res)     
  '''      