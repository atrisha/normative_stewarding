'''
Created on 19 Aug 2022

@author: Atrisha
'''
import math
import os
import numpy as np
import math
from functools import reduce

alpha = None

beta = None

R = 1

discount_factor = 0.5

d = None

c = None

punisher_prop = 0.6

players_per_group = 100

num_players = 100
'''
def get_beta_discrete(inp_x,a,b):
    beta_discrete = {prop:beta.pdf(x=inp_x,a,b)/sum(beta.pdf(x=np.arange(0.1,1.1,.1),a,b)) for prop in np.arange(0.1,1.1,.1)}
    return beta_discrete
'''



# Paths
DATA_DIRECTORY = os.path.join(os.getcwd(), 'data')
POLICY_FILE = os.path.join(DATA_DIRECTORY, 'inst_opt_policy.json')
SAMPLING_RATIOS_FILE = os.path.join(DATA_DIRECTORY, 'inst_sampling_ratios.json')
RHETORIC_MODEL_FILE = os.path.join(os.getcwd(), 'pickles', 'rhet_eq_estimation.pkl')

# Common Priors
COMMON_PRIOR_APPR_INPUT = (5, 3)  # Beta distribution prior for approvers' input belief
COMMON_PRIOR_APPR = (5, 3)        # Beta distribution prior for approvers' belief
COMMON_PRIOR_DISAPPR = (3, 5)     # Beta distribution prior for disapprovers' belief
COMMON_PROPORTION_PRIOR = (5, 5)  # Beta distribution prior for the proportion of approvers

# Simulation Attributes
ATTR_DICT = {
    'distr_params': {
        'mean_op_degree_apr': 0.6,  # Mean opinion degree for approvers
        'mean_op_degree_disapr': 0.4,  # Mean opinion degree for disapprovers
        'apr_weight': 0.5,  # Weight assigned to approvers
        'SD': 0.2,  # Standard deviation for opinion degree
    },
    'distr_shape': 'U',  # Distribution shape for the simulation
    'extensive': False,  # Whether we are running participatory instiuations
    'common_prior_appr': COMMON_PRIOR_APPR,
    'common_prior_disappr': COMMON_PRIOR_DISAPPR,
    'common_proportion_prior': COMMON_PROPORTION_PRIOR,
    'common_prior_appr_input': COMMON_PRIOR_APPR_INPUT,
    'only_intensive': False,  # Whether to run only ideological institutions
    'homogenous_priors': True,  # Whether priors are homogenous
    'num_players': 100,  # Number of players in the simulation
    'alpha': 0.5,  # Moderation strictness parameter
    'tailored_alpha': False,  # Whether moderation is tailored. This is always False.
    'lambda_outgroup': 0.5,  # Influence factor for out-group effects
    'lambda_ingroup': 1.5,  # Influence factor for in-group effects
    'normal_constr_w': 0.2,  # Constraint weight for normal distribution
    'rhet_thresh_mean': 0.3,  # Threshold rhetoric at which agent stays silent
    'update_rate': 0.2,  # Update rate for opinion change
    'inst_update_rate': 0.2,  # Update rate for institution behavior
    'num_batches': 10,  # Number of batches for simulation
    'num_timesteps': 100,  # Number of timesteps in the simulation
    'print_log': False,  # Whether to print detailed logs
    'verbose': False,  # Whether to enable verbose output
    'show_plots': True,  # Whether to show plots
}

# Runtime Parameters
RUN_PARAM = {
    'common_prior_appr_input': COMMON_PRIOR_APPR_INPUT,
    'common_prior_appr': COMMON_PRIOR_APPR,
    'common_prior_disappr': COMMON_PRIOR_DISAPPR,
    'common_proportion_prior': COMMON_PROPORTION_PRIOR,
    'normal_constr_w': ATTR_DICT['normal_constr_w'],  # Signalling Constraint parameter
    'credible': True,  # Unused param
}


def calc_sum_util(util_val,d):
    scaled_disc = 1- (1-d)*(1-discount_factor)
    sum = 0
    iter = 1
    disc_factor_multiplier = scaled_disc
    while disc_factor_multiplier > math.pow(10,-5):
        sum += disc_factor_multiplier * util_val
        disc_factor_multiplier = disc_factor_multiplier*scaled_disc
    return sum



def calc_update_util(util_val,theta,d,c):
    scaled_disc = 1- (1-d)*(1-discount_factor)
    sum = util_val*scaled_disc
    disc_factor_multiplier = scaled_disc**2
    while disc_factor_multiplier > math.pow(10,-3):
        sum += disc_factor_multiplier * (0.5*(1-d)*(((2*theta-1)*R)-(2*c*theta)))
        disc_factor_multiplier = disc_factor_multiplier*scaled_disc
    return sum

cen_true_distr = None
cen_belief = None
minority_op_mode = None
op_mode = None
payoff_tol = None
risk_tol = None

def get_maj_opinion():
    return 'A' if sum(cen_true_distr[2:])/sum(cen_true_distr) >= 0.5 else 'D'


if __name__ == "__main__":
    print(os.getcwd())