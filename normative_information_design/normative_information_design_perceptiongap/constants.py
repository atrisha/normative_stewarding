'''
Created on 19 Aug 2022

@author: Atrisha
'''
import math

import numpy as np

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

