'''
Created on 7 Sept 2022

@author: Atrisha
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import scipy.linalg
import scipy.stats as stats
from scipy.stats import beta
import itertools
from Equilibria import CorrelatedEquilibria, PureNashEquilibria
from collections import Counter
#import rpy2.robjects as robjects
#import rpy2.robjects.numpy2ri
#from rpy2.robjects.packages import importr
import warnings
#from astropy.stats import sigma_clipping
#from pymoo.util.ref_dirs import get_reference_directions
import functools
import operator
import scipy.special
from scipy.stats import dirichlet
import torch
import os
from sklearn.ensemble import RandomForestRegressor
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot_beta(a,b,ax=None,color=None,label=None,linestyle='-'):
    x = np.linspace(beta.ppf(0.01, a, b),beta.ppf(0.99, a, b), 100)
    plot_color = 'black' if color is None else color
    if ax is None:
        plt.figure(figsize=(7,7))
        plt.xlim(0, 1)
        plt.plot(x, beta.pdf(x, a, b), linestyle='-', color=plot_color)
        plt.title('Beta Distribution', fontsize='15')
        plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
        plt.ylabel('Probability', fontsize='15')
        #plt.show()
    else:
        ax.set_xlim(0, 1)
        ax.plot(x, beta.pdf(x, a, b), linestyle=linestyle if linestyle is not None else '-', color=plot_color,label=label)
        #ax.set_xlabel('Values of Random Variable X (0, 1)', fontsize='15')
        #ax.set_ylabel('Probability', fontsize='15')
        
    
def eq(a,b):
    return a==b 

def less(a,b):
    return a < b

def leq(a,b):
    return a <= b

def generate_2_x_2_staghunt(r_info,players,for_recommendation_ex_post):
    get_oth_opinion = lambda x : 'A' if x == 'D' else 'D'
    num_players = 2
    all_strats = list(itertools.product(*[['A','D'],['A','D']]))
    max_payoff = 1
    payoff_dict = dict()
    for pl_idx, player in enumerate(players):
        assert(player.opinion_val<=1)
        strat_player_opinion = ''.join([player.opinion]*num_players)
        strat_not_player_opinion = ''.join([get_oth_opinion(player.opinion)]*num_players)
        player.act_payoffs = {strat_player_opinion:player.opinion_val,strat_not_player_opinion:max_payoff-player.opinion_val}
        if for_recommendation_ex_post:
            minorty_op_check = True if player.opinion != r_info.maj_opinion else False
        else:
            minorty_op_check = True if player.opinion != player.oth_opinion_bel else False
        if minorty_op_check:
            ''' Player descriptive opinion belief in minority. So risk of sharing not opinion is less that sharing opinion '''
            r_player_opinion = np.random.uniform(max(0,player.opinion_val-(1-player.opinion_val)),max_payoff)
            r_not_player_opinion = np.random.uniform()*r_player_opinion
        else:
            ''' Player descriptive opinion belief in majority '''
            r_not_player_opinion = np.random.uniform(player.opinion_val-(1-player.opinion_val),max_payoff)
            r_player_opinion = np.random.uniform(player.opinion_val-(1-player.opinion_val),r_not_player_opinion)
        strat_player_opinion_defection = ''.join([player.opinion if i==pl_idx else get_oth_opinion(player.opinion) for i in np.arange(num_players)])
        strat_not_player_opinion_defection = ''.join([get_oth_opinion(player.opinion) if i==pl_idx else player.opinion for i in np.arange(num_players)])
        player.act_payoffs[strat_player_opinion_defection] = player.act_payoffs[strat_player_opinion] - r_player_opinion
        player.act_payoffs[strat_not_player_opinion_defection] = player.act_payoffs[strat_not_player_opinion] - r_not_player_opinion
    payoff_dict = {}
    for k in list(players[0].act_payoffs.keys()):
        payoff_dict[tuple(list(k))] = [players[i].act_payoffs[k] for i in np.arange(num_players)]
    #pne = PureNashEquilibria()
    #pne_res = pne.solve(payoff_dict)
    return payoff_dict

get_oth_opinion = lambda x : 'A' if x == 'D' else 'D'

def check_pdrd_constraint(pl_op,pl_bel,pl_idx,payoff_dict_inp):
    payoff_dict = {''.join(list(k)):v for k,v in payoff_dict_inp.items()}
    try:
        if pl_op != pl_bel:
            if pl_idx == 0:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx]) >= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx])
            else:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx]) >= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx])   
            #assert payoff_dict[''.join([pl_op,pl_op])][pl_idx] < payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx], str(payoff_dict)+'\n'+str([pl_op,pl_bel,pl_idx])
        else:
            if pl_idx == 0:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx]) <= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx])
            else:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx]) <= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx])
        assert payoff_dict[''.join([pl_op,pl_op])][pl_idx] >= payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx], str(payoff_dict[''.join([pl_op,pl_op])][pl_idx]) +  str(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx])
    except AssertionError:
        f=1
        raise
        
def get_rd_eq(payoff_dict, pl_idx, pl):
    pl_op = pl.opinion
    pl_not_op = get_oth_opinion(pl_op)
    risk_of_op_val = abs(payoff_dict[(pl_op,pl_op)][pl_idx] - payoff_dict[(pl_op,get_oth_opinion(pl_op))][pl_idx])
    risk_of_not_op_val = abs(payoff_dict[(pl_not_op,pl_not_op)][pl_idx] - payoff_dict[(pl_not_op,get_oth_opinion(pl_not_op))][pl_idx])
    if pl_idx == 0:
        if  risk_of_op_val > risk_of_not_op_val : \
                   return (pl_not_op,pl_not_op)
        else:
            return (pl_op,pl_op)
    else:
        if abs(payoff_dict[(pl_op,pl_op)][pl_idx] - payoff_dict[(get_oth_opinion(pl_op),pl_op)][pl_idx]) \
               > abs(payoff_dict[(pl_not_op,pl_not_op)][pl_idx] - payoff_dict[(get_oth_opinion(pl_not_op),pl_not_op)][pl_idx]): \
                   return (pl_not_op,pl_not_op)
        else:
            return (pl_op,pl_op)
def get_act_risk(payoff_dict, pl_idx, act):
    if pl_idx == 0:
        risk_of_act_val = abs(payoff_dict[(act,act)][pl_idx] - payoff_dict[(act,get_oth_opinion(act))][pl_idx])
    else:
        risk_of_act_val = abs(payoff_dict[(act,act)][pl_idx] - payoff_dict[(get_oth_opinion(act),act)][pl_idx])
    return risk_of_act_val

def get_pd_eq(payoff_dict, pl_idx, pl):
    pl_op = pl.opinion
    pl_not_op = get_oth_opinion(pl_op)
    return (pl_op,pl_op) if payoff_dict[(pl_op,pl_op)][pl_idx] > payoff_dict[(pl_not_op,pl_not_op)][pl_idx] else (pl_not_op,pl_not_op)
        
op_category_list = ['SD','D','A','SA']

def sample_op_val(dirichlet_parms,person_object,from_person_object=None):
    multi_nom_parms = np.random.dirichlet(alpha=dirichlet_parms)
    ''' The opinion categories stand for disstrong approval to strong approval'''
                   
    op_category_sample = np.random.choice(op_category_list,p=multi_nom_parms)
    appr_val_map = {'SD':[0,.25],'D':[0.25,0.5],'A':[0.5,0.75],'SA':[0.75,1]}
    appr_val = np.random.uniform(low=appr_val_map[op_category_sample][0], high=appr_val_map[op_category_sample][1])  
    if from_person_object is None:
        person_object.opinion_expanded = op_category_sample
        person_object.opinion = 'A' if appr_val > 0.5 else 'D' if appr_val < 0.5 else np.random.choice(['A','D'])
        person_object.appr_val = appr_val
        person_object.opinion_val = appr_val if person_object.opinion == 'A' else 1-appr_val
    else:
        person_object.opinion_expanded = from_person_object.opinion_expanded
        person_object.opinion = from_person_object.opinion
        person_object.appr_val = from_person_object.appr_val
        person_object.opinion_val = from_person_object.opinion_val
    return appr_val

def get_expanded_action_freq(players_list):
    players_expanded_ops = [pl.action for pl in players_list if pl.action != 'N']
    freq_vals_counter = Counter(players_expanded_ops)
    return [freq_vals_counter['SD'],freq_vals_counter['D'],freq_vals_counter['A'],freq_vals_counter['SA']]
               

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
def correlation_constraints(ref_cor_index,opinion_param_samples):
    N = opinion_param_samples.shape[0]
    corr_matrix_upp_bound = np.full(shape=(N,N),fill_value=-np.inf) if ref_cor_index[1] is None else np.full(shape=(N,N,2),fill_value=-np.inf)
    mu_s = opinion_param_samples.tolist()
    maj_relation_matrix = np.zeros(shape=(N,N))
    it = np.nditer(maj_relation_matrix, flags=['multi_index'])
    for x in it:
        idx = it.multi_index
        if (opinion_param_samples[idx[0]]-0.5)*(opinion_param_samples[idx[1]]-0.5) < 0:
            ''' one is in approval and other in diapproval '''
            maj_relation_matrix[idx[0],idx[1]] = 0
        else:
            maj_relation_matrix[idx[0],idx[1]] = 1
    assert (maj_relation_matrix==maj_relation_matrix.T).all() , 'matrix is not symmetric. Something is wrong'
    psi_i = lambda mu : np.sqrt(mu/(1-mu)) 
    for i in np.arange(N):
        for j in np.arange(N):
            if i > j:
                r_ij = [max([-psi_i(mu_s[i])*psi_i(mu_s[j]),-1/(psi_i(mu_s[i])*psi_i(mu_s[j]))]),min([psi_i(mu_s[i])/psi_i(mu_s[j]),psi_i(mu_s[j])/psi_i(mu_s[i])])]
                if ref_cor_index[1] is None:
                    corr_matrix_upp_bound[i,j] = np.random.uniform(0,r_ij[1]) if maj_relation_matrix[i,j]==1 else np.random.uniform(r_ij[0],0)
                else:
                    corr_matrix_upp_bound[i,j] = np.array([r_ij[0],r_ij[1]])
            if i==j:
                corr_matrix_upp_bound[i,j] = 1 if ref_cor_index[1] is None else np.array([1,1])
    for i in np.arange(N):
        for j in np.arange(N):
            if np.any(corr_matrix_upp_bound[i,j] == -np.inf):
                corr_matrix_upp_bound[i,j] = corr_matrix_upp_bound[j,i] if ref_cor_index[1] is None else corr_matrix_upp_bound[j,i,:]
    corr_sums = np.sum(corr_matrix_upp_bound,axis=0)
    return corr_matrix_upp_bound
    
def plot_gaussian(mu,variance,ax=None,vert_line_at = None):
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    if ax is None:
        plt.fill(x, stats.norm.pdf(x, mu, sigma))
    else:
        ax.fill(x, stats.norm.pdf(x, mu, sigma))
    if vert_line_at is not None:
        y_lim = plt.gca().get_ylim()
        if ax is None:
            plt.plot([vert_line_at]*2,[0,y_lim[-1]],'--')
        else:
            ax.plot([vert_line_at]*2,[0,y_lim[-1]],'--')
    #plt.show()

def list_to_str(l):
    return [str(x) for x in l]

def corr2cov(corr_mat,variance_list):
    cov_matrix = np.ones(shape=corr_mat.shape)
    var_matrix = np.zeros(shape=corr_mat.shape)
    np.fill_diagonal(var_matrix, variance_list)
    cov_matrix = var_matrix @ corr_mat @ var_matrix
    return cov_matrix

def generate_n_way_samples_copula(beta_params, corr_matrix, n_samples):
    beta_var = [stats.beta(a=beta_params[i][0],b=beta_params[i][1]).var() for i in np.arange(len(beta_params))]
    N = corr_matrix.shape[0]
    ''' we need to scale the covariance since the original covariance is with respect to beta distribution'''
    cov_matrix  = corr2cov(corr_matrix,beta_var)*(10**3)
    try:
        mvnorm = stats.multivariate_normal(mean=[0]*N, cov=cov_matrix)
    except ValueError:
        print(cov_matrix)
        ''' Positive semi definite check fail '''
        ''' https://math.stackexchange.com/questions/332456/how-to-make-a-matrix-positive-semidefinite '''
        cov_abs = np.absolute(cov_matrix)
        sum_non_diag = np.sum(cov_abs) - np.trace(cov_abs)
        min_diag = np.min(np.diagonal(cov_matrix))
        if min_diag > sum_non_diag:
            ''' Some other problem, raise'''
            raise
        else:
            _diff = (sum_non_diag-min_diag)+np.finfo(np.float32).eps
            np.fill_diagonal(cov_matrix, cov_matrix.diagonal() + _diff)
            if not np.all(np.linalg.eigvals(cov_matrix) > 0):
                raise
            mvnorm = stats.multivariate_normal(mean=[0]*N, cov=cov_matrix)
    x = mvnorm.rvs(n_samples)
    norm = stats.norm()
    x_unif = norm.cdf(x)
    m_list = [stats.beta(a=x[0],b=x[1]) for x in beta_params]
    x_trans = [m_list[i].ppf(x_unif[:,i]) for i in np.arange(len(m_list))]
    x_trans = np.asarray(x_trans).T
    marginals = np.mean(x_trans,axis=0)
    return x_trans
    
def generate_samples_copula(beta_params,ref_cor_index, n_samples):
    opinion_param_means = np.array([stats.beta(a=beta_params[i][0],b=beta_params[i][1]).mean() for i in np.arange(len(beta_params))])
    corr_mat = correlation_constraints(ref_cor_index,opinion_param_means)
    samples = generate_n_way_samples_copula(beta_params, corr_mat, n_samples)
    samples = np.where(samples < 0.5, 0, 1)
    return samples,corr_mat,None
    

def generate_samples(op_marginals,ref_cor_index):
    success = False
    while(not success):
        success = True
        base = importr('base')
        mipfp = importr('mipfp')
        questionr = importr('questionr')
        effectsize = importr('effectsize')
        infotheo = importr('infotheo')
        
        robjects.r('''
            # create a function `f`
                    f <- function(or,p){
              #or <- Corr2Odds(or,marg.probs=p)$odds
              rownames(or) <- colnames(or) <- c("n1", "n2", "n3", "n4")
              # estimating the joint-distribution
              p.joint <- ObtainMultBinaryDist(corr = or, marg.probs = p)
              samples <- RMultBinary(n = 100, mult.bin.dist = p.joint)$binary.sequences
              samples <- data.frame(samples)
              return(samples)
            }
            
            ''')
        robjects.r('''
            # create a function `f`
                    f_mutual_info <- function(samples){
                    mutual_info_mat <- mutinformation(samples)
              return(mutual_info_mat)
            }
            
            ''')
        rpy2.robjects.numpy2ri.activate()
        r_f = robjects.globalenv['f']
        #neg_cor_index = np.random.choice(np.arange(0,4))
        #opinion_param_samples = np.random.uniform(low=0.2,high=0.9,size=4)
        opinion_param_samples = np.array(op_marginals)
        corr_mat = correlation_constraints(ref_cor_index,opinion_param_samples)
        nr,nc = corr_mat.shape
        Br = robjects.r.matrix(corr_mat, nrow=nr, ncol=nc)
        robjects.r.assign("C", Br)
        try:
            p = robjects.FloatVector(opinion_param_samples)
            res = r_f(Br,p)
            output = np.array([list([res[j][i] for j in range(res.ncol)]) for i in range(res.nrow)])
            
            robjects.r.assign("samples", res)
            robjects.r('mi_mat <- mutinformation(samples)')
            '''
            samples_r = robjects.r.matrix(output, nrow=output.shape[0], ncol=output.shape[1])
            r_f_mi = robjects.globalenv['f_mutual_info']
            print(samples_r)
            '''
            
            mi_matrix = robjects.r['mi_mat']
            mi_matrix = np.asarray(a=mi_matrix,dtype=np.float16)
        except Exception:
            print('ran with',corr_mat,opinion_param_samples)
            success = False
    return output,corr_mat,mi_matrix

def generate_corr_mat_grid(op_marginals,ref_cor_index):
    opinion_param_samples = np.array(op_marginals)
    corr_mat = correlation_constraints(ref_cor_index,opinion_param_samples)
    return corr_mat

def generate_grid_samples(corr_mat,op_marginals,ref_cor_index):
    success = False
    while(not success):
        success = True
        base = importr('base')
        mipfp = importr('mipfp')
        questionr = importr('questionr')
        effectsize = importr('effectsize')
        infotheo = importr('infotheo')
        
        robjects.r('''
            # create a function `f`
                    f <- function(or,p){
              #or <- Corr2Odds(or,marg.probs=p)$odds
              rownames(or) <- colnames(or) <- c("n1", "n2", "n3", "n4")
              # estimating the joint-distribution
              p.joint <- ObtainMultBinaryDist(corr = or, marg.probs = p)
              samples <- RMultBinary(n = 100, mult.bin.dist = p.joint)$binary.sequences
              samples <- data.frame(samples)
              return(samples)
            }
            
            ''')
        robjects.r('''
            # create a function `f`
                    f_mutual_info <- function(samples){
                    mutual_info_mat <- mutinformation(samples)
              return(mutual_info_mat)
            }
            
            ''')
        rpy2.robjects.numpy2ri.activate()
        r_f = robjects.globalenv['f']
        if ref_cor_index[1] is not None:
            extracted_corr_mat = np.full(shape=(4,4),fill_value=-np.inf)
            for i in np.arange(extracted_corr_mat.shape[0]):
                for j in np.arange(extracted_corr_mat.shape[1]):
                    if i==ref_cor_index[0] or j==ref_cor_index[0]:
                        extracted_corr_mat[i,j] = np.linspace(corr_mat[i,j,0],corr_mat[i,j,1],5)[ref_cor_index[1]]
                    else:
                        extracted_corr_mat[i,j] = corr_mat[i,j,1]
            corr_mat = extracted_corr_mat
        opinion_param_samples = np.array(op_marginals)
        nr,nc = corr_mat.shape
        Br = robjects.r.matrix(corr_mat, nrow=nr, ncol=nc)
        robjects.r.assign("C", Br)
        try:
            p = robjects.FloatVector(opinion_param_samples)
            res = r_f(Br,p)
            output = np.array([list([res[j][i] for j in range(res.ncol)]) for i in range(res.nrow)])
            
            robjects.r.assign("samples", res)
            robjects.r('mi_mat <- mutinformation(samples)')
            '''
            samples_r = robjects.r.matrix(output, nrow=output.shape[0], ncol=output.shape[1])
            r_f_mi = robjects.globalenv['f_mutual_info']
            print(samples_r)
            '''
            
            mi_matrix = robjects.r['mi_mat']
            mi_matrix = np.asarray(a=mi_matrix,dtype=np.float16)
        except Exception:
            print('ran with',corr_mat,opinion_param_samples)
            success = False
    marginals = np.sum(output,axis=0)/100
    return output,corr_mat,mi_matrix

def get_mom_update(sample_mean,sample_var):
    mom_alpha_est = lambda u,v : u * (((u*(1-u))/v) - 1)
    mom_beta_est = lambda u,v : (1-u) * (((u*(1-u))/v) - 1)
    return np.array([mom_alpha_est(sample_mean,sample_var), mom_beta_est(sample_mean,sample_var)]).reshape((1,2))

def em(xs, thetas, max_iter=1000, tol=1e-6):
    """Expectation-maximization for coin sample problem."""
    ll_old = -np.infty
    thetas = np.where(thetas < 10**-6,thetas+(10**-6),thetas)
    thetas = np.where(thetas > (1-10**-6),thetas-(10**-6),thetas)
    #warnings.simplefilter('error',RuntimeWarning)
    for i in range(max_iter):
        try:
            ll = np.array([np.sum(xs * np.log(theta), axis=1) for theta in thetas])
            lik = np.exp(ll)
            ws = lik/lik.sum(0)
            exps = np.array([w[:, None] * xs for w in ws])
            thetas = np.array([expr.sum(0)/expr.sum() for expr in exps])
            thetas = np.where(thetas < 10**-6,thetas+(10**-6),thetas)
            thetas = np.where(thetas > (1-10**-6),thetas-(10**-6),thetas)
            ll_new = np.sum([w*l for w, l in zip(ws, ll)])
            if np.abs(ll_new - ll_old) < tol:
                break
            ll_old = ll_new
        except RuntimeWarning:
            f=1
            raise
    return i, thetas, ws

def mle(xs):
    xs = list(xs)
    updated_thetas = dict()
    updated_props = Counter([x[1] for x in xs])
    updated_props = {k:v/len(xs) for k,v in updated_props.items()}
    for x in xs:
        if x[1] not in updated_thetas:
            updated_thetas[x[1]] = []
        updated_thetas[x[1]].append(x[0][0]/10)
    updated_thetas = {k:np.mean(v) for k,v in updated_thetas.items()}
    return updated_thetas,updated_props

def get_priors_from_true(true_distr):
    ''' Convert true means to beta params '''
    alphas = [round(x*10) for x in true_distr]
    betas = [10-x for x in alphas]
    try:
        priors = [np.random.beta(x[0],x[1]) if (x[0]>0 and x[1]>0) else np.random.beta(1,10) if x[0] <= 0 else np.random.beta(10,1) for x in zip(alphas,betas)]
    except ValueError:
        print('ran with',true_distr)
        raise
    return priors

def generate_correleted_opinions(marginal_params,correlation_val,size):
    mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., correlation_val], 
                                                     [correlation_val, 1.]])
    x = mvnorm.rvs(size)
    norm = stats.norm()
    x_unif = norm.cdf(x)
    m1 = stats.beta(a=marginal_params[0][0], b=marginal_params[0][1])
    m2 = stats.beta(a=marginal_params[1][0], b=marginal_params[1][1])
    x1_trans = m1.ppf(x_unif[:, 0])
    x2_trans = m2.ppf(x_unif[:, 1])
    samples = np.column_stack([x1_trans,x2_trans])
    return samples

def est_beta_from_mu_sigma(mu, variance, update_rate=None):
    if abs(mu-1) < 10**-6:
        return (1,0)
    elif mu < 10**-6:
        return (0,1)
    else:
    # Ensure mu is within the valid range (0, 1)
        mu = max(min(mu, 1 - 10**-6), 10**-6)
        # Avoid division by zero or negative variance
        variance = max(variance, 10**-6)
        
        # Calculate alpha using the corrected formula
        alpha = (mu ** 2 * ((1 - mu) / variance - 1 / mu))
        beta = alpha * (1 / mu - 1)
        
        # Optional check for calculated mu
        mu_check = alpha / (alpha + beta)
        if np.isnan(mu_check) or np.isnan(mu):
            f=1
            
        assert (mu-0.5)*(mu_check-0.5) >= 0, "mu_check failed: mu = {}, mu_check = {}".format(mu, mu_check)
        _param_min = np.min((alpha, beta))
        if _param_min < 1:
            mu_cross_100 = mu*100
            _alpha,_beta = np.clip(mu_cross_100,10**-3,100-10**-6),np.clip(100-mu_cross_100,10**-3,100-10**-3)
            alpha = _alpha/min(_alpha,_beta)
            beta = _beta/min(_alpha,_beta)
            assert (mu-0.5)*(mu_check-0.5) >= 0, "mu_check failed: mu = {}, mu_check = {}, alpha ={}, beta = {}".format(mu, mu_check, alpha, beta)
        if update_rate is not None:
            alpha = alpha/(alpha+beta) * update_rate
            beta = update_rate - alpha
        return (alpha, beta)


class Gaussian_plateu_distribution():
    ''' https://stats.stackexchange.com/a/203756 '''
    def __init__(self,mu,sigma,w):
        self.mu = mu
        self.sigma =sigma
        self.w = w
        self.root_2_pi_sigma = math.sqrt(2 * math.pi * self.sigma)
        self.h = 1 / (1 + (self.w / self.root_2_pi_sigma))
    
    def pdf(self,x):
        root_2_pi_sigma = math.sqrt(2*math.pi*self.sigma)
        h = 1/(1+(self.w/root_2_pi_sigma))   
        exponent = lambda x,side : math.exp((-1/(2*self.sigma**2))*(x-self.mu+(self.w/2))**2) if side=='l' else math.exp((-1/(2*self.sigma**2))*(x-self.mu-(self.w/2))**2)
        if isinstance(x, np.ndarray):
            results = np.zeros_like(x)  # Initialize result array
            # Calculate for left Gaussian tail
            left_mask = x <= self.mu - (self.w / 2)
            results[left_mask] = (self.h / self.root_2_pi_sigma) * np.exp(-0.5 * ((x[left_mask] - self.mu + self.w / 2) / self.sigma) ** 2)
            
            # Calculate for right Gaussian tail
            right_mask = x >= self.mu + (self.w / 2)
            results[right_mask] = (self.h / self.root_2_pi_sigma) * np.exp(-0.5 * ((x[right_mask] - self.mu - self.w / 2) / self.sigma) ** 2)
            
            # Calculate for the plateau region
            plateau_mask = ~left_mask & ~right_mask
            results[plateau_mask] = self.h / self.root_2_pi_sigma
            
            return results
        else:
            if x <= self.mu-(self.w/2):
                return (h/root_2_pi_sigma)*exponent(x,'l')
            elif x >= self.mu+(self.w/2):
                return (h/root_2_pi_sigma)*exponent(x,'r')
            else:
                return h/root_2_pi_sigma
        
    def _generate_gaussian_plateau_samples(self, n_samples):
        """
        Generate samples from a Gaussian plateau distribution.
        
        Parameters:
        - mu: mean of the Gaussian parts.
        - sigma: standard deviation of the Gaussian parts.
        - w: width of the plateau.
        - n_samples: total number of samples to generate.
        """
        mu, sigma, w = self.mu, self.sigma, self.w
        # Determine the number of samples for each region
        n_plateau = int(n_samples * (w / (w + 2 * sigma)))  # Approximation
        n_gaussian = n_samples - n_plateau
        
        # Generate samples for the plateau region
        plateau_samples = np.random.uniform(low=mu - w/2, high=mu + w/2, size=n_plateau)
        
        # Generate samples for the Gaussian tails
        # Note: This simplification generates all Gaussian samples without distinguishing between left and right
        gaussian_samples = np.random.normal(loc=mu, scale=sigma, size=n_gaussian)
        
        # Combine the samples
        all_samples = np.concatenate([plateau_samples, gaussian_samples])
        
        return all_samples
    
    def var(self):
            
        n_samples = 10000  # Number of samples to generate
        
        # Generate samples
        samples = self._generate_gaussian_plateau_samples(n_samples)
        return np.var(samples)


def generate_simplex_points(n_samples,n_dim):
    ''' https://www.egr.msu.edu/~kdeb/papers/c2020002.pdf '''
    ref_dirs = get_reference_directions("energy", n_dim, n_samples, seed=1)
    #Scatter().add(ref_dirs).show()
    return ref_dirs

def generate_unit_lattice(n_samples,n_dim):
    ''' https://stackoverflow.com/questions/12864445/how-to-convert-the-output-of-meshgrid-to-the-corresponding-array-of-points '''
    mesh_data = np.meshgrid(*tuple([np.linspace(0,1,n_samples) for n in np.arange(n_dim)]))
    lattice_points = np.vstack(map(np.ravel, mesh_data)).T
    #plt.plot(mesh_data[0], mesh_data[1], marker='o', color='k', linestyle='none')
    #plt.show()
    return lattice_points

def beta_pdf(x,a,b,domain_vals):
    beta_func = [(v**(a-1))*((1-v)**(b-1)) for v in domain_vals]
    beta_func = np.sum(beta_func)
    return ((x**(a-1))*((1-x)**(b-1)))/beta_func

def beta_var(a,b):
    return (a*b) / ( ((a+b)**2) * (a+b+1) )

def beta_mean(params):
    a,b = params
    return a/(a+b)

def distributionalize(priors,posterior_float):
    if not isinstance(posterior_float, tuple):
        _var = beta_var(priors[0], priors[1])
        return est_beta_from_mu_sigma(posterior_float, _var)
    else:
        return posterior_float

def dirichlet_pdf(x, alpha):
    
    return dirichlet.pdf(x, alpha)
    '''
    return (math.gamma(sum(alpha)) / 
          functools.reduce(operator.mul, [math.gamma(a) for a in alpha]) *
          functools.reduce(operator.mul, [x[i]**(alpha[i]-1.0) for i in range(len(alpha))]))
    '''
    
def runif_in_simplex(n_samples,n_dim):
    ''' Return uniformly random vector in the n-simplex '''

    k = np.random.exponential(scale=1.0, size=(n_samples,n_dim))
    return k / np.sum(k,axis=1)[:,None]

def get_target_expected_reward(model,state):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rewards = []
    actions = torch.rand(size=(1000,1),dtype=torch.float32, device=device)
    state_repeat = state.repeat(1000,1)
    input_as_batch = torch.cat((state_repeat,actions),axis=1).unsqueeze(0)
    out = model.forward(input_as_batch)
    max_reward = torch.max(out)
    act_index = torch.argmax(out) 
    argmax_act = actions[act_index]
    return max_reward, argmax_act

def approx_index(ndarr,val,tol):
    x = abs(ndarr - val) < tol
    indx = np.argmax(x)
    return indx


def posterior_estimator(common_prior,signal_distribution,update_group,constr_w):
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
    if update_group == 'appr':
        posterior_space = [x for x in np.linspace(0.01, 0.99, 50) if x >= 0.5]
    else:
        posterior_space = [x for x in np.linspace(0.01, 0.99, 50) if x < 0.5]
    
    priors_rescaled, likelihood_rescaled = dict(), dict()
    for x in np.linspace(0.01,0.99,50):
        priors_rescaled[x] = beta_pdf(x, common_prior[0], common_prior[1],np.linspace(0.01,0.99,50))
        _constr_distr = Gaussian_plateu_distribution(signal_distribution,.01,constr_w)
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
    return exp_x, var_x


def generate_posterior_prediction_model(group_type, constr_w):
    # Step 1: Data Generation
    # You'll need to replace this with your actual function calls
    X = []
    y = []
    signal_space = np.linspace(0, 0.49, 50) if group_type == 'disappr' else np.linspace(0.5, 1, 50)
    for a in np.linspace(0.1, 100, 50):  # Reduced for simplicity
        print(a)
        for b in np.linspace(0.1, 100, 50):
            for s in signal_space:
                exp_x, var_x = posterior_estimator((a, b), s, group_type,constr_w)
                X.append([a, b, s])
                y.append([exp_x, var_x])

    # Step 2 & 3: Model Selection and Training
    model = RandomForestRegressor()
    model.fit(X, y)
    pickle.dump(model, open('approximator_'+group_type+'_'+str(constr_w).replace('.','-')+'.pkl','wb'))

def predict_posterior(model_in, a, b, s):
    input_mean = beta_mean((a,b))
    resp = model_in.predict([[a, b, s]])
    group_type = 'appr' if input_mean >= 0.5 else 'disappr'
    pred_mu,pred_var = resp[0][0],resp[0][1]
    if (pred_mu-input_mean)*(s-input_mean) < 0:
        shift_val = abs(pred_mu-input_mean)
        pred_mu = input_mean + shift_val if s > input_mean else input_mean - shift_val
    pred_mu = min(1,max(0.5,pred_mu)) if group_type == 'appr' else min(0.5,max(0,pred_mu))
    return pred_mu,pred_var

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def generate_rhetoric_equilibrium_estimation_model(run_param):
    import numpy as np
    import itertools
    from scipy.optimize import minimize_scalar
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import math
    from tqdm import tqdm
    import random

    # Defining the function
    def equation(x, n, o, a, lambda_in, lambda_out):
        return min(1,((n * o * lambda_in * (1 - x)) / (a-(1-n)*(o*lambda_out**x)) )**(1 / x))

    # Function to find the max x where the curve crosses the y=x line
    def find_max_x_intersection(params):
        def local_equation(x):
            return equation(x, *params)
        samples = np.linspace(0.01, 1, 10)
        try:
            y_vals = np.asarray([(x,1) if equation(x, *params) > x else (x,-1) for x in samples])
        except TypeError:
            print(params)
            raise
        result = np.max(y_vals[:,1])*np.min(y_vals[:,1])
        if result > 0:
            return 0
        else:
            sign_change_index = np.where(np.diff(np.sign(y_vals[:,1])))[0][0]
            r_eq = np.mean(y_vals[sign_change_index:sign_change_index+2,0]) if sign_change_index<y_vals.shape[0]-2 else y_vals[sign_change_index,0]
            return r_eq
           
        
        
    
    # Sampling parameter ranges
    prop_samples = np.arange(0.5,1,0.01)  # Samples for h
    opinion_samples = np.arange(0.5,1,0.01)  # Samples for o
    alpha_samples = np.arange(0.1,1,0.1)              # Fixed values for a
    lambda_ingroup_samples = [run_param['attr_dict']['lambda_ingroup']]
    lambda_outgroup_samples = [run_param['attr_dict']['lambda_outgroup']]
    # Generating all combinations of parameters
    parameter_combinations = list(itertools.product(prop_samples, opinion_samples, alpha_samples, lambda_ingroup_samples, lambda_outgroup_samples))
    print(len(parameter_combinations))
    parameter_combinations = random.sample(parameter_combinations, 1000)
    # Calculating the maximum x for each parameter combination
    # max_x_values = [find_max_x_intersection(params) for params in parameter_combinations]
    filtered_parameter_combinations, max_x_values_filtered = [], []
    
    
    for _params in tqdm(parameter_combinations):
        params = list(_params)
        #params.insert(3, (1-params[-1]))
        filtered_parameter_combinations.append(params)
        max_x_values_filtered.append(find_max_x_intersection(params))
    
    #filtered_parameter_combinations = [x for x in parameter_combinations]
    #max_x_values_filtered = [find_max_x_intersection(params) for params in parameter_combinations]
    # Preparing the dataset for the regression model
    X = np.array(filtered_parameter_combinations)
    y = np.array(max_x_values_filtered)


    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Creating and training the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting and evaluating the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Outputting the model's performance
    print(f"MSE: {mse}")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    pickle.dump(model, open(os.path.join(os.getcwd(),'pickles','rhet_eq_estimation.pkl'),'wb'))
    return model

#generate_posterior_prediction_model('appr',0.1)
#generate_posterior_prediction_model('disappr',0.1)

def test_rhetoric_equilibrium_estimation():
    n_fixed = 0.5
    a_fixed = 0.6
    lamb_in_fixed = 1.5

    model = generate_rhetoric_equilibrium_estimation_model({'attr_dict': {'lambda_ingroup': lamb_in_fixed}})

    o_values_plot = np.linspace(0.5, 1, 100)
    input_data = np.array([[n_fixed, o, a_fixed, lamb_in_fixed] for o in o_values_plot])

    file_path = os.path.join(os.getcwd(),'pickles','rhet_eq_estimation.pkl')
    model = pickle.load(open(file_path,'rb'))
    predicted_values = model.predict(input_data)

    plt.figure(figsize=(10, 6))
    plt.plot(o_values_plot, predicted_values, label='Predicted Values', color='blue')
    plt.xlabel('o values')
    plt.ylabel('Predicted x values')
    plt.title('Predicted x values vs. o for fixed h, a, n, j')
    plt.legend()
    plt.grid(True)
    plt.show()

#test_rhetoric_equilibrium_estimation()
'''
# Step 5: Use Model as Estimator
def approximate_estimator(model_in, a, b, s):
    return model_in.predict([[a, b, s]])
group_type = 'disappr'
constr_w = 0.3
generate_posterior_prediction_model(group_type, constr_w)
model_in = pickle.load(open('approximator_'+group_type+'_'+str(constr_w).replace('.','-')+'.pkl','rb'))
signal_space = np.linspace(0, 0.49, 10) if group_type == 'disappr' else np.linspace(0.5, 1, 10)
# Example usage
err_m,err_v = [],[]
for iter in np.arange(1):
    #a,b = np.random.uniform(0,100), np.random.uniform(0,100)
    a,b = 3,4
    for s in [0.5]:
        approximated_output = approximate_estimator(model_in,a,b,s)
        tv = posterior_estimator((a,b), s, group_type, constr_w)
        print(approximated_output[0][0],tv[0])
        err_m.append(abs(approximated_output[0][0]-tv[0]))
        err_v.append(abs(approximated_output[0][1]-tv[1]))

print(np.mean(err_m),np.mean(err_v))
print(np.std(err_m),np.std(err_v))
'''
'''
gpd_obj = Gaussian_plateu_distribution(0.6,.05,.3)
plt.plot(np.linspace(0,1,1000),[gpd_obj.pdf(x) for x in np.linspace(0,1,1000)],color='black')
plt.text(0.25, 1, '$\mu=0.6$', fontsize=12)
plt.text(0.25, 0.95, '$\sigma = 0.05$', fontsize=12)
plt.text(0.25, 0.90, '$h=0.3$', fontsize=12)
plt.show()
'''
            
'''
opinions,corr_mat = generate_samples()
opinion_marginals = np.sum(opinions,axis=0)/100
opinion_marginals_props = [0.4,0.2,0.3,0.1]
sample_contexts = np.random.choice(np.arange(len(opinion_marginals_props)),size=100,p=opinion_marginals_props)
samples = [(np.sum(np.random.choice([1,0],size=10,p=[opinion_marginals[sample_contexts[i]],1-opinion_marginals[sample_contexts[i]]])),) for i in np.arange(100)]
xs = [(s[0],10-s[0]) for s in samples]
#xs = np.array([(5,5), (9,1), (8,2), (4,6), (7,3)])
#thetas = np.array([[0.6, 0.4], [0.5, 0.5]])
thetas = np.array([[x, 1-x] for x in opinion_marginals])

i, thetas, ws = em(xs, thetas)
print(i,'EM est.')
for theta in thetas:
    print(theta)
ws = np.mean(ws,axis=1)
print('prop',ws)
print('True est.')
print(opinion_marginals)
'''
'''
thetas = np.array([[1.52085510e-311, 9.99999000e-001],
 [5.76850271e-001, 4.23149729e-001],
 [9.45689684e-001, 5.43103164e-002],
 [8.11721432e-001, 1.88278568e-001]])
xs = [(8, 2), (8, 2), (9, 1), (10, 0), (8, 2), (10, 0), (8, 2), (7, 3), (9, 1), (10, 0), (10, 0), (8, 2), (10, 0), (6, 4), (10, 0), (6, 4), (7, 3), (10, 0), (9, 1), (10, 0), (4, 6), (10, 0), (5, 5), (7, 3), (8, 2), (4, 6), (10, 0), (8, 2), (8, 2), (5, 5), (9, 1), (9, 1), (10, 0), (5, 5), (9, 1), (6, 4), (8, 2), (7, 3), (5, 5), (8, 2), (7, 3), (6, 4), (7, 3), (5, 5), (10, 0), (8, 2), (5, 5), (9, 1), (10, 0), (10, 0), (7, 3), (7, 3), (5, 5), (9, 1), (6, 4), (6, 4), (6, 4), (8, 2), (9, 1), (5, 5), (5, 5), (7, 3)]
em(xs,thetas)
'''