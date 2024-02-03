import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sympy as sp
from sympy.integrals.intpoly import cross_product
import pandas as pd
from IPython.display import display

def solver(): 
    '''
    # Create symbols x1 to x4
    k, a, u = sp.symbols('k a u')
    
    # Create symbols p and q
    
    
    # Define the equations
    equations = [
    sp.Eq(((1-k)*a*(1-(u/k))) - (k*(1-(u/(1-k)))) , 0)
    ]
    solution = sp.solve(equations, k)
    '''
    k = sp.Symbol('k')
    a = sp.Symbol('a')
    u = sp.Symbol('u')
    b = sp.Symbol('b')
    x = sp.Symbol('x')
    #solution = sp.solveset(((1-k)*a*(1-(u/k))) - (k*a*(1-(u/(1-k)))), a, domain=sp.S.Reals)
    solution = sp.solveset(((1-k)*a*0.5*(1-(u**2/k**2))) - (k*a*0.5*(1-(u/(1-k)))**2) + ((1-k)*b*(1-(u/k))) - (k*b*(1-(u/(1-k)))), a, domain=sp.S.Reals)
    expr1 = ((1-k)*a*0.5*(1-(u**2/k**2))) - (k*a*0.5*(1-(u/(1-k)))**2) + ((1-k)*b*(1-(u/k))) - (k*b*(1-(u/(1-k))))
    # Print the solution
    print("Solution:", solution)
    print(sp.latex(eval('-0.8*b*k*(1.0*k - 1.0)*(1.0*k - 0.5)*(1.0*k**2 - 1.0*k + 1.0*u)/(0.4*k**5 + 0.4*k**4*u - 1.0*k**4 - 0.4*k**3*u + 0.8*k**3 + 0.6*k**2*u**2 - 0.2*k**2 - 0.6*k*u**2 + 0.2*u**2)')))
    expr = -0.8*b*k*(1.0*k - 1.0)*(1.0*k - 0.5)*(1.0*k**2 - 1.0*k + 1.0*u)/(0.4*k**5 + 0.4*k**4*u - 1.0*k**4 - 0.4*k**3*u + 0.8*k**3 + 0.6*k**2*u**2 - 0.2*k**2 - 0.6*k*u**2 + 0.2*u**2)
    for _k in np.linspace(0,1,10):
        for _u in np.linspace(0,1,10):
            for _b in np.linspace(0,1,10):
                eval_expr = expr.subs([(k, _k),(u, _u), (b, _b)])
                #print(_u,_k,_b,eval_expr)
                eval_expr1 = expr1.subs([(k, _k),(u, _u), (b, _b), (a,eval_expr)])
                print(eval_expr,eval_expr1)
                '''
                if eval_expr !=0:
                    int_l = sp.integrate(eval_expr*x**2 + _b*x, (x, 0, (1-(_u/(1-_k)))))
                    int_h = sp.integrate(eval_expr*x**2 + _b*x, (x, _u/_k, 1))
                    if int_l >0 and int_h >0:
                        print(int_l,int_h,int_h/(int_l+int_h),_k)
                '''
def executor():
    x1,x2,x3,x4 = 4/7,0,3/7,1
    p,q = 0.7, 0.3
    qri_cond_is = (x1*p)/(x1*p+x2*q)#*((x1+x2)/2))
    qri_cond_gs = (x3*p)/(x3*p+x4*q)#*((x3+x4)/2)
    qrg_cond_is = (x2*q)/(x1*p+x2*q)#*((x1+x2)/2))
    qrg_cond_gs = (x4*q)/(x3*p+x4*q)#*((x3+x4)/2)
    print(qri_cond_is,qri_cond_gs,qrg_cond_is,qrg_cond_gs)
    qri = qri_cond_is*((x1+x2)/2) + qri_cond_gs*((x3+x4)/2)
    print('guilty percentage times',p*qrg_cond_is+q*qrg_cond_gs)
    print('posterior guilty',(qrg_cond_is+qrg_cond_gs)/(qri_cond_is+qri_cond_gs+qrg_cond_is+qrg_cond_gs))
    
def beta_cdf():
    
    from scipy.stats import beta
    
    # Define the parameters of the Beta distribution
    a1 = 2  # Shape parameter
    b1 = 25    # Shape parameter
    a2 = 8  # Shape parameter
    b2 = 2   # Shape parameter
    mix = 0.15
    # Generate x values for the CDF plot
    x = np.linspace(0, 1, 1000)
    theta = mix*(a1/(a1+b1)) + (1-mix)*(a2/(a2+b2))
    u_bar = 0.2
    l_l_unclipped = 1-(u_bar/(1-theta))
    l_h_unclipped = u_bar/theta
    l_l = np.clip(1-(u_bar/(1-theta)),0,0.5)
    l_h = np.clip(u_bar/theta,0.5,1)

    # Calculate the CDF values using the beta distribution's CDF function
    cdf_l = mix*beta.cdf(l_l, a1, b1) + (1-mix)*beta.cdf(l_l, a2, b2)
    cdf_r = mix*(1 - beta.cdf(l_h, a1, b1)) + (1-mix)*(1 - beta.cdf(l_h, a2, b2))
    pdf = mix*beta.pdf(x, a1, b1) + (1-mix)*beta.pdf(x, a2, b2)
    pdf_e = []
    for _x in np.linspace(0,1,10000):
        if _x < l_l:
            pdf_e.append(mix*beta.pdf(_x, a1, b1) + (1-mix)*beta.pdf(_x, a2, b2))
        elif _x >= l_h:
            pdf_e.append(mix*beta.pdf(_x, a1, b1) + (1-mix)*beta.pdf(_x, a2, b2))
        else:
            pdf_e.append(0)
    norm_pdf_e = [x/np.sum(pdf_e) for x in pdf_e]
    exp_pdf_e = np.sum([x1*x2 for x1,x2 in zip(np.linspace(0,1,10000),norm_pdf_e)])
    plt.plot(x, pdf,'black')
    plt.plot(np.linspace(0,1,10000), pdf_e,'red')
    print('theta',theta)
    print('unclipped','l_l,l_h',l_l_unclipped,l_h_unclipped)
    print('l_l,l_h',l_l,l_h)
    print('new theta',exp_pdf_e)
    
    plt.show()


pr,pb = (0.3,0.7)
u_bar,theta = 0.2,pb   
def calc_bayes_plausible_distr(rr,rb,br,bb,priors):
    #rr,rb,br,bb = sp.symbols('rr,rb,br,bb')
    #pr,pb = sp.symbols('pr,pb')
    #rr,rb,br,bb = 6/7,1/3,1/7,2/3
    pr,pb = priors
    r_signal = (rr/(rr+br))*pr + (rb/(rb+bb))*pb
    b_signal = (br/(rr+br))*pr + (bb/(rb+bb))*pb
    pos_state_r_signal_r = (rr*pr)/(rr*pr + rb*pb)
    pos_state_r_signal_b = (br*pr)/(br*pr + bb*pb)
    pos_state_b_signal_r = (rb*pb)/(rb*pb + rr*pr)
    pos_state_b_signal_b = (bb*pb)/(bb*pb + br*pr)
    expected_posterior_r = r_signal*pos_state_r_signal_r + b_signal*pos_state_r_signal_b
    expected_posterior_b = r_signal*pos_state_b_signal_r + b_signal*pos_state_b_signal_b
    #equations = [expected_posterior_r-pr,expected_posterior_b-pb,rr+br-1,rb+bb-1]
    #solutions = sp.solve(equations, rr,rb,br,bb, dict=True)
    #solved = {bb: 1.0 - rb, br: 1.0 - rr}
    #print('expected posteriors',expected_posterior_r,expected_posterior_b)
    #print(solutions)
    #receivers = ['rs','rh','bh','bs']
    #r_signal_ll = pos_state_r_signal_r*np.clip(1-(u_bar/pos_state_r_signal_r),0,0.5) + pos_state_b_signal_r*np.clip(1-(u_bar/(1-pos_state_b_signal_r)),0,0.5)
    #r_signal_lh = pos_state_r_signal_r*np.clip(u_bar/(1-pos_state_r_signal_r),0.5,1)  + pos_state_b_signal_r*np.clip(u_bar/pos_state_b_signal_r,0.5,1) 
    #b_signal_ll = pos_state_r_signal_b*np.clip(1-(u_bar/pos_state_r_signal_b),0,0.5) + pos_state_b_signal_b*np.clip(1-(u_bar/(1-pos_state_b_signal_b)),0,0.5)
    #b_signal_lh = pos_state_r_signal_b*np.clip(u_bar/(1-pos_state_r_signal_b),0.5,1)  + pos_state_b_signal_b*np.clip(u_bar/pos_state_b_signal_b,0.5,1)
    #exp_ll,exp_lh = r_signal*r_signal_ll + b_signal*b_signal_ll, r_signal*r_signal_lh + b_signal*b_signal_lh
    return expected_posterior_r,expected_posterior_b

if __name__ == "__main__":
    lst = []
    cols = ['rr','rb','orig_diff', 'changed_diff','orig_vals_l','orig_vals_h','changed_vals_l','changed_vals_h']
    
    priors = (0.6,0.4)
    #for rr in np.linspace(0.5,1,100):
    rr = 1
    for rb in np.linspace(0.01,0.99,100):
        bb = 1 - rb
        br = 1 - rr
        exp_r,exp_b = calc_bayes_plausible_distr(rr,rb,br,bb,priors)
        print(rb,exp_r,exp_b)
        #l_l = np.clip(1-(u_bar/(1-theta)),0,0.5)
        #l_h = np.clip(u_bar/theta,0.5,1) 
        #print([abs(l_l-l_h),abs(exp_ll-exp_lh),l_l,l_h,exp_ll,exp_lh])
        #lst.append([rr,rb,abs(l_l-l_h),abs(exp_ll-exp_lh),l_l,l_h,exp_ll,exp_lh])
    #df = pd.DataFrame(lst, columns=cols)
    #print(df.loc[df['changed_vals_l'].idxmin()])       
    #print(df.loc[df['changed_vals_h'].idxmax()])       
    '''     
    ctx_w = [0.3,0.3,0.4]
    ctx_marg_theta = [0.7]*3
    from itertools import product
    cross_product = list(product([0,1], repeat=3))
    for j in cross_product:
        exp_util = np.prod(np.asarray([nt if o==1 else (1-nt) for nt,o in zip(ctx_marg_theta,list(j))]))
        jtheta = np.sum(j)/3
        exp_util2 = np.sum(np.asarray([nt*o + (1-nt)*(1-o) for nt,o in zip(ctx_marg_theta,[jtheta]*3)]))/3
        print(j,exp_util,exp_util2, jtheta)
    '''