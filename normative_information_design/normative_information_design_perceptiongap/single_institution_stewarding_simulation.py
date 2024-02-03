import solving_tools
import perception_gap_information_design
import numpy as np
import os
import utils
import pickle

if __name__ == "__main__":
    print('Extensive Stewarding Simulation')
    print(  '===============================')
    common_prior_appr_input = (5,2)
    common_prior_appr = (5,2)
    common_prior_disappr = (2,5)
    common_proportion_prior = (5,5)
    inst_type = 'extensive'
    attr_dict = {'distr_params':{'mean_op_degree_apr':0.7,'mean_op_degree_disapr':0.3,'apr_weight':0.5,'SD':0.2},
                                                            'distr_shape':'U',
                                                              'extensive': True if inst_type=='extensive' else False,
                                                              'common_prior_appr': common_prior_appr,
                                                              'common_prior_disappr': common_prior_disappr,
                                                              'common_proportion_prior': common_proportion_prior,
                                                              'common_prior_appr_input': common_prior_appr_input,
                                                              'only_intensive': True,
                                                              'homogenous_priors': True,
                                                              'num_players':100,
                                                              'alpha':0.3,
                                                              'outgroup_rhetoric_intensity':0.3,
                                                              'normal_constr_w':0.1,
                                                              'num_batches':10}
    run_param ={'common_prior_appr_input':common_prior_appr_input,
                        'common_prior_appr':common_prior_appr,
                        'common_prior_disappr':common_prior_disappr,
                        'common_proportion_prior':common_proportion_prior,
                        'normal_constr_w':attr_dict['normal_constr_w'],
                        'only_intensive':False,
                        'credible':True}
    run_param['attr_dict'] = attr_dict
    file_path = 'data/rhet_eq_estimation.pkl'
    if os.path.exists(file_path):
        run_param['rhetoric_estimation_model'] = pickle.load(open(file_path, "rb"))
    else:
        print('Generating Rhetoric Equilibrium Estimation Model')
        model = utils.generate_rhetoric_equilibrium_estimation_model()
        run_param['rhetoric_estimation_model'] = model
        pickle.dump(model, open(file_path, "wb"))
    attr_dict['rhetoric_estimation_model'] = run_param['rhetoric_estimation_model']
    run_param['posterior_prediction_model'] = dict()
    run_param['posterior_prediction_model']['appr'] = pickle.load(open('data/approximator_appr_'+str(run_param['normal_constr_w']).replace('.','-')+'.pkl','rb'))
    run_param['posterior_prediction_model']['disappr'] = pickle.load(open('data/approximator_disappr_'+str(run_param['normal_constr_w']).replace('.','-')+'.pkl','rb'))
    attr_dict['posterior_prediction_model'] = run_param['posterior_prediction_model']
    opt_policy_ingroup = solving_tools.run_simulation(inst_type, 'ingroup', attr_dict, show_plots=True)
    print('Optimal Policy for Ingroup: ', opt_policy_ingroup)
    extensive_ingroup_optimal = {attr_dict['normal_constr_w']:{'type':'appr', 'opt_signals':opt_policy_ingroup}}
    opt_policy_outgroup = solving_tools.run_simulation(inst_type, 'outgroup', attr_dict, show_plots=True)
    print('Optimal Policy for Outgroup: ', opt_policy_outgroup)
    extensive_outgroup_optimal = {attr_dict['normal_constr_w']:{'type':'disappr', 'opt_signals':{k:np.random.uniform(min(k,v),max(k,v)) for k,v in opt_policy_outgroup.items() if k < 0.5}}}

    #perception_gap_information_design.single_inst_run(extensive_outgroup_optimal=extensive_outgroup_optimal,extensive_ingroup_optimal=extensive_ingroup_optimal,attr_dict=attr_dict,run_param=run_param)
    #print('Extensive Stewarding Simulation Complete')


