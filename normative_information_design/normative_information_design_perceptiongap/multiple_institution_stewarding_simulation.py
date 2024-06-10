import solving_tools
import perception_gap_information_design
import numpy as np
import os
import json
import pickle
import utils

def save_to_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4)

def load_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    print('Multiple Stewarding Simulation')
    print(  '===============================')
    common_prior_appr_input = (5,2)
    common_prior_appr = (5,2)
    common_prior_disappr = (2,5)
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
                                                                      'only_intensive': True,
                                                                      'homogenous_priors': True,
                                                                      'num_players':100,
                                                                      'alpha':0.6,
                                                                      'lambda_ougroup':0.5,
                                                                      'lambda_ingroup':1.5,
                                                                      'normal_constr_w':0.1,
                                                                      'rhet_thresh_mean':0.25,
                                                                      'num_batches':10,
                                                                      'num_timesteps':100,
                                                                      'print_log':False,
                                                                      'verbose':False,}
    run_param ={'common_prior_appr_input':common_prior_appr_input,
                        'common_prior_appr':common_prior_appr,
                        'common_prior_disappr':common_prior_disappr,
                        'common_proportion_prior':common_proportion_prior,
                        'normal_constr_w':attr_dict['normal_constr_w'],
                        'only_intensive':True,
                        'credible':True}
    run_param['attr_dict'] = attr_dict
    file_path = os.path.join(os.getcwd(),'pickles','rhet_eq_estimation.pkl')
    if os.path.exists(file_path):
        run_param['rhetoric_estimation_model'] = pickle.load(open(file_path, "rb"))
    else:
        print('Generating Rhetoric Equilibrium Estimation Model')
        model = utils.generate_rhetoric_equilibrium_estimation_model(run_param)
        run_param['rhetoric_estimation_model'] = model
        pickle.dump(model, open(file_path, "wb"))
    
    if len(inst_opt_policy) == 0 or len(inst_sampling_ratios) == 0:
        for inst_type in ['intensive','extensive']:
            attr_dict['extensive'] = True if inst_type=='extensive' else False
            attr_dict['rhetoric_estimation_model'] = run_param['rhetoric_estimation_model']
            inst_opt_policy[inst_type] = {'appr':dict(), 'disappr': dict()}
            
            opt_policy_ingroup, max_rewards_ingroup = solving_tools.run_simulation(inst_type, 'ingroup', attr_dict, show_plots=True)
            print(f'Optimal {inst_type} Policy for Ingroup: ', opt_policy_ingroup)
            for k,v in opt_policy_ingroup.items():
                inst_opt_policy[inst_type]['appr'][k] = (None,v)
                inst_opt_policy[inst_type]['disappr'][round(1-k,1)] = (1-v,None)
            
            opt_policy_outgroup, max_rewards_outgroup = solving_tools.run_simulation(inst_type, 'outgroup', attr_dict, show_plots=True)
            print('Optimal Policy for Outgroup: ', opt_policy_outgroup)
            inst_sampling_ratios[inst_type] = {'appr':max_rewards_ingroup, 'disappr':max_rewards_outgroup}
            print(f'Sampling Ratios for {inst_type} : ', inst_sampling_ratios[inst_type])
            for k,v in opt_policy_outgroup.items():
                if k <= 0.5:
                    inst_opt_policy[inst_type]['disappr'][k] = (inst_opt_policy[inst_type]['disappr'][k][0],v)
                inst_opt_policy[inst_type]['appr'][round(k+0.5,1)] = (1-v,inst_opt_policy[inst_type]['appr'][round(k+0.5,1)][1])
                
            
            #extensive_ingroup_optimal = {attr_dict['normal_constr_w']:{'type':'appr', 'opt_signals':opt_policy_ingroup}}
            
        save_to_json(inst_opt_policy, filename)
        save_to_json(inst_sampling_ratios, filename_sampling_ratios)
    else:
        for inst_type in ['intensive','extensive']:
            for k,v in inst_opt_policy[inst_type].items():
                inst_opt_policy[inst_type][k] = {round(float(_k),1):tuple(_v) for _k,_v in inst_opt_policy[inst_type][k].items()}
    #run_param['posterior_prediction_model'] = dict()
    #run_param['posterior_prediction_model']['appr'] = pickle.load(open(os.path.join(os.getcwd(),'pickles','approximator_appr_'+str(run_param['normal_constr_w']).replace('.','-')+'.pkl','rb')))
    #run_param['posterior_prediction_model']['disappr'] = pickle.load(open(os.path.join(os.getcwd(),'pickles','approximator_disappr_'+str(run_param['normal_constr_w']).replace('.','-')+'.pkl','rb')))
    run_param['attr_dict']['homogenous_priors'] = False      
    run_param['extensive_optimal'] = inst_opt_policy['extensive']
    run_param['intensive_optimal'] = inst_opt_policy['intensive']
    run_param['inst_sampling_ratios'] = inst_sampling_ratios
    perception_gap_information_design.multiple_inst_run(attr_dict=attr_dict,run_param=run_param)
    '''
    perception_gap_information_design.single_inst_run(extensive_outgroup_optimal = inst_opt_policy[inst_type],
                                                      extensive_ingroup_optimal = inst_opt_policy[inst_type],
                                                      attr_dict=attr_dict,
                                                      run_param=run_param)
    '''
    if os.path.exists(file_path):
        os.remove(file_path)
    print('Stewarding Simulation Complete')


