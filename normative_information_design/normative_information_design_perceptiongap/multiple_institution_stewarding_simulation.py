import solving_tools
import perception_gap_information_design
import numpy as np
import os
import argparse
import json
import pickle
import utils
from utils import load_from_json
import constants

def save_to_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4)




def main():
    parser = argparse.ArgumentParser(
        description="Run simulations for Stewarding Simulation."
    )
    parser.add_argument(
        "simulation_type",
        choices=["single_institution", "multiple_institutions", "optimal_signalling_plots"],
        help="Specify which simulation to run: single_institution or multiple_institutions or optimal_signalling_plots.",
    )
    args = parser.parse_args()

    print('Stewarding Simulation')
    print(  '===============================')
    common_prior_appr_input = constants.COMMON_PRIOR_APPR_INPUT
    common_prior_appr = constants.COMMON_PRIOR_APPR
    common_prior_disappr = constants.COMMON_PRIOR_DISAPPR
    common_proportion_prior = constants.COMMON_PROPORTION_PRIOR
    inst_opt_policy, inst_sampling_ratios = {}, {}
    data_directory = os.path.join(os.getcwd(), 'data')
    filename = os.path.join(data_directory, 'inst_opt_policy.json')
    filename_sampling_ratios = os.path.join(data_directory, 'inst_sampling_ratios.json')

    if os.path.isfile(filename):
        inst_opt_policy = load_from_json(filename)
    if os.path.isfile(filename_sampling_ratios):
        inst_sampling_ratios = load_from_json(filename_sampling_ratios)
        
    attr_dict = constants.ATTR_DICT
    run_param =constants.RUN_PARAM
    run_param['attr_dict'] = attr_dict
    ''' This is a regression model constructed to estimate the equilibrium rhetoric. This is done to make the simulation more efficient.'''
    file_path = os.path.join(os.getcwd(),'pickles','rhet_eq_estimation.pkl')
    if os.path.exists(file_path):
        run_param['rhetoric_estimation_model'] = pickle.load(open(file_path, "rb"))
    else:
        print('Generating Rhetoric Equilibrium Estimation Model')
        model = utils.generate_rhetoric_equilibrium_estimation_model(run_param)
        run_param['rhetoric_estimation_model'] = model
        pickle.dump(model, open(file_path, "wb"))
    
    if args.simulation_type == "optimal_signalling_plots":
        regenerate = True
        show_plots = True
    else:
        regenerate = False
        show_plots = False
    ''' 
    If the optimal policy and sampling ratios have not been generated, the run the simulation to solve the optimal policy and store that in the json file.
    '''
    if len(inst_opt_policy) == 0 or len(inst_sampling_ratios) == 0 or regenerate:
        for inst_type in ['intensive','extensive']:
            attr_dict['extensive'] = True if inst_type=='extensive' else False
            attr_dict['rhetoric_estimation_model'] = run_param['rhetoric_estimation_model']
            inst_opt_policy[inst_type] = {'appr':dict(), 'disappr': dict()}
            
            opt_policy_ingroup, max_rewards_ingroup = solving_tools.run_simulation(inst_type, 'ingroup', attr_dict, show_plots=show_plots)
            print(f'Optimal {inst_type} Policy for Ingroup: ', opt_policy_ingroup)
            for k,v in opt_policy_ingroup.items():
                inst_opt_policy[inst_type]['appr'][k] = (None,v) if inst_type == 'intensive' else (v,v)
                if inst_type == 'intensive':
                    inst_opt_policy[inst_type]['disappr'][round(1-k,1)] = (1-v,None)
            
            opt_policy_outgroup, max_rewards_outgroup = solving_tools.run_simulation(inst_type, 'outgroup', attr_dict, show_plots=show_plots)
            print('Optimal Policy for Outgroup: ', opt_policy_outgroup)
            inst_sampling_ratios[inst_type] = {'appr':max_rewards_ingroup, 'disappr':max_rewards_outgroup}
            print(f'Sampling Ratios for {inst_type} : ', inst_sampling_ratios[inst_type])
            for k,v in opt_policy_outgroup.items():
                if inst_type == 'intensive':
                    if k <= 0.5:
                        inst_opt_policy[inst_type]['disappr'][k] = (inst_opt_policy[inst_type]['disappr'][k][0],v)
                    inst_opt_policy[inst_type]['appr'][round(k+0.5,1)] = (1-v,inst_opt_policy[inst_type]['appr'][round(k+0.5,1)][1])
                else:
                    inst_opt_policy[inst_type]['disappr'][k] = (v,v)
            if inst_type == 'extensive':
                inst_opt_policy[inst_type]['disappr'][0.5] = (inst_opt_policy[inst_type]['appr'][0.5][0],inst_opt_policy[inst_type]['appr'][0.5][1])
                
            
            #extensive_ingroup_optimal = {attr_dict['normal_constr_w']:{'type':'appr', 'opt_signals':opt_policy_ingroup}}
        if not args.simulation_type == "optimal_signalling_plots":     
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
    run_param['attr_dict']['tailored_alpha'] = False   
    run_param['extensive_optimal'] = inst_opt_policy['extensive']
    run_param['intensive_optimal'] = inst_opt_policy['intensive']
    run_param['inst_sampling_ratios'] = inst_sampling_ratios

    if args.simulation_type == "single_institution":
        print("Running Single Institution Simulation...")
        perception_gap_information_design.multiple_inst_run(attr_dict=attr_dict,run_param=run_param)
    elif args.simulation_type == "multiple_institutions":
        print("Running Multiple Institutions Simulation...")
        perception_gap_information_design.run_sim_single_institution(run_param)
    
    
    if args.simulation_type == "single_institution" or args.simulation_type == "multiple_institutions":
        if os.path.exists(file_path):
            os.remove(file_path)
        print('Stewarding Simulation Complete')

if __name__ == "__main__":
    main()
