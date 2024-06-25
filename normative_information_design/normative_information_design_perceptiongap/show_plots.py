import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines
import os
from scipy.stats import ttest_ind

def single_institution_plots():
    def read_and_label(file_path, group_label):
        df = pd.read_csv(file_path)
        df = df[df['opinion'] >= 0.5]
        df['group'] = group_label
        df['appr_belief'] = np.where(df['opinion'] >= 0.5, df['in_belief'], df['out_belief'])
        df['disappr_belief'] = np.where(df['opinion'] >= 0.5, df['out_belief'], df['in_belief'])
        return df


    # Load and label dataframes
    df_control = read_and_label(os.path.join(os.getcwd(),'data','ta=False_single_'+'intensive'+'_control_U.csv'), 'Control (no stewarding)')
    df_treatment_extensive = read_and_label(os.path.join(os.getcwd(),'data','ta=False_single_'+'extensive'+'_treatment_U.csv'), 'Participatory')
    df_treatment_intensive = read_and_label(os.path.join(os.getcwd(),'data','ta=False_single_'+'intensive'+'_treatment_U.csv'), 'Ideological')

    # Combine all dataframes

    df_combined = pd.concat([ df_treatment_extensive, df_treatment_intensive,df_control]).reset_index(drop=True)
    df_combined = df_combined[df_combined['alpha'] >= 0.0]
    '''
    df_treatment_extensive_ta = read_and_label(os.path.join(os.getcwd(),'data','ta=True_single_'+'extensive'+'_treatment_U.csv'), 'Participatory')
    df_treatment_intensive_ta = read_and_label(os.path.join(os.getcwd(),'data','ta=True_single_'+'intensive'+'_treatment_U.csv'), 'Ideological')
    df_combined_ta = pd.concat([ df_treatment_extensive_ta, df_treatment_intensive_ta]).reset_index(drop=True)
    df_combined_ta['alpha'] = 'Targeted'
    df_combined = pd.concat([df_combined,df_combined_ta]).reset_index(drop=True)
    '''

    # Optionally, save the modified DataFrame back to a new CSV file
    #df_treatment_intensive_ta.to_csv('modified_combined_extensive.csv', index=False)
    df_combined.to_csv('modified_combined_extensive.csv', index=False)
    

    # Plotting
    plt.style.use('ggplot')
    sns.set_palette("dark")
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    '''
    # Appr Belief plot with alpha as another hue group
    sns.lineplot(data=df_combined, x='time_step', y='appr_belief', hue='alpha', style='group',
                estimator='mean', ci="sd", 
                legend='full',palette='bright')
    '''
    # Disappr Belief plot on the same axes for direct comparison
    sns.lineplot(data=df_combined, x='time_step', y='disappr_belief', hue='alpha', style='group',
                estimator='mean', ci="sd", 
                legend='full',palette='bright')

    axs.set_title('Mean Outgroup and Ingroup Belief over Time within a population (of approvers)')
    axs.set_xlabel('Time Step')
    axs.set_ylabel('Mean Belief')
    '''
    legend_elements = []
    alpha_values = df_combined['alpha'].unique()
    group_values = df_combined['group'].unique()
    
    for group in group_values:
        for alpha in alpha_values:
            legend_elements.append(mlines.Line2D([], [], color=sns.color_palette("bright")[list(alpha_values).index(alpha)], 
                                                 label=f'{group}, alpha={alpha}'))
    
    plt.legend(handles=legend_elements, title='Stewarding institution type and platform moderation strictness')
    '''
    plt.legend()

    plt.tight_layout()

    # Plot for Proportion of Positive Participation Over Time by Group and alpha
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_combined, x='time_step', y='participation', hue='alpha', style='group',
                ci='sd', estimator='mean', markers=True,palette='bright',markersize=10)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title('Population participation rate under different institutional stewarding policies over time')
    plt.xlabel('Time Step')
    plt.ylabel('Participation proportion')
    plt.tight_layout()
    plt.show()

def multiple_institutions_plot():
    file_path = os.path.join(os.getcwd(),'data','multiple_homo=FalseU.csv')
    df = pd.read_csv(file_path)
    df = df[df['listened_to'] != 'none']
    df_opinion_high = df[df['opinion'] >= 0.5]
    df_opinion_low = df[df['opinion'] < 0.5]

    # Create scatter plots
    plt.figure(figsize=(12, 6))

    # Scatter plot for opinion >= 0.5
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_opinion_high, x='listened_to', y='out_belief')
    plt.title('Opinion >= 0.5')
    plt.xlabel('Listened To')
    plt.ylabel('Out Belief')

    # Scatter plot for opinion < 0.5
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_opinion_low, x='listened_to', y='out_belief')
    plt.title('Opinion < 0.5')
    plt.xlabel('Listened To')
    plt.ylabel('Out Belief')

    # Adjust layout and show plot
    plt.tight_layout()
    '''
    df = pd.read_csv(file_path)

    df_filtered = df[(df['opinion'] > 0.5) & ((df['listened_to'] != 'none') | (df['time_step'] == 0))]

    
    # Create a line plot
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df_filtered, x='time_step', y='out_belief', hue='listened_to', 
                estimator='mean', ci='sd')

    plt.title('Mean and SD of Out Belief Over Time (Opinion > 0.5)')
    plt.xlabel('Time Step')
    plt.ylabel('Out Belief')
    plt.legend(title='Listened To')

    # Show the plot
    plt.show()
    '''
    df = pd.read_csv(file_path)
    df.dropna(subset=['opinion', 'out_belief', 'time_step', 'listened_to'], inplace=True)

    #df = df[df['listened_to'] != 'none']
    #df_filtered = df[df['opinion'] >= 0.5]

    # Find the maximum time_step
    #max_time_step = df_filtered['time_step'].max()

    # Filter the data to include only rows with the maximum time_step
    #df_max_time_step = df_filtered[df_filtered['time_step'] == max_time_step]

    # Create a scatter plot
    #plt.figure(figsize=(10, 6))
    #sns.scatterplot(data=df_max_time_step, x='opinion', y='out_belief', hue='listened_to', palette='viridis')
    

    print(df.dtypes)

    # Convert data types if necessary
    df['opinion'] = pd.to_numeric(df['opinion'], errors='coerce')
    df['out_belief'] = pd.to_numeric(df['out_belief'], errors='coerce')
    df['time_step'] = pd.to_numeric(df['time_step'], errors='coerce')
    df['listened_to'] = df['listened_to'].astype('category')
    
    # Filter and plot again
    df_filtered = df[df['opinion'] >= 0.0]
    df_filtered['opinion cluster'] = pd.cut(df_filtered['opinion'], bins=[0.5, 0.6, 0.8, 1], right=False,labels=['moderate', 'severe', 'extreme'], include_lowest=True)
    max_time_step = df_filtered['time_step'].max()
    df_max_time_step = df_filtered[df_filtered['time_step'] > max_time_step*0.5]
    
    df_max_time_step_disappr_op = df[df['time_step'] == max_time_step]
    df_max_time_step_disappr_op = df_max_time_step_disappr_op[df_max_time_step_disappr_op['opinion'] < 0.5]
    print(np.mean(df_max_time_step_disappr_op['opinion']))
    

    
    # Create a scatter plot with a regression line
    #sns.lmplot(data=df_max_time_step, x='opinion', y='out_belief', hue='listened_to', palette='viridis', x_ci='sd')
    #sns.boxplot(data=df_max_time_step, x='opinion cluster', y='out_belief', hue='listened_to', palette='viridis')
    '''
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title('Scatter Plot of Out Belief vs. Opinion at Max Time Step')
    plt.xlabel('Opinion')
    plt.ylabel('Out Belief')
    plt.legend(title='Listened To')
    '''
    df = df_max_time_step
    df['opinion_group'] = pd.cut(df['opinion'], bins=[0, 0.5, 1], right=False, labels=['<0.5', '>=0.5'])

    # Group by 'listened_to' and 'opinion_group', then calculate mean and SD
    stats_df = df.groupby(['listened_to', 'opinion_group']).agg({
        'opinion': ['mean', 'std'],
        'out_belief': ['mean', 'std']
    })

    # Print the summary statistics
    print("Summary Statistics:")
    print(stats_df)

    # Collect data for t-tests
    t_test_results = []
    cohens_d_results = []
    categories = df['listened_to'].unique()
    for group in df['opinion_group'].cat.categories:
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat1 = categories[i]
                cat2 = categories[j]
                data1 = df[(df['listened_to'] == cat1) & (df['opinion_group'] == group)]
                data2 = df[(df['listened_to'] == cat2) & (df['opinion_group'] == group)]
                
                # Perform t-tests
                t_stat_opinion, p_val_opinion = ttest_ind(data1['opinion'], data2['opinion'], equal_var=False, nan_policy='omit')
                t_stat_out_belief, p_val_out_belief = ttest_ind(data1['out_belief'], data2['out_belief'], equal_var=False, nan_policy='omit')
                
                # Calculate Cohen's d
                n1, n2 = len(data1['opinion']), len(data2['opinion'])
                s1, s2 = data1['opinion'].std(), data2['opinion'].std()
                pooled_sd_opinion = np.sqrt(((n1-1) * s1**2 + (n2-1) * s2**2) / (n1 + n2 - 2))
                cohens_d_opinion = (data1['opinion'].mean() - data2['opinion'].mean()) / pooled_sd_opinion
                
                s1, s2 = data1['out_belief'].std(), data2['out_belief'].std()
                pooled_sd_out_belief = np.sqrt(((n1-1) * s1**2 + (n2-1) * s2**2) / (n1 + n2 - 2))
                cohens_d_out_belief = (data1['out_belief'].mean() - data2['out_belief'].mean()) / pooled_sd_out_belief

                # Collect t-test results
                t_test_results.append({
                    'comparison': f"{cat1} vs {cat2}",
                    'opinion_group': group,
                    'opinion_t_stat': t_stat_opinion,
                    'opinion_p_value': p_val_opinion,
                    'out_belief_t_stat': t_stat_out_belief,
                    'out_belief_p_value': p_val_out_belief,
                    'cohens_d_opinion': cohens_d_opinion,
                    'cohens_d_out_belief': cohens_d_out_belief
                })

    # Convert t-test results to DataFrame
    t_test_df = pd.DataFrame(t_test_results)

    # Print t-test results along with Cohen's d
    print("\nT-Test and Cohen's D Results:")
    print(t_test_df)
    #plt.show()

def plot_opinion_distr():
    from scipy.stats import gaussian_kde
    file_path = os.path.join(os.getcwd(),'data','multiple_homo=FalseU.csv')
    df = pd.read_csv(file_path)
    df_filtered = df[df['time_step'] == 0]
    df_filtered = df_filtered[df_filtered['run_id'] == 0]

    # Plot the PDF of the 'opinion' column
    plt.figure(figsize=(8, 6))  # Optional: adjust the size of the plot
    counts, bins, patches = plt.hist(df_filtered['opinion'], bins=10, edgecolor='black', alpha=0.5, label='Histogram')

    # Calculate the bin centers from the bin edges
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Calculate KDE using gaussian_kde from scipy, which allows more control
    kde = gaussian_kde(df_filtered['opinion'])
    # Evaluate the KDE on the bin centers
    kde = gaussian_kde(df_filtered['opinion'], bw_method='silverman')

    # Evaluate the KDE on a finer grid for a smoother line
    fine_bin_centers = np.linspace(bins[0], bins[-1], 400)
    kde_values = kde.evaluate(fine_bin_centers)

    # Manually adjust the y-values of the KDE to match the histogram's scale
    # Multiply by the total counts and bin width to convert density to counts
    bin_width = np.diff(bins)[0]
    scaled_kde_values = kde_values * bin_width * len(df_filtered['opinion'])

    # Plot the scaled KDE
    plt.plot(fine_bin_centers, scaled_kde_values, color='red', label='Density Function')
    plt.title('PDF of Opinion at Time Step 0')
    plt.xlabel('Opinion')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    #plot_opinion_distr()
    #single_institution_plots()
    multiple_institutions_plot()
