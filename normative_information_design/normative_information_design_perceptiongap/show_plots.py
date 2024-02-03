import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines

def single_institution_plots():
    def read_and_label(file_path, group_label):
        df = pd.read_csv(file_path)
        df = df[df['opinion'] >= 0.5]
        df['group'] = group_label
        df['appr_belief'] = np.where(df['opinion'] >= 0.5, df['in_belief'], df['out_belief'])
        df['disappr_belief'] = np.where(df['opinion'] >= 0.5, df['out_belief'], df['in_belief'])
        return df


    # Load and label dataframes
    df_control = read_and_label('data/single_'+'extensive'+'_control_U.csv', 'Control (no stewarding)')
    df_treatment_extensive = read_and_label('data/single_'+'extensive'+'_treatment_U.csv', 'Extensive inst.')
    df_treatment_intensive = read_and_label('data/single_'+'intensive'+'_treatment_U.csv', 'Intensive inst.')

    # Combine all dataframes

    df_combined = pd.concat([ df_treatment_extensive, df_treatment_intensive,df_control]).reset_index(drop=True)


    # Optionally, save the modified DataFrame back to a new CSV file
    df_combined.to_csv('modified_combined_extensive.csv', index=False)

    # Plotting
    plt.style.use('ggplot')
    sns.set_palette("dark")
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))

    # Appr Belief plot with alpha as another hue group
    sns.lineplot(data=df_combined, x='time_step', y='appr_belief', hue='alpha', style='group',
                estimator='mean', ci="sd", 
                legend='full',palette='bright')

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
    file_path = 'data/multiple_homo=False_U.csv'
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

    df = pd.read_csv(file_path)

    df_filtered = df[(df['opinion'] > 0.5) & ((df['listened_to'] != 'none') | (df['time_step'] == 0))]

    '''
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

    df = df[df['listened_to'] != 'none']
    df_filtered = df[df['opinion'] >= 0.5]

    # Find the maximum time_step
    max_time_step = df_filtered['time_step'].max()

    # Filter the data to include only rows with the maximum time_step
    df_max_time_step = df_filtered[df_filtered['time_step'] == max_time_step]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    #sns.scatterplot(data=df_max_time_step, x='opinion', y='out_belief', hue='listened_to', palette='viridis')
    

    print(df.dtypes)

    # Convert data types if necessary
    df['opinion'] = pd.to_numeric(df['opinion'], errors='coerce')
    df['out_belief'] = pd.to_numeric(df['out_belief'], errors='coerce')
    df['time_step'] = pd.to_numeric(df['time_step'], errors='coerce')
    df['listened_to'] = df['listened_to'].astype('category')
    
    # Filter and plot again
    df_filtered = df[df['opinion'] >= 0.5]
    max_time_step = df_filtered['time_step'].max()
    df_max_time_step = df_filtered[df_filtered['time_step'] > max_time_step/2]
    
    df_max_time_step_disappr_op = df[df['time_step'] == max_time_step]
    df_max_time_step_disappr_op = df_max_time_step_disappr_op[df_max_time_step_disappr_op['opinion'] < 0.5]
    print(np.mean(df_max_time_step_disappr_op['opinion']))
    
    # Create a scatter plot with a regression line
    sns.lmplot(data=df_max_time_step, x='opinion', y='out_belief', hue='listened_to', 
               palette='viridis', x_ci='sd')

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title('Scatter Plot of Out Belief vs. Opinion at Max Time Step')
    plt.xlabel('Opinion')
    plt.ylabel('Out Belief')
    plt.legend(title='Listened To')
    plt.show()

if __name__ == '__main__':
    single_institution_plots()
    #multiple_institutions_plot()
