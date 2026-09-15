import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load and prepare the data
df = pd.read_csv('output/finetuning_results.csv') # Adjusted to match the output folder from your main.py

# Split 'Strategy' into 'Base_Strategy' and 'Epoch'
df[['Base_Strategy', 'Epoch']] = df['Strategy'].str.split('_E', expand=True)
df['Epoch'] = df['Epoch'].astype(int)

# Ensure the output directory exists
os.makedirs("img", exist_ok=True)

# 2. Define the exact groupings based on your markdown tables
# (Including R180 and Weight38 in the Low Rank group as they are part of that sweep methodology)
groups = {
    "Low Rank": {
        "models": ['Full', 'R10', 'Weight2', 'R180', 'Weight38'],
        "filename": "img/low_rank_finetuning.png"
    },
    "Model Parameter Based Comparison": {
        "models": ['Full', 'Greedy', 'R150', 'Weight32'],
        "filename": "img/model_param_finetuning.png"
    }
}

# 3. Plotting function for grouped bar charts
def plot_finetuning_group(group_df, models_list, title_text, filename):
    epochs = sorted(group_df['Epoch'].unique())
    
    # Set up the figure and axes
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    fig.suptitle(title_text, fontsize=16, y=0.96)
    
    # Variables for grouped bar calculations
    x = np.arange(len(models_list))
    width = 0.2
    multiplier_center = (len(epochs) - 1) / 2
    
    metrics = [
        ('Val Char Acc', 'Validation Character Accuracy', axes[0], 'Accuracy'),
        ('Val Seq Acc', 'Validation Sequence Accuracy', axes[1], 'Accuracy'),
        # ('Val Loss', 'Validation Loss', axes[2], 'Loss')
    ]
    
    for metric_col, title, ax, ylabel in metrics:
        for i, epoch in enumerate(epochs):
            offset = width * i
            
            # Filter and align data to the specific model list order
            epoch_data = group_df[group_df['Epoch'] == epoch].set_index('Base_Strategy').reindex(models_list)
            
            # Plot the bars tightly together
            ax.bar(x + offset, epoch_data[metric_col], width, label=f'Epoch {epoch}')
        
        # Formatting
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Center x-ticks under the grouped bars
        ax.set_xticks(x + width * multiplier_center)
        ax.set_xticklabels(models_list, rotation=0, fontsize=11)
        
        # Adjust legend position based on metric to avoid blocking the bars
        legend_loc = 'lower right' if 'Acc' in metric_col else 'upper right'
        ax.legend(title='Fine-Tuning', loc=legend_loc)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if 'Acc' in metric_col:
            ax.set_ylim(0, 1.05)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() # Close to prevent overlapping states between the two plots

# 4. Generate the plots
for group_title, config in groups.items():
    print(f"Generating plot for {group_title}...")
    
    # Filter the dataframe for only the models in this specific group
    filtered_df = df[df['Base_Strategy'].isin(config['models'])]
    
    # Run the plotting function
    plot_finetuning_group(
        filtered_df, 
        config['models'], 
        group_title, 
        config['filename']
    )

print("Plots successfully saved to img/low_rank_finetuning.png and img/model_param_finetuning.png")