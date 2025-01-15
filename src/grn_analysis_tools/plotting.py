import matplotlib.pyplot as plt
import numpy as np
import math
# import scanpy as sc
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from . import grn_formatting

def plot_auroc_auprc(
    confusion_matrix_score_dict: dict,
    save_path: str
    ):
    """Plots the AUROC and AUPRC"""
    
    # Define figure and subplots for combined AUROC and AUPRC plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))    
    
    for method, score_dict in confusion_matrix_score_dict.items():
        y_true = score_dict['y_true']
        y_scores = score_dict['y_scores']
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        prc_auc = auc(recall, precision)
        
        axes[0].plot(fpr, tpr, label=f'{method} AUROC = {roc_auc:.2f}')
        axes[0].plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance
        
        axes[1].plot(recall, precision, label=f'{method} AUPRC = {prc_auc:.2f}')
        
    axes[0].set_title(f"AUROC")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_ylim((0,1))
    axes[0].set_xlim((0,1))
        
        
    axes[1].set_title(f"AUPRC")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim((0,1))
    axes[1].set_xlim((0,1))
    
    # Place the ROC legend below its plot
    axes[0].legend(
        loc="upper center",  # Anchor the legend at the upper center of the bounding box
        bbox_to_anchor=(0.5, -0.2),  # Move it below the axes
        ncol=1,  # Number of columns in the legend
    )

    # Place the PR legend below its plot
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=1,
    )

    # Adjust layout and display the figure
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    
def plot_all_samples_auroc_auprc(
    confusion_matrix_score_dict: dict,
    save_path: str
):
    """
    Plots combined ROC and PR curves for multiple methods on the same plot.

    Parameters:
    ----------
    confusion_matrix_score_dict : dict
        Dictionary containing y_true and y_scores for each method.
    save_path : str
        Path to save the resulting plot.
    """
    # Define figure for ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Colors for the plot
    colors = plt.cm.tab10.colors

    # Iterate through methods and plot all sublists, with one label per method
    for i, (method, score_dict) in enumerate(confusion_matrix_score_dict.items()):
        roc_aucs = []
        prc_aucs = []

        # Iterate over samples within the method
        for j, (y_true, y_scores) in enumerate(zip(score_dict['y_true'], score_dict['y_scores'])):
            # Assign a unique color for each sample
            sample_color = colors[j % len(colors)]

            # Calculate ROC and PR metrics
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)

            # Compute AUROC and AUPRC
            roc_auc = auc(fpr, tpr)
            prc_auc = auc(recall, precision)

            # Append metrics for averaging
            roc_aucs.append(roc_auc)
            prc_aucs.append(prc_auc)

            # Plot individual curves
            axes[0].plot(fpr, tpr, color=sample_color, alpha=0.7, label=f'Sample {j+1} (AUROC = {roc_auc:.2f})')
            axes[1].plot(recall, precision, color=sample_color, alpha=0.7, label=f'Sample {j+1} (AUPRC = {prc_auc:.2f})')

        # Add a label for method averages
        axes[0].plot([], [], color=colors[i % len(colors)], linewidth=2)
        axes[1].plot([], [], color=colors[i % len(colors)], linewidth=2)

        
    # Customize ROC plot
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)  # Diagonal line for random performance
    axes[0].set_title(f"{method} ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_ylim((0, 1))
    axes[0].set_xlim((0, 1))

    # Customize PR plot
    axes[1].set_title(f"{method} Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim((0, 1))
    axes[1].set_xlim((0, 1))
    
    axes[0].legend(
    loc="upper center", 
    bbox_to_anchor=(0.5, -0.3),
    ncol=1,  # More columns for wrapping
    )

    axes[1].legend(
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.3),
        ncol=1,
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    
def plot_multiple_method_auroc_auprc(
    confusion_matrix_score_dict: dict,
    save_path: str
    ):
    """
    Plots combined ROC and PR curves for multiple methods on the same plot.

    Parameters:
    ----------
    confusion_matrix_score_dict : dict
        Dictionary containing y_true and y_scores for each method.
    save_path : str
        Path to save the resulting plot.
    """
    # Define figure for ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors for the plot
    colors = plt.cm.tab10.colors

    # Iterate through methods and plot all sublists, with one label per method
    for i, (method, score_dict) in enumerate(confusion_matrix_score_dict.items()):
        all_fpr = []
        all_tpr = []
        all_precision = []
        all_recall = []
        roc_aucs = []
        prc_aucs = []

        # Process each sublist
        for y_true, y_scores in zip(score_dict['y_true'], score_dict['y_scores']):
            # Calculate ROC and PR metrics
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)

            # Compute AUROC and AUPRC
            roc_auc = auc(fpr, tpr)
            prc_auc = auc(recall, precision)

            # Append metrics for averaging
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_precision.append(precision)
            all_recall.append(recall)
            roc_aucs.append(roc_auc)
            prc_aucs.append(prc_auc)

            # Plot individual curves without labels
            axes[0].plot(fpr, tpr, color=colors[i % len(colors)], alpha=1)
            axes[1].plot(recall, precision, color=colors[i % len(colors)], alpha=1)

        # Compute average AUROC and AUPRC
        avg_roc_auc = sum(roc_aucs) / len(roc_aucs)
        avg_prc_auc = sum(prc_aucs) / len(prc_aucs)

        # Add single label for method with average metrics
        axes[0].plot([], [], label=f'{method} (AUROC = {avg_roc_auc:.2f})', color=colors[i % len(colors)])
        axes[1].plot([], [], label=f'{method} (AUPRC = {avg_prc_auc:.2f})', color=colors[i % len(colors)])

    # Customize ROC plot
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)  # Diagonal line for random performance
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_ylim((0,1))
    axes[0].set_xlim((0,1))

    # Customize PR plot
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim((0,1))
    axes[1].set_xlim((0,1))
    
        # Place the ROC legend below its plot
    axes[0].legend(
        loc="upper center",  # Anchor the legend at the upper center of the bounding box
        bbox_to_anchor=(0.5, -0.2),  # Move it below the axes
        ncol=1,  # Number of columns in the legend
    )

    # Place the PR legend below its plot
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=1,
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_normal_and_randomized_roc_prc(
    confusion_matrix_score_dict: dict,
    save_path: str
):
    """
    Plots combined ROC and PRC curves for normal and randomized scores of multiple methods.

    Parameters:
    ----------
    confusion_matrix_score_dict : dict
        Dictionary containing normal and randomized y_true and y_scores for each method.
    save_path : str
        Path to save the resulting plot.
    """
    # Define figure for ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors for the plot
    colors = plt.cm.tab10.colors
    
    # Iterate through methods
    for i, (method, score_dict) in enumerate(confusion_matrix_score_dict.items()):
        # Extract normal and randomized scores
        normal_y_true = score_dict['normal_y_true']
        normal_y_scores = score_dict['normal_y_scores']
        randomized_y_true = score_dict['randomized_y_true']
        randomized_y_scores = score_dict['randomized_y_scores']

        # Initialize lists for averaging metrics
        all_fpr_normal, all_tpr_normal, all_fpr_random, all_tpr_random = [], [], [], []
        all_precision_normal, all_recall_normal, all_precision_random, all_recall_random = [], [], [], []
        roc_aucs_normal, roc_aucs_random, prc_aucs_normal, prc_aucs_random = [], [], [], []

        # Process normal scores
        y_true = normal_y_true.values
        y_scores = normal_y_scores.values

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        prc_auc = auc(recall, precision)

        all_fpr_normal.append(fpr)
        all_tpr_normal.append(tpr)
        all_precision_normal.append(precision)
        all_recall_normal.append(recall)
        roc_aucs_normal.append(roc_auc)
        prc_aucs_normal.append(prc_auc)

        # Plot individual curves for normal
        axes[0].plot(fpr, tpr, color=colors[i % len(colors)], linestyle='-')
        axes[1].plot(recall, precision, color=colors[i % len(colors)], linestyle='-')

        # Process randomized scores
        y_true = randomized_y_true.values
        y_scores = randomized_y_scores.values
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        prc_auc = auc(recall, precision)

        all_fpr_random.append(fpr)
        all_tpr_random.append(tpr)
        all_precision_random.append(precision)
        all_recall_random.append(recall)
        roc_aucs_random.append(roc_auc)
        prc_aucs_random.append(prc_auc)

        # Plot individual curves for randomized
        axes[0].plot(fpr, tpr, color=colors[i % len(colors)], linestyle='--')
        axes[1].plot(recall, precision, color=colors[i % len(colors)], linestyle='--')

        # Compute average metrics for labeling
        avg_roc_auc_normal = sum(roc_aucs_normal) / len(roc_aucs_normal)
        avg_prc_auc_normal = sum(prc_aucs_normal) / len(prc_aucs_normal)
        avg_roc_auc_random = sum(roc_aucs_random) / len(roc_aucs_random)
        avg_prc_auc_random = sum(prc_aucs_random) / len(prc_aucs_random)

        # Add single label for normal and randomized
        axes[0].plot([], [], label=f'{method} Normal (AUROC = {avg_roc_auc_normal:.2f})', color=colors[i % len(colors)], linestyle='-')
        axes[0].plot([], [], label=f'{method} Randomized (AUROC = {avg_roc_auc_random:.2f})', color=colors[i % len(colors)], linestyle='--')
        axes[1].plot([], [], label=f'{method} Normal (AUPRC = {avg_prc_auc_normal:.2f})', color=colors[i % len(colors)], linestyle='-')
        axes[1].plot([], [], label=f'{method} Randomized (AUPRC = {avg_prc_auc_random:.2f})', color=colors[i % len(colors)], linestyle='--')

    # Customize ROC plot
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)  # Diagonal line for random performance
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_ylim((0, 1))
    axes[0].set_xlim((0, 1))

    # Customize PR plot
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim((0, 1))
    axes[1].set_xlim((0, 1))
    
        # Place the ROC legend below its plot
    axes[0].legend(
        loc="upper center",  # Anchor the legend at the upper center of the bounding box
        bbox_to_anchor=(0.5, -0.2),  # Move it below the axes
        ncol=1,  # Number of columns in the legend
    )

    # Place the PR legend below its plot
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=1,
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_multiple_histogram_with_thresholds(
    ground_truth_dict: dict,
    inferred_network_dict: dict,
    save_path: str
    ) -> None:
    """
    Generates histograms of the TP, FP, TN, FN score distributions for each method. Uses a lower threshold of 1 stdev below
    the mean ground truth score. 

    Parameters
    ----------
        ground_truth_dict (dict): _description_
        inferred_network_dict (dict): _description_
        result_dir (str): _description_
    """
    
    num_methods = len(ground_truth_dict.keys())

    # Maximum columns per row
    max_cols = 4

    # Calculate the number of rows and columns
    num_cols = min(num_methods, max_cols)  # Up to 4 columns
    num_rows = math.ceil(num_methods / max_cols)  # Rows needed to fit all methods

    # logging.debug(f"Number of rows: {num_rows}, Number of columns: {num_cols}")
    
    plt.figure(figsize=(18, 8))

    # Plot for each method
    for i, method_name in enumerate(ground_truth_dict.keys()):  
        
        # Extract data for the current method
        ground_truth_scores = ground_truth_dict[method_name]['Score'].dropna()
        inferred_scores = inferred_network_dict[method_name]['Score'].dropna()
        
        plt.subplot(num_rows, num_cols, i+1)
        
        # Define the threshold
        lower_threshold = np.mean(ground_truth_scores) - np.std(ground_truth_scores)
        
        mean = np.mean(np.concatenate([ground_truth_scores, inferred_scores]))
        std_dev = np.std(np.concatenate([ground_truth_scores, inferred_scores]))

        xmin = mean - 4 * std_dev
        xmax = max(1e-6, mean + 4 * std_dev)
        
        # Split data into below and above threshold
        tp = ground_truth_scores[(ground_truth_scores >= lower_threshold) & (ground_truth_scores < xmax)]
        fn = ground_truth_scores[(ground_truth_scores <= lower_threshold) & (ground_truth_scores > xmin)]
        
        fp = inferred_scores[(inferred_scores >= lower_threshold) & (inferred_scores < xmax)]
        tn = inferred_scores[(inferred_scores <= lower_threshold) & (inferred_scores > xmin)]
        
        # Define consistent bin edges for the entire dataset based on the number of values
        num_bins = 200
        all_scores = np.concatenate([tp, fn, fp, tn])
        
        # Ensure that concatenating all scores creates a non-empty array
        if len(all_scores) == 0:
            raise ValueError("The combined scores array is empty, cannot generate histogram with thresholds.")
        
        bin_edges = np.linspace(np.min(all_scores), np.max(all_scores), num_bins)
        bin_edges = np.sort(np.unique(np.append(bin_edges, lower_threshold)))
        
        # Plot histograms for Oracle Score categories with consistent bin sizes
        plt.hist(tn, bins=bin_edges, alpha=0.7, color='#b6cde0', label='True Negative (TN)')
        plt.hist(fn, bins=bin_edges, alpha=0.7, color='#efc69f', label='False Negative (FN)')
        
        # Plot the positive values on top to make sure there is no ovelap
        plt.hist(fp, bins=bin_edges, alpha=0.7, color='#4195df', label='False Positive (FP)')
        plt.hist(tp, bins=bin_edges, alpha=0.7, color='#dc8634', label='True Positive (TP)')

        # Plot threshold line
        plt.axvline(x=lower_threshold, color='black', linestyle='--', linewidth=2)
        plt.title(f"{method_name.capitalize()} Score Distribution")
        plt.xlabel(f"log2 {method_name.capitalize()} Score")
        # plt.ylim(1, None)
        # plt.xlim([-20,20])
        plt.ylabel("Frequency")
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=200)
    plt.close()

def plot_cell_expression_histogram(
    adata_rna, 
    output_dir: str, 
    cell_type: str = None,
    ymin: int | float = 0,
    ymax: int | float = 100,
    xmin: int | float = 0,
    xmax: int | float = None,
    filename: str = 'avg_gene_expr_hist',
    filetype: str = 'png'
    
    ):
    """
    Plots a histogram showing the distribution of cell gene expression by percent of the population.
    
    Assumes the data has cells as columns and genes as rows

    Parameters
    ----------
        adata_rna (sc.AnnData):
            scRNAseq dataset with cells as columns and genes as rows
        output_dir (str):
            Directory to save the graph in
        cell_type (str):
            Can specify the cell type if the dataset is a single cell type
        ymin (int | float):
            The minimum y-axis value (in percentage, default = 0)
        ymax (int | float):
            The maximum y-axis value (in percentage, default = 100)
        xmin (int | float):
            The minimum x-axis value (default = 0)
        xmax (int | float):
            The maximum x-axis value
        filename (str):
            The name of the file to save the figure as
        filetype (str):
            The file extension of the figure (default = png)
    """        
    
    n_cells = adata_rna.shape[0]
    n_genes = adata_rna.shape[1]
    
    # Create the histogram and calculate bin heights
    plt.hist(adata_rna.obs["n_genes"], bins=30, edgecolor='black', weights=np.ones_like(adata_rna.obs["n_genes"]) / n_cells * 100)
    
    if cell_type == None:
        plt.title(f'Distribution of the number of genes expressed by cells in the dataset')
    else: 
        plt.title(f'Distribution of the number of genes expressed by {cell_type}s in the dataset')
        
    plt.xlabel(f'Number of genes expressed ({n_genes} total genes)')
    plt.ylabel(f'Percentage of cells ({n_cells} total cells)')
    
    plt.yticks(np.arange(0, min(ymax, 100) + 1, 5), [f'{i}%' for i in range(0, min(ymax, 100) + 1, 5)])
    plt.ylabel(f'Percentage of cells ({n_cells} total cells)')
    
    plt.savefig(f'{output_dir}/{filename}.{filetype}', dpi=300)
    plt.close()

def plot_metric_by_step_adjusted(sample_resource_dict, metric, ylabel, title, filename, divide_by_cpu=False):
    """
    Plots the specified metric for each sample and step.
    
    Args:
    - sample_resource_dict: Dictionary containing sample resource data.
    - metric: The metric to plot (e.g., 'user_time', 'system_time').
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - filename: Path to save the plot.
    - divide_by_cpu: Boolean indicating if the metric should be divided by 'percent_cpu'.
    """
    # Extract the metric data for each sample and step
    samples = sorted(sample_resource_dict.keys())
    steps = [f'Step_0{i}0' for i in [1, 2, 3, 5]]
    
    metric_data = {sample: [
        (sample_resource_dict[sample][step][metric] / sample_resource_dict[sample][step]['percent_cpu'] 
         if divide_by_cpu else sample_resource_dict[sample][step][metric])
        for step in steps
    ] for sample in samples}

    # Plotting the figure with each step on the x-axis and the specified metric on the y-axis
    x = np.arange(len(steps))  # x locations for the groups
    bar_width = 0.02  # Width of the bars
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each sample as a bar group, with each sample's bars slightly offset
    for i, sample in enumerate(samples):
        ax.bar(x + i * bar_width, metric_data[sample], bar_width, label=f'{sample.split("_")[0]} Cells')

    # Labeling the plot
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (len(samples) - 1) / 2)
    ax.set_xticklabels(steps, rotation=45)
    plt.figlegend(loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=8, ncol=1)

    # Show plot
    plt.tight_layout(rect=[0, 0.06, 0.85, 0.90])
    plt.savefig(filename, dpi=200)

def plot_total_metric_by_sample(sample_resource_dict, metric, ylabel, title, filename, divide_by_cpu=False):
    """
    Plots the total specified metric across all steps for each sample.
    
    Args:
    - sample_resource_dict: Dictionary containing sample resource data.
    - metric: The metric to plot (e.g., 'user_time', 'system_time').
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - filename: Path to save the plot.
    - divide_by_cpu: Boolean indicating if the metric should be divided by 'percent_cpu'.
    """
    # Calculate the total metric for each sample across all steps
    samples = sorted(sample_resource_dict.keys())
    
    if metric == 'max_ram':
        total_metric_data = [max(sample_resource_dict[sample][step][metric] for step in sample_resource_dict[sample]) for sample in samples]
    if metric == 'percent_cpu':
        total_metric_data = [sum(sample_resource_dict[sample][step][metric] for step in sample_resource_dict[sample]) / len(sample_resource_dict[sample]) for sample in samples]
    else:
        total_metric_data = []
        for sample in samples:
            total = sum(
                sample_resource_dict[sample][step][metric] / sample_resource_dict[sample][step]['percent_cpu'] 
                if divide_by_cpu else sample_resource_dict[sample][step][metric]
                for step in sample_resource_dict[sample]
            )
            total_metric_data.append(total)

    # Plotting the total metric for each sample
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(samples))  # x locations for the samples
    bar_width = 0,15  # Width of the bars


    ax.bar(x, total_metric_data, bar_width, color='blue', label=metric)

    # Labeling the plot
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([sample.split("_")[0] for sample in samples], rotation=45)

    # Show plot
    plt.tight_layout()
    plt.savefig(filename, dpi=200)

def plot_inference_score_histogram(inferred_network_df: pd.DataFrame, method_name: str, save_path: str):
    import os

    plt.figure(figsize=(18, 8))
    
    # Check for "Score" column
    if "Score" not in inferred_network_df.columns:
        raise ValueError("The input DataFrame must contain a 'Score' column.")
    
    inferred_network_copy = inferred_network_df.copy()
    
    # Calculate the mean and standard deviation as scalars
    mean_score = np.mean(inferred_network_copy["Score"])
    std_score = np.mean(inferred_network_copy["Score"])

    # Calculate thresholds for filtering
    top_threshold = mean_score + (std_score * 5)
    bottom_threshold = max(1e-6, float(mean_score) - (float(std_score) * 5))

    
    # Subset the scores to within the threshold
    inferred_network_copy = inferred_network_copy[
        (inferred_network_copy["Score"] <= top_threshold) & 
        (inferred_network_copy["Score"] >= bottom_threshold)
    ]
    
    # Handle non-positive scores for log transformation
    if (inferred_network_copy["Score"] <= 0).any():
        raise ValueError("Scores must be positive for log2 transformation.")
    
    # Set the Scores to a log2 transformation
    inferred_network_copy["Score"] = np.log2(inferred_network_copy["Score"])
    
    # Plot histogram
    plt.hist(
        inferred_network_copy["Score"],
        bins=200,
        alpha=1,
        color='#4195df',
        label="Inferred edge scores"
    )
    
    plt.title(f"{method_name.capitalize()} Inferred Network Score Distribution")
    plt.xlabel(f"log2 {method_name.capitalize()} Score")
    plt.ylabel("Frequency")
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

    return inferred_network_copy
