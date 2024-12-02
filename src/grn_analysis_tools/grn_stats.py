import pandas as pd
import numpy as np
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from . import plotting
from . import grn_formatting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')   

def classify_interactions_by_threshold(ground_truth_df: pd.DataFrame, inferred_network_df: pd.DataFrame, lower_threshold: int | float = None):
    """
    Adds the columns 'true_interaction' and 'predicted_interaction' to the ground truth and inferred network dataframes.
    
    For ground truth edges, the true interaction is set to 1. The predicted interactions are set to 1 for edge scores 
    above the lower threshold (predicted positive) and 0 for edge scores below the lower threshold (predicted negative). 
    
    For inferred network edges not in the ground truth, the true interaction is set to 0. The predicted interactions are
    set to 1 for edge scores above the lower threshold (predicted positive) and 0 for edge scores below the lower 
    threshold (predicted negative).

    Parameters
    ----------
        ground_truth_df (pd.DataFrame): 
            This DataFrame contains the ground truth network in standard "Source" "Target" "Score" format.
        inferred_network_df (pd.DataFrame): 
            This DataFrame contains the inferred GRN network in standard "Source" "Target" "Score" format, with ground truth
            edges removed and genes not found in the ground truth network removed.
        lower_threshold (int | float, optional): 
            Optional threshold of ground truth scores used classify inferred GRN edge scores as True or False. Defaults
            to one standard deviation below the mean of the ground truth score distribution.

    Returns
    ----------
        ground_truth_df_copy (pd.DataFrame): 
            Ground truth network in standard "Source" "Target" "Score" format with added columns "true_interaction" and 
            "predicted_interaction".
            
        inferred_network_df_copy (pd.DataFrame):
            Inferred network in standard "Source" "Target" "Score" format with added columns "true_interaction" and 
            "predicted_interaction".
    """
    
    required_columns = ["Source", "Target", "Score"]
    missing_columns_ground_truth = set(required_columns) - set(ground_truth_df.columns)
    if missing_columns_ground_truth:
        raise ValueError(f"Missing required columns in ground_truth_df: {missing_columns_ground_truth}. Columns present: \
            {list(ground_truth_df.columns)}")
    
    missing_columns_inferred = set(required_columns) - set(inferred_network_df.columns)
    if missing_columns_inferred:
        raise ValueError(f"Missing required columns in ground_truth_df: {missing_columns_inferred}. Columns present: \
            {list(inferred_network_df.columns)}")
    
    # Ensure that the "Score" column exists in both the ground truth and inferred networks
    if ground_truth_df["Score"].isna().all():
        raise ValueError("Ground truth scores are empty. Please make sure that you have used \
            'add_inferred_scores_to_ground_truth', remove_ground_truth_edges_from_inferred', \
            and 'remove_tf_tg_not_in_ground_truth' to prepare the dataset")
    
    if inferred_network_df["Score"].isna().all():
        raise ValueError("Inferred network scores are empty. Ensure there is overlap between \
            the TFs and TGs in the ground truth and inferred networks")
    
    overlap = pd.merge(inferred_network_df, ground_truth_df, on=['Source', 'Target'])
    if not overlap.empty:
        logging.warning(f"{len(overlap)} overlapping edges detected between the inferred network \
            and ground truth. Removing these edges.")
        
        # Remove any overlapping ground truth edges from the inferred network
        inferred_network_df = grn_formatting.remove_ground_truth_edges_from_inferred(ground_truth_df, inferred_network_df)
        
        # Remove any TFs and TGs not found in the ground truth from the inferred network
        inferred_network_df = grn_formatting.remove_tf_tg_not_in_ground_truth(ground_truth_df, inferred_network_df)
        
    
    ground_truth_df_copy = ground_truth_df.copy()
    inferred_network_df_copy = inferred_network_df.copy()
    
    if lower_threshold == None:
        gt_mean = ground_truth_df['Score'].mean()
        gt_std = ground_truth_df['Score'].std()

        # Define the lower threshold
        lower_threshold = gt_mean - 1 * gt_std
    
    # Classify ground truth Score
    ground_truth_df_copy['true_interaction'] = 1
    ground_truth_df_copy['predicted_interaction'] = np.where(
        ground_truth_df_copy['Score'] >= lower_threshold, 1, 0)
    
    if ground_truth_df_copy['predicted_interaction'].empty:
        logging.warning("There are no predicted interactions for the ground truth network with the \
            given lower_threshold")

    # Classify non-ground truth Score (trans_reg_minus_ground_truth_df)
    inferred_network_df_copy['true_interaction'] = 0
    inferred_network_df_copy['predicted_interaction'] = np.where(
        inferred_network_df_copy['Score'] >= lower_threshold, 1, 0)

    if inferred_network_df_copy['predicted_interaction'].empty:
        logging.warning("There are no predicted interactions for the inferred network with the \
            given lower_threshold")
    
    return ground_truth_df_copy, inferred_network_df_copy

def calculate_accuracy_metrics(
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame,
    num_edges_for_early_precision: int = 1000,
    ) -> tuple[dict, dict]:
    
    """
    Calculates accuracy metrics for an inferred network.
    
    Uses a lower threshold as the cutoff between true and false values. The 
    default lower threshold is set as 1 standard deviation below the mean 
    ground truth Score.
    
    True Positive: ground truth Score above the lower threshold
    False Positive: non-ground truth Score above the lower threshold
    True Negative: non-ground truth Score below the lower threshold
    False Negative: ground truth Score below the lower threshold
    
    Parameters
    ----------
        ground_truth_df (pd.DataFrame):
            The ground truth dataframe with inferred network Score
            
        inferred_network_df (pd.DataFrame):
            The inferred GRN dataframe

    Returns
    ----------
        summary_dict (dict):
            A dictionary of the TP, TN, FP, and FN Score along with y_true and y_pred.
                
        confusion_matrix_score_dict (dict):
            A dictionary of the accuracy metrics and their values

    """
    
    required_columns = ["Source", "Target", "Score", "true_interaction", "predicted_interaction"]
    
    # Check to make sure the ground truth dataset has the required columns
    missing_columns_ground_truth = set(required_columns) - set(ground_truth_df.columns)
    if missing_columns_ground_truth:
        raise ValueError(f"Missing required columns in ground_truth_df: {missing_columns_ground_truth}. Columns present: \
            {list(ground_truth_df.columns)}")
    
    # Check to make sure the inferred network has the required columns
    missing_columns_inferred = set(required_columns) - set(inferred_network_df.columns)
    if missing_columns_inferred:
        raise ValueError(f"Missing required columns in inferred_network_df: {missing_columns_inferred}. Columns present: \
            {list(inferred_network_df.columns)}")

    
    # Ensure that the "Score" column exists in both the ground truth and inferred networks
    for col in ["true_interaction", "predicted_interaction"]:
        if ground_truth_df[col].isna().all():
            raise ValueError(f"Ground truth {col} column is completely missing (all NaN). Ensure it is created properly.")
        if inferred_network_df[col].isna().all():
            raise ValueError(f"Inferred network {col} column is completely missing (all NaN). Ensure it is created properly.")

    # Concatenate dataframes for AUC and further analysis
    auc_df = pd.concat([ground_truth_df, inferred_network_df])

    # Calculate the confusion matrix
    y_true = auc_df['true_interaction'].dropna()
    y_pred = auc_df['predicted_interaction'].dropna()
    y_scores = auc_df['Score'].dropna()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculations for accuracy metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    jaccard_index = tp / (tp + fp + fn)

    # Weighted Jaccard Index
    weighted_tp = ground_truth_df.loc[ground_truth_df['predicted_interaction'] == 1, 'Score'].sum()
    weighted_fp = inferred_network_df.loc[inferred_network_df['predicted_interaction'] == 1, 'Score'].sum()
    weighted_fn = ground_truth_df.loc[ground_truth_df['predicted_interaction'] == 0, 'Score'].sum()
    weighted_jaccard_index = weighted_tp / (weighted_tp + weighted_fp + weighted_fn)
    
    # Early Precision Rate for top 1000 predictions
    top_edges = auc_df.nlargest(int(num_edges_for_early_precision), 'Score')
    early_tp = top_edges[top_edges['true_interaction'] == 1].shape[0]
    early_fp = top_edges[top_edges['true_interaction'] == 0].shape[0]
    early_precision_rate = early_tp / (early_tp + early_fp)
    
    accuracy_metric_dict = {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "jaccard_index": jaccard_index,
        "weighted_jaccard_index": weighted_jaccard_index,
        "early_precision_rate": early_precision_rate,
    }
    
    confusion_matrix_score_dict = {
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_scores': y_scores
        }
    
    return accuracy_metric_dict, confusion_matrix_score_dict

def create_randomized_inference_scores(    
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame,
    lower_threshold: int = None,
    num_edges_for_early_precision: int = 1000,
    histogram_save_path: str = None
    ):
    
    """
    Uses random permutation to randomize the edge Score for inferred network and ground truth network dataframes.
    Randomly reshuffles and calculates accuracy metrics from the new Score.

    Parameters
    ----------
        ground_truth_df (pd.DataFrame): 
            The ground truth network dataframe containing columns "Source" and "Target"
        inferred_network_df (pd.DataFrame): 
            The inferred network dataframe containing columns "Source", "Target", and "Score"
        lower_threshold (int, optional): 
            Can optionally provide a lower threshold value. Defaults to 1 stdev below the ground truth mean.
        num_edges_for_early_precision (int, optional): 
            Can optionally provide the number of edges for calculating early precision rate. Defaults to 1000.
        histogram_save_path (str, optional):
            Can optionlly provide a save path to create a histogram of the randomized ground truth GRN scores
            vs. the randomized inferred network scores

    Returns
    ----------
        accuracy_metric_dict (dict):
            A dictionary containing the calculated accuracy metrics. Contains keys:
            "precision"
            "recall"
            "specificity"
            "accuracy"
            "f1_score"
            "jaccard_index"
            "weighted_jaccard_index"
            "early_precision_rate"
        
        confusion_matrix_dict (dict):
            A dictionary containing the results of the confution matrix and metrics for AUROC
            and AUPRC. Contains keys:
            "true_positive"
            "true_negative"
            "false_positive"
            "false_negative"
            "y_true"
            "y_pred"
            "y_scores"
    """
    
    # Create a copy of the inferred network and ground truth dataframe
    inferred_network_df_copy = inferred_network_df.copy()
    ground_truth_df_copy = ground_truth_df.copy()
    
    # Drop any columns that end up being infinity or nan values
    ground_truth_df_copy["Score"] = ground_truth_df_copy["Score"].replace([np.inf, -np.inf], np.nan).dropna()
    inferred_network_df_copy["Score"] = inferred_network_df_copy["Score"].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Ensure there are still scores after filtering
    if ground_truth_df_copy["Score"].empty:
        raise ValueError("Ground truth scores are empty after cleaning.")
    
    if inferred_network_df_copy["Score"].empty:
        raise ValueError("Inferred network scores are empty after cleaning.")
    
    # Extract the Score for each edge
    inferred_network_score = inferred_network_df_copy["Score"]
    ground_truth_score = ground_truth_df_copy["Score"]
    
    # Combine the scores from the ground truth and inferred network
    total_scores = pd.concat(
        [ground_truth_df["Score"], inferred_network_df["Score"]]
    ).values
    
    # Randomly reassign scores back to the ground truth and inferred network
    resampled_inferred_network_scores = np.random.choice(total_scores, size=len(inferred_network_score), replace=True)
    resampled_ground_truth_scores = np.random.choice(total_scores, size=len(ground_truth_score), replace=True)
    
    # Replace the edge Score in the copied dataframe with the resampled Score
    inferred_network_df_copy["Score"] = resampled_inferred_network_scores
    ground_truth_df_copy["Score"] = resampled_ground_truth_scores

    # Calculate the lower threshold of the randomized scores(if not provided)
    if lower_threshold == None:
        randomized_ground_truth_mean = ground_truth_df_copy["Score"].mean()
        randomized_ground_truth_stdev = ground_truth_df_copy["Score"].std()
        
        # Make sure that the mean and standard deviation are not nan
        if np.isnan(randomized_ground_truth_mean) or np.isnan(randomized_ground_truth_stdev):
            raise ValueError("Mean or standard deviation of ground truth scores is NaN.")
        
        # Default lower threshold is 1 stdev below the ground truth mean
        lower_threshold = randomized_ground_truth_mean - 1 * randomized_ground_truth_stdev
    
    # Calculate the true and predicted interactions with the randomized lower threshold
    ground_truth_df_copy, inferred_network_df_copy = classify_interactions_by_threshold(ground_truth_df_copy, inferred_network_df_copy, lower_threshold)
    
    # Recalculate the accuracy metrics for the resampled Score
    randomized_accuracy_metric_dict, randomized_confusion_matrix_dict = calculate_accuracy_metrics(
        ground_truth_df_copy,
        inferred_network_df_copy,
        num_edges_for_early_precision
        )
    
    randomized_inferred_dict = {'Randomized': inferred_network_df_copy}
    randomized_ground_truth_dict = {'Randomized': ground_truth_df_copy}
    
    if histogram_save_path != None:
        
        plotting.plot_multiple_histogram_with_thresholds(randomized_ground_truth_dict, randomized_inferred_dict, histogram_save_path)
    
    return randomized_accuracy_metric_dict, randomized_confusion_matrix_dict

def calculate_auroc(confusion_matrix_score_dict: dict):
    y_true = confusion_matrix_score_dict['y_true']
    y_scores = confusion_matrix_score_dict['y_scores']
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return roc_auc

def calculate_auprc(confusion_matrix_score_dict: dict):
    y_true = confusion_matrix_score_dict['y_true']
    y_scores = confusion_matrix_score_dict['y_scores']
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    prc_auc = auc(recall, precision)
    
    return prc_auc

