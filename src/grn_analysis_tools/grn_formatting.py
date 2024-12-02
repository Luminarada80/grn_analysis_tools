import logging
import pandas as pd
from typing import TextIO
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams
from . import plotting


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')   

# Set font to Arial and adjust font sizes
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,  # General font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 16,  # Axis label font size
    'xtick.labelsize': 14,  # X-axis tick label size
    'ytick.labelsize': 14,  # Y-axis tick label size
    'legend.fontsize': 14  # Legend font size
})

def log_and_write(file: TextIO, message: str) -> None:
    """
    Helper function for logging and writing a message to a file.
    
    Parameters:
        file (TextIO): An open file object where the message will be written.
        message (str): The message to be logged and written to the file.
    """
    logging.debug(message)
    file.write(message + '\n')

def source_target_cols_uppercase(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the Source and Target columns to uppercase for a dataframe
    
    Parameters
    ----------
        df (pd.DataFrame): GRN dataframe with "Source" and "Target" gene name columns

    Returns
    ----------
        pd.DataFrame: The same dataframe with uppercase gene names
    """
    df["Source"] = df["Source"].str.upper()
    df["Target"] = df["Target"].str.upper()

def create_standard_dataframe(
    network_df: pd.DataFrame,
    source_col: str = None,
    target_col: str = None,
    score_col: str = None,
    is_ground_truth: bool = False,
    ) -> pd.DataFrame:
    
    """
    Standardizes inferred GRN dataframes to have three columns with "Source", "Target", and "Score".
    Makes all TF and TG names uppercase.
    
    Parameters
    ----------
        network_df (pd.dataframe):
            Inferred GRN dataframe or ground truth (if ground truth, set 'is_ground_truth' to True)
        
        source_col (str):
            The column name that should be used for the TFs. Default is "Source"
        
        target_col (str):
            The column name that should be used for the TGs. Default is "Target"
        
        score_col (str):
            The column name that should be used as the score. Default is "Score"
    
    Returns
    ----------
        standardized_df (pd.DataFrame):
            A dataframe with three columns: "Source", "Target, and "Score"
    """
    
    # Check to make sure the dataframe is input correctly
    if network_df.empty:
        raise ValueError("The input dataframe is empty. Please provide a valid dataframe.")
    
    # Create a copy of the network_df
    network_df = network_df.copy()

    source_col = source_col.capitalize() if source_col else "Source"
    target_col = target_col.capitalize() if target_col else "Target"
    score_col = score_col.capitalize() if score_col else "Score"

    # Detect if the DataFrame needs to be melted
    if source_col in network_df.columns and target_col in network_df.columns:  
        
        # Assign default Score if missing and is_ground_truth=True
        if score_col not in network_df.columns and is_ground_truth:
            logging.debug(f"Ground truth detected. Assigning default score of 0 to '{score_col}'.")
            network_df[score_col] = 0
            
        elif score_col not in network_df.columns:
            raise ValueError(f"Input dataframe must contain a '{score_col}' column unless 'is_ground_truth' is True.")
        
        # Rename and clean columns
        standardized_df = network_df.rename(columns={source_col: "Source", target_col: "Target", score_col: "Score"})     
        
    
    # The dataframe needs to be melted, there are more than 3 columns and no "Source" or "Target" columns
    elif network_df.shape[1] > 3:
        
        num_rows, num_cols = network_df.shape
        
        logging.debug(f'Original dataframe has {num_rows} rows and {num_cols} columns')
        
        logging.debug(f'\nOld df before melting:')
        logging.debug(network_df.head())     
        # TFs are columns, TGs are rows
        if num_rows >= num_cols:
            logging.debug(f'\t{num_cols} TFs, {num_rows} TGs, and {num_cols * num_rows} edges')
            # Transpose the columns and rows to prepare for melting
            network_df = network_df.T
            
            # Reset the index to make the TFs a column named 'Source'
            network_df = network_df.reset_index()
            network_df = network_df.rename(columns={'index': 'Source'})  # Rename the index column to 'Source'
    
            standardized_df = network_df.melt(id_vars="Source", var_name="Target", value_name="Score")
            
        # TFs are rows, TGs are columns
        elif num_cols > num_rows:
            logging.debug(f'\t{num_rows} TFs, {num_cols} TGs, and {num_cols * num_rows} edges')
            
            # Reset the index to make the TFs a column named 'Source'
            network_df = network_df.reset_index()
            network_df = network_df.rename(columns={'index': 'Source'})  # Rename the index column to 'Source'
    
            standardized_df = network_df.melt(id_vars="Source", var_name="Target", value_name="Score")

    else:
        raise ValueError("Input dataframe must either be in long format with 'Source' and 'Target' columns, "
                         "or matrix format with row and column names representing TGs and TFs.")


    # Final check to make sure the dataframe has the correct targets
    standardized_df = standardized_df[["Source", "Target", "Score"]]
    
    # Validates the structure of the dataframe before returning it
    if not all(col in standardized_df.columns for col in ["Source", "Target", "Score"]):
        raise ValueError("Standardized dataframe does not contain the required columns.")
    
    logging.debug(f"Standardized dataframe created with {len(standardized_df):,} edges.")
    return standardized_df

def add_inferred_scores_to_ground_truth(
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame,
    score_column_name: str = "Score"
    ) -> pd.DataFrame:
    
    """
    Merges the inferred network Score with the ground truth dataframe, returns a ground truth network
    with a column containing the inferred Source -> Target score for each row.
    
    Parameters
    ----------
        ground_truth_df (pd.DataFrame):
            The ground truth dataframe. Columns should be "Source" and "Target"
            
        inferred_network_df (pd.DataFrame):
            The inferred GRN dataframe. Columns should be "Source", "Target", and "Score"
        
        score_column_name (str):
            Renames the "Score" column to a specific name, used if multiple datasets are being compared.
    
    Returns
    ----------
        ground_truth_with_scores (pd.DataFrame):
            Ground truth dataframe with a "Score" column corresonding to the edge Score for each 
            Target Source pair from the inferred network
    
    """
    # Make sure that ground truth and inferred network "Source" and "Target" columns are uppercase
    source_target_cols_uppercase(ground_truth_df)
    source_target_cols_uppercase(inferred_network_df)
    
    # Take the Source and Target from the ground truth and the Score from the inferred network to create a new df
    ground_truth_with_scores = pd.merge(
        ground_truth_df, 
        inferred_network_df[["Source", "Target", "Score"]], 
        left_on=["Source", "Target"], 
        right_on=["Source", "Target"], 
        how="left"
    ).rename(columns={"Score": score_column_name})

    return ground_truth_with_scores

def remove_ground_truth_edges_from_inferred(
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Removes ground truth edges from the inferred network after setting the ground truth Score.
    
    After this step, the inferred network does not contain any ground truth edge Score. This way, the 
    inferred network and ground truth network Score can be compared.
    
    Parameters
    ----------
        ground_truth_df (pd.DataFrame):
            Ground truth df with columns "Source" and "Target" corresponding to TFs and TGs
        
        inferred_network_df (pd.DataFrame):
            The inferred GRN df with columns "Source" and "Target" corresponding to TFs and TGs

    Returns
    ----------
        inferred_network_no_ground_truth (pd.DataFrame):
            The inferred GRN without the ground truth Score
    """
    # Make sure that ground truth and inferred network "Source" and "Target" columns are uppercase
    source_target_cols_uppercase(ground_truth_df)
    source_target_cols_uppercase(inferred_network_df)
    
    # Get a list of the ground truth edges to separate 
    ground_truth_edges = set(zip(ground_truth_df['Source'], ground_truth_df['Target']))
    
    # Create a new dataframe without the ground truth edges
    inferred_network_no_ground_truth = inferred_network_df[
        ~inferred_network_df.apply(lambda row: (row['Source'], row['Target']) in ground_truth_edges, axis=1)
    ]
    
    # Ensure there are no overlaps by checking for any TF-TG pairs that appear in both dataframes
    overlap_check = pd.merge(
        inferred_network_no_ground_truth[['Source', 'Target']], 
        ground_truth_df[['Source', 'Target']], 
        on=['Source', 'Target'], 
        how='inner'
    )

    # Check to make sure there is no overlap between the ground truth and inferred networks
    if overlap_check.shape[0] > 0:
        logging.warning("There are still ground truth pairs in trans_reg_minus_ground_truth_df!")
    
    return inferred_network_no_ground_truth

def remove_tf_tg_not_in_ground_truth(
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame
    ) -> pd.DataFrame:
    
    """
    Removes TFs and TGs from the inferred network where either the "Source" or "Target" gene is not
    found in the ground truth network.
    
    This allows for the evaluation of the GRN inference method by removing TFs and TGs that could
    not be found in the ground truth. The inferred edges may exist, but they will be evaluated as 
    incorrect if they are not seen in the ground truth network. This way, we can determine if an 
    inferred edge is true or false by its presence in the ground truth network and inferred edge score.
    
    Parameters
    ----------
        ground_truth_df (pd.DataFrame):
            Ground truth df with columns "Source" and "Target" corresponding to TFs and TGs
        
        inferred_network_df (pd.DataFrame):
            The inferred GRN df with columns "Source" and "Target" corresponding to TFs and TGs
    
    Returns
    ----------
        aligned_inferred_network (pd.DataFrame):
            Returns the inferred network containing only rows with only edges where both the "Source"
            and "Target" genes are found in the ground truth network.
    
    """
    # Make sure that ground truth and inferred network "Source" and "Target" columns are uppercase
    source_target_cols_uppercase(ground_truth_df)
    source_target_cols_uppercase(inferred_network_df)
    
    # Extract unique TFs and TGs from the ground truth network
    ground_truth_tfs = set(ground_truth_df['Source'])
    ground_truth_tgs = set(ground_truth_df['Target'])
    
    # Subset cell_oracle_network to contain only rows with TFs and TGs in the ground_truth
    aligned_inferred_network = inferred_network_df[
        (inferred_network_df['Source'].isin(ground_truth_tfs)) &
        (inferred_network_df['Target'].isin(ground_truth_tgs))
    ]
    
    return aligned_inferred_network
