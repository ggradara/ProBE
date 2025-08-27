import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import sys
import os
import argparse
from datetime import datetime

start_time = time.time()


def log_list(list_to_log, filename, message=""):
    """
    Appends a list to a text file for debugging, with an optional message.
    This function opens the file in 'append' mode, so it will not
    overwrite previous logs in the same file.

    Args:
        list_to_log (list): The list you want to write to the file.
        filename (str): The path to the output log file.
        message (str, optional): A descriptive message to print before the list.
                                 Helps identify the log's context.
    """
    # Use 'a' for append mode. This adds to the file instead of overwriting it.
    # The 'with' statement ensures the file is properly closed.
    with open(filename, 'a') as f:
        # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Write the optional message as a header to know what you're looking at
        if message:
            f.write(f"--- {message} ---\n")
        else:
            f.write(f"--- --- ---\n")

        # Write each item from the list on its own line
        # Using str(item) is a safe way to handle lists with mixed types (numbers, etc.)
        if list_to_log:
            # for item in list_to_log:
            #     f.write(f"{str(item)}\n")
            f.write(f"{str(list_to_log)}\n")
        else:
            f.write("<< The list was empty or None >>\n")

        # Add a blank line for better separation between log entries
        f.write("\n")



def compute_benchmark_by_cafa(path, model, cafa, steps, current_datetime, dir_tree):

    tool_name = model
    
    if cafa == "C4":
        cafa_column = "CAFA4_ID"
    else:
        cafa_column = "Query_ID"
        

    min_thresh = 0
    max_thresh = 1
    thresholds = np.arange(min_thresh, max_thresh, steps)
    
    tool_predictions_address = path
    
    aspects = ["M", "C", "P"]
    precision_across_aspects_NK = []
    recall_across_aspects_NK = []
    FDR_across_aspects_NK = []
    F1score_across_aspects_NK = []
    gp_across_aspects_NK = []

    precision_across_aspects_LK = []
    recall_across_aspects_LK = []
    FDR_across_aspects_LK = []
    F1score_across_aspects_LK = []
    gp_across_aspects_LK = []
    
    for aspect in aspects:

        tool_predictions_full = pd.read_csv(tool_predictions_address, sep="\t")
        
        tool_predictions = tool_predictions_full[tool_predictions_full["Ontology"] == aspect]
        tool_predictions["GO_ID"] = tool_predictions["GO_ID"].str.replace("_", ":") 
        
        # Read the file and get the dataframe
        knowledge_type = "NK"   # NK or LK
        ground_truth_address = dir_tree['gt_dir_path'] + f"/ground_truth_{knowledge_type}_{cafa}.tsv"
        ground_truth_full = pd.read_csv(ground_truth_address, sep="\t") 
        
        # Sort alphabetically (the algorithm expects alphabetical order)
        ground_truth = ground_truth_full[ground_truth_full["Aspect"] == aspect]
        
        ground_truth = ground_truth.sort_values(by=[cafa_column, "GO_ID"], ascending=[True, True])
        gt_grouped = ground_truth.groupby(cafa_column)

        tot_recall_NK = []
        tot_precision_NK = []
        tot_F1score_NK = []
        tot_FDR_NK = []


        print(f"Benchmark evaluator for {tool_name} in {cafa}")

        # NK cycle
        for threshold in tqdm(thresholds, desc=f"Processing NK thresholds for aspect {aspect}"):
            total_TP_NK = 0
            total_FP_NK = 0
            total_FN_NK = 0
            total_P_NK = 0
            gp_no_pred_NK = 0  # GPs with no predictions
            
            tp_thresh = tool_predictions[tool_predictions["Score"]>threshold]
            tp_grouped = tp_thresh.groupby('Query_ID')
            
            for group_id, group_gt in gt_grouped:  # Get a group of rows selected by ID
                if group_id in tp_grouped.groups:  # Ensure the ID exists in both DataFrames
                    group_tp = tp_grouped.get_group(group_id)  # Retrieve predictions regarding the specific ID
                    GO_pred = group_tp['GO_ID']       
                else:  # If not behave as if there are no predictions
                    GO_pred = []
                    gp_no_pred_NK += 1
                    
                GO_pred = set(GO_pred)
                GO_gt = group_gt['GO_ID']
                GO_gt = set(GO_gt)
                
                TP_predictions = list(GO_pred & GO_gt)  # The TP predictions are the shared ones
                FP_predictions = list(GO_pred - GO_gt)  # Predictions that are not in the gt
                FN_predictions = list(GO_gt - GO_pred)  # Predictions missed from the gt
                
                total_TP_NK += len(TP_predictions)
                total_FP_NK += len(FP_predictions)
                total_FN_NK += len(FN_predictions)
                total_P_NK += len(GO_gt)
                    
            if (total_TP_NK + total_FP_NK) == 0:  # 0 div check precision
                precision = 0
            else:
                precision = total_TP_NK/(total_TP_NK + total_FP_NK)
            if total_P_NK == 0:  # 0 div check recall
                recall = 0
            else:
                recall = total_TP_NK/total_P_NK  
            if (precision + recall) == 0:  # 0 div check F1 score
                F1_score = 0
            else:
                F1_score = 2*(precision * recall)/(precision + recall)
            if (total_TP_NK + total_FP_NK) == 0: # 0 div check FDR
                FDR = 0
            else:
                FDR = total_FP_NK / (total_TP_NK + total_FP_NK)
            
            tot_precision_NK.append(precision)
            tot_recall_NK.append(recall)

            tot_F1score_NK.append(F1_score)
            tot_FDR_NK.append(FDR)

        NK_time = time.time()
        elapsed_time = NK_time - start_time  # End timer NK
        print(f"Elapsed time: {elapsed_time:.3f} seconds to process NK cycle")
        
        print(f"Max precision {knowledge_type}: {max(tot_precision_NK)}")
        print(f"Max recall {knowledge_type}: {max(tot_recall_NK)}")
        print(f"Max F1score {knowledge_type}: {max(tot_F1score_NK)}")
        print(f"Max FDR {knowledge_type}: {max(tot_FDR_NK)}")
        print(f"GP with no predictions: {gp_no_pred_NK}")
        
        
        precision_across_aspects_NK.append(tot_precision_NK)
        recall_across_aspects_NK.append(tot_recall_NK)
        FDR_across_aspects_NK.append(tot_FDR_NK)
        F1score_across_aspects_NK.append(tot_F1score_NK)
        gp_across_aspects_NK.append(gp_no_pred_NK)


        # Read the file and get the dataframe
        knowledge_type = "LK"   # NK or LK
        # if knowledge_type == "LK":
        #     knowledge_type = "KK_LK"
        ground_truth_address = dir_tree['gt_dir_path'] + f"/ground_truth_LK_KK_{cafa}.tsv"
        ground_truth_full = pd.read_csv(ground_truth_address, sep="\t") 
        ground_truth = ground_truth_full[ground_truth_full["Aspect"] == aspect]
        # Sort alphabetically (the algorithm expects alphabetical order)
        ground_truth = ground_truth.sort_values(by=[cafa_column, "GO_ID"], ascending=[True, True])
        gt_grouped = ground_truth.groupby(cafa_column)
        # tp_grouped = tool_predictions.groupby('Query_ID')

        tot_recall_LK = []
        tot_precision_LK = []
        tot_F1score_LK = []
        tot_FDR_LK = []
        

        print(f"Benchmark evaluator for {tool_name} in {cafa}")

        # LK cycle
        for threshold in tqdm(thresholds, desc=f"Processing LK thresholds for aspect {aspect}"):
            total_TP_LK = 0
            total_FP_LK = 0
            total_FN_LK = 0
            total_P_LK = 0
            gp_no_pred_LK = 0  # GPs with no predictions
            
            # tp_grouped_thresh = tp_grouped[tp_grouped["Score"]>threshold]
            tp_thresh = tool_predictions[tool_predictions["Score"]>threshold]
            tp_grouped = tp_thresh.groupby('Query_ID')
            
            for group_id, group_gt in gt_grouped:  # Get a group of rows selected by ID
                if group_id in tp_grouped.groups:  # Ensure the ID exists in both DataFrames
                    group_tp = tp_grouped.get_group(group_id)  # Retrieve predictions regarding the specific ID
                    GO_pred = group_tp['GO_ID']       
                else:  # If not behave as if there are no predictions
                    GO_pred = []
                    gp_no_pred_LK += 1
                    
                GO_pred = set(GO_pred)
                GO_gt = group_gt['GO_ID']
                GO_gt = set(GO_gt)
                
                # Salviamo tutti questi dati in un dizionario per ogni gp? (magari con la threshold come chiave)
                TP_predictions = list(GO_pred & GO_gt)  # The TP predictions are the shared ones
                FP_predictions = list(GO_pred - GO_gt)  # Predictions that are not in the gt
                FN_predictions = list(GO_gt - GO_pred)  # Predictions missed from the gt
                
                total_TP_LK += len(TP_predictions)
                total_FP_LK += len(FP_predictions)
                total_FN_LK += len(FN_predictions)
                total_P_LK += len(GO_gt)
                    
            if (total_TP_LK + total_FP_LK) == 0:  # 0 div check precision
                precision = 0
            else:
                precision = total_TP_LK/(total_TP_LK + total_FP_LK)
            if total_P_LK == 0:  # 0 div check recall
                recall = 0
            else:
                recall = total_TP_LK/total_P_LK  

            if (precision + recall) == 0:  # 0 div check F1 score
                F1_score = 0
            else:
                F1_score = 2*(precision * recall)/(precision + recall)
            if (total_TP_LK + total_FP_LK) == 0: # 0 div check FDR
                FDR = 0
            else:
                FDR = total_FP_LK / (total_TP_LK + total_FP_LK)
            
            # Dovrebbero essere dizionari ma per la fase di prototipizzazione saranno delle liste
            tot_precision_LK.append(precision)
            tot_recall_LK.append(recall)
            #### DEBUGGING
            # debugging_TP_LK.append(total_TP_LK)
            # debugging_P_LK.append(total_P_LK)
            ###
            tot_F1score_LK.append(F1_score)
            tot_FDR_LK.append(FDR)

        LK_time = time.time()
        elapsed_time = LK_time - NK_time  # End timer LK
        tot_elapsed_time = LK_time - start_time
        print(f"Elapsed time: {elapsed_time:.3f} seconds to process LK cycle, {tot_elapsed_time:.3f} in total")
        
        print(f"Max precision {knowledge_type} for aspect {aspect}: {max(tot_precision_LK)}")
        print(f"Max recall {knowledge_type} for aspect {aspect}: {max(tot_recall_LK)}")
        print(tot_recall_LK)
        print(f"Max F1score {knowledge_type} for aspect {aspect}: {max(tot_F1score_LK)}")
        print(f"Max FDR {knowledge_type} for aspect {aspect}: {max(tot_FDR_LK)}")
        
        precision_across_aspects_LK.append(tot_precision_LK)
        recall_across_aspects_LK.append(tot_recall_LK)
        FDR_across_aspects_LK.append(tot_FDR_LK)
        F1score_across_aspects_LK.append(tot_F1score_LK)
        gp_across_aspects_LK.append(gp_no_pred_LK)
        
        
        titles = [f"Precision  for {tool_name}", 
                f"Recall  for {tool_name}", 
                f"FDR  for {tool_name}", 
                f"F1Score  for {tool_name}"
        ]


        NK_dir_name = f"NK_by_aspect_{tool_name}_{current_datetime}"
        NK_path = os.path.join(dir_tree['btp_dir_path'] + f"/benchmark_{cafa}/by_aspect/NK_by_aspect", NK_dir_name)
        if not os.path.exists(NK_path):
            os.makedirs(NK_path)
            
        NK_dir_name_o4e = f"NK_one_for_each_{tool_name}_{current_datetime}"
        NK_path_o4e = os.path.join(NK_path, NK_dir_name_o4e)
        if not os.path.exists(NK_path_o4e):
            os.makedirs(NK_path_o4e)


        fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
        axes = axes.flatten()  # Flatten the 2D array for easy iteration
        y_data = [tot_precision_NK, tot_recall_NK, tot_FDR_NK, tot_F1score_NK]
        # Loop through each subplot
        for i, ax in enumerate(axes):
            y = y_data[i]  # Select y data
            max_idx = np.argmax(y)  # Get index of max value
            max_x = thresholds[max_idx]  # X value of max
            max_y = y[max_idx]  # Y value of max

            ax.plot(thresholds, y)
            ax.scatter(max_x, max_y, color='red', s=100, label=f"{max_y:.5f}")  # Red dot for max
            ax.set_title(titles[i])  # Set title
            ax.legend()
            # ax.grid(True)
        fig.suptitle(f'NK benchmark for {tool_name} in {cafa} for aspect {aspect}', fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"{NK_path_o4e}/NK_benchmark_{tool_name}_{aspect}_{cafa}.png", format="png")
        
        
        LK_dir_name = f"LK_by_aspect_{tool_name}_{current_datetime}"
        LK_path = os.path.join(dir_tree['btp_dir_path'] + f"/benchmark_{cafa}/by_aspect/LK_by_aspect", LK_dir_name)
        if not os.path.exists(LK_path):
            os.makedirs(LK_path)
            
        LK_dir_name_o4e = f"LK_one_for_each_{tool_name}_{current_datetime}"
        LK_path_o4e = os.path.join(LK_path, LK_dir_name_o4e)
        if not os.path.exists(LK_path_o4e):
            os.makedirs(LK_path_o4e)

        fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
        axes = axes.flatten()  # Flatten the 2D array for easy iteration
        y_data = [tot_precision_LK, tot_recall_LK, tot_FDR_LK, tot_F1score_LK]
        # Loop through each subplot
        for i, ax in enumerate(axes):
            y = y_data[i]  # Select y data
            max_idx = np.argmax(y)  # Get index of max value
            max_x = thresholds[max_idx]  # X value of max
            max_y = y[max_idx]  # Y value of max

            ax.plot(thresholds, y)
            ax.scatter(max_x, max_y, color='red', s=100, label=f"{max_y:.5f}")  # Red dot for max
            ax.set_title(titles[i])  # Set title
            ax.legend()
            # ax.grid(True)
        fig.suptitle(f'LK benchmark for {tool_name} in {cafa}', fontsize=16)
        plt.subplots_adjust(top=0.9)  # Increase the top margin
        plt.savefig(f"{LK_path_o4e}/LK_benchmark_{tool_name}_{aspect}_{cafa}.png", format="png")


    # NK graph
    curve_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    highlight_colors = ["#004c99", "#800000", "#005c00"]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array for easy iteration
    y_data_1 = [precision_across_aspects_NK[0], recall_across_aspects_NK[0], FDR_across_aspects_NK[0], F1score_across_aspects_NK[0]]
    y_data_2 = [precision_across_aspects_NK[1], recall_across_aspects_NK[1], FDR_across_aspects_NK[1], F1score_across_aspects_NK[1]]
    y_data_3 = [precision_across_aspects_NK[2], recall_across_aspects_NK[2], FDR_across_aspects_NK[2], F1score_across_aspects_NK[2]]
    label = [f"Aspect M NK", f"Aspect C NK", f"Aspect P NK"]
    # Loop through each subplot
    for i, ax in enumerate(axes):
        # Get data for both curves
        y1 = y_data_1[i]
        y2 = y_data_2[i]
        y3 = y_data_3[i]

        # Find max values
        max_idx_1 = np.argmax(y1)
        max_x_1, max_y_1 = thresholds[max_idx_1], y1[max_idx_1]

        max_idx_2 = np.argmax(y2)
        max_x_2, max_y_2 = thresholds[max_idx_2], y2[max_idx_2]
        
        max_idx_3 = np.argmax(y3)
        max_x_3, max_y_3 = thresholds[max_idx_3], y3[max_idx_3]

        # Plot both curves
        ax.plot(thresholds, y1, label=label[0], color=curve_colors[0])
        ax.plot(thresholds, y2, label=label[1], color=curve_colors[1])
        ax.plot(thresholds, y3, label=label[2], color=curve_colors[2])

        # Highlight max values with different colors
        ax.scatter(max_x_1, max_y_1, color=highlight_colors[0], s=70,  label=f"{max_y_1:.5f}")
        ax.scatter(max_x_2, max_y_2, color=highlight_colors[1], s=70,  label=f"{max_y_2:.5f}")
        ax.scatter(max_x_3, max_y_3, color=highlight_colors[2], s=70,  label=f"{max_y_3:.5f}")
        
        # Set title, legend, and grid
        ax.set_title(titles[i])
        ax.legend()
        # ax.grid(True)
        
    # Adjust layout
    fig.suptitle(f'Comprehensive benchmark for {tool_name} in NK {cafa}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Increase the top margin
    plt.savefig(f"{NK_path}/benchmark_by_aspect_{tool_name}_{cafa}_NK.png", format="png")

    # no data version
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array for easy iteration
    y_data_1 = [precision_across_aspects_NK[0], recall_across_aspects_NK[0], FDR_across_aspects_NK[0], F1score_across_aspects_NK[0]]
    y_data_2 = [precision_across_aspects_NK[1], recall_across_aspects_NK[1], FDR_across_aspects_NK[1], F1score_across_aspects_NK[1]]
    y_data_3 = [precision_across_aspects_NK[2], recall_across_aspects_NK[2], FDR_across_aspects_NK[2], F1score_across_aspects_NK[2]]
    label = [f"Aspect M NK", f"Aspect C NK", f"Aspect P NK"]
    # Loop through each subplot
    for i, ax in enumerate(axes):
        # Get data for both curves
        y1 = y_data_1[i]
        y2 = y_data_2[i]
        y3 = y_data_3[i]

        # Find max values
        max_idx_1 = np.argmax(y1)
        max_x_1, max_y_1 = thresholds[max_idx_1], y1[max_idx_1]

        max_idx_2 = np.argmax(y2)
        max_x_2, max_y_2 = thresholds[max_idx_2], y2[max_idx_2]
        
        max_idx_3 = np.argmax(y3)
        max_x_3, max_y_3 = thresholds[max_idx_3], y3[max_idx_3]

        # Plot both curves
        ax.plot(thresholds, y1, label=label[0], color=curve_colors[0])
        ax.plot(thresholds, y2, label=label[1], color=curve_colors[1])
        ax.plot(thresholds, y3, label=label[2], color=curve_colors[2])

        # Highlight max values with different colors
        ax.scatter(max_x_1, max_y_1, color=highlight_colors[0], s=70)
        ax.scatter(max_x_2, max_y_2, color=highlight_colors[1], s=70)
        ax.scatter(max_x_3, max_y_3, color=highlight_colors[2], s=70)
        
        # Set title, legend, and grid
        ax.set_title(titles[i])
        ax.legend()
        # ax.grid(True)
        
    # Adjust layout
    fig.suptitle(f'Comprehensive benchmark for {tool_name} in NK {cafa}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Increase the top margin
    plt.savefig(f"{NK_path}/benchmark_by_aspect_{tool_name}_{cafa}_NK_nodata.png", format="png")
    
    
    # LK Graph
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array for easy iteration
    y_data_1 = [precision_across_aspects_LK[0], recall_across_aspects_LK[0], FDR_across_aspects_LK[0], F1score_across_aspects_LK[0]]
    y_data_2 = [precision_across_aspects_LK[1], recall_across_aspects_LK[1], FDR_across_aspects_LK[1], F1score_across_aspects_LK[1]]
    y_data_3 = [precision_across_aspects_LK[2], recall_across_aspects_LK[2], FDR_across_aspects_LK[2], F1score_across_aspects_LK[2]]
    label = [f"Aspect M LK", f"Aspect C LK", f"Aspect P LK",]
    # Loop through each subplot
    for i, ax in enumerate(axes):
        # Get data for both curves
        y1 = y_data_1[i]
        y2 = y_data_2[i]
        y3 = y_data_3[i]

        # Find max values
        max_idx_1 = np.argmax(y1)
        max_x_1, max_y_1 = thresholds[max_idx_1], y1[max_idx_1]

        max_idx_2 = np.argmax(y2)
        max_x_2, max_y_2 = thresholds[max_idx_2], y2[max_idx_2]
        
        max_idx_3 = np.argmax(y3)
        max_x_3, max_y_3 = thresholds[max_idx_3], y3[max_idx_3]

        # Plot both curves
        ax.plot(thresholds, y1, label=label[0], color=curve_colors[0])
        ax.plot(thresholds, y2, label=label[1], color=curve_colors[1])
        ax.plot(thresholds, y3, label=label[2], color=curve_colors[2])

        # Highlight max values with different colors
        ax.scatter(max_x_1, max_y_1, color=highlight_colors[0], s=70,  label=f"{max_y_1:.5f}")
        ax.scatter(max_x_2, max_y_2, color=highlight_colors[1], s=70,  label=f"{max_y_2:.5f}")
        ax.scatter(max_x_3, max_y_3, color=highlight_colors[2], s=70,  label=f"{max_y_3:.5f}")
        
        # Set title, legend, and grid
        ax.set_title(titles[i])
        ax.legend()
        # ax.grid(True)
    # Adjust layout
    fig.suptitle(f'Comprehensive benchmark for {tool_name} in LK {cafa}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Increase the top margin
    plt.savefig(f"{LK_path}/benchmark_by_aspect_{tool_name}_{cafa}_LK.png", format="png")
    

    # No data version
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array for easy iteration
    y_data_1 = [precision_across_aspects_LK[0], recall_across_aspects_LK[0], FDR_across_aspects_LK[0], F1score_across_aspects_LK[0]]
    y_data_2 = [precision_across_aspects_LK[1], recall_across_aspects_LK[1], FDR_across_aspects_LK[1], F1score_across_aspects_LK[1]]
    y_data_3 = [precision_across_aspects_LK[2], recall_across_aspects_LK[2], FDR_across_aspects_LK[2], F1score_across_aspects_LK[2]]
    label = [f"Aspect M LK", f"Aspect C LK", f"Aspect P LK",]
    # Loop through each subplot
    for i, ax in enumerate(axes):
        # Get data for both curves
        y1 = y_data_1[i]
        y2 = y_data_2[i]
        y3 = y_data_3[i]

        # Find max values
        max_idx_1 = np.argmax(y1)
        max_x_1, max_y_1 = thresholds[max_idx_1], y1[max_idx_1]

        max_idx_2 = np.argmax(y2)
        max_x_2, max_y_2 = thresholds[max_idx_2], y2[max_idx_2]
        
        max_idx_3 = np.argmax(y3)
        max_x_3, max_y_3 = thresholds[max_idx_3], y3[max_idx_3]

        # Plot both curves
        ax.plot(thresholds, y1, label=label[0], color=curve_colors[0])
        ax.plot(thresholds, y2, label=label[1], color=curve_colors[1])
        ax.plot(thresholds, y3, label=label[2], color=curve_colors[2])

        # Highlight max values with different colors
        ax.scatter(max_x_1, max_y_1, color=highlight_colors[0], s=70)
        ax.scatter(max_x_2, max_y_2, color=highlight_colors[1], s=70)
        ax.scatter(max_x_3, max_y_3, color=highlight_colors[2], s=70)
        
        # Set title, legend, and grid
        ax.set_title(titles[i])
        ax.legend()
        # ax.grid(True)
    # Adjust layout
    fig.suptitle(f'Comprehensive benchmark for {tool_name} in LK {cafa}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Increase the top margin
    plt.savefig(f"{LK_path}/benchmark_by_aspect_{tool_name}_{cafa}_LK_nodata.png", format="png")



    comp_dir_name = f"comp_by_aspect_{tool_name}_{current_datetime}"
    comp_path = os.path.join(dir_tree['btp_dir_path'] + f"/benchmark_{cafa}/by_aspect/comprehensive_by_aspect", comp_dir_name)
    if not os.path.exists(comp_path):
        os.makedirs(comp_path)
    
    
    # curve_colors_6 = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    # highlight_colors_6 = ["#004c99", "#800000", "#005c00", "#cc5500", "#5d2b7a", "#5a2e1f"]

    curve_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    highlight_colors = ["#004c99", "#800000", "#005c00"]
    line_styles = ["-", "--"]
    markers = ["o", "^"] 
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array for easy iteration
    # NK curves
    y_data_1 = [precision_across_aspects_NK[0], recall_across_aspects_NK[0], FDR_across_aspects_NK[0], F1score_across_aspects_NK[0]]
    y_data_2 = [precision_across_aspects_NK[1], recall_across_aspects_NK[1], FDR_across_aspects_NK[1], F1score_across_aspects_NK[1]]
    y_data_3 = [precision_across_aspects_NK[2], recall_across_aspects_NK[2], FDR_across_aspects_NK[2], F1score_across_aspects_NK[2]]
    # LK curves
    y_data_4 = [precision_across_aspects_LK[0], recall_across_aspects_LK[0], FDR_across_aspects_LK[0], F1score_across_aspects_LK[0]]
    y_data_5 = [precision_across_aspects_LK[1], recall_across_aspects_LK[1], FDR_across_aspects_LK[1], F1score_across_aspects_LK[1]]
    y_data_6 = [precision_across_aspects_LK[2], recall_across_aspects_LK[2], FDR_across_aspects_LK[2], F1score_across_aspects_LK[2]]
    label = [f"Aspect M NK", f"Aspect C NK", f"Aspect P NK", f"Aspect M LK", f"Aspect C LK", f"Aspect P LK"]
    # Loop through each subplot
    for i, ax in enumerate(axes):
        # Get data for both curves
        y1 = y_data_1[i]
        y2 = y_data_2[i]
        y3 = y_data_3[i]
        y4 = y_data_4[i]
        y5 = y_data_5[i]
        y6 = y_data_6[i]

        # Find max values
        max_idx_1 = np.argmax(y1)
        max_x_1, max_y_1 = thresholds[max_idx_1], y1[max_idx_1]

        max_idx_2 = np.argmax(y2)
        max_x_2, max_y_2 = thresholds[max_idx_2], y2[max_idx_2]

        max_idx_3 = np.argmax(y3)
        max_x_3, max_y_3 = thresholds[max_idx_3], y3[max_idx_3]

        max_idx_4 = np.argmax(y4)
        max_x_4, max_y_4 = thresholds[max_idx_4], y4[max_idx_4]

        max_idx_5 = np.argmax(y5)
        max_x_5, max_y_5 = thresholds[max_idx_5], y5[max_idx_5]

        max_idx_6 = np.argmax(y6)
        max_x_6, max_y_6 = thresholds[max_idx_6], y6[max_idx_6]

        # Plot both curves
        ax.plot(thresholds, y1, label=label[0], color=curve_colors[0], linestyle=line_styles[0])
        ax.plot(thresholds, y2, label=label[1], color=curve_colors[1], linestyle=line_styles[0])
        ax.plot(thresholds, y3, label=label[2], color=curve_colors[2], linestyle=line_styles[0])
        ax.plot(thresholds, y4, label=label[3], color=curve_colors[0], linestyle=line_styles[1])
        ax.plot(thresholds, y5, label=label[4], color=curve_colors[1], linestyle=line_styles[1])
        ax.plot(thresholds, y6, label=label[5], color=curve_colors[2], linestyle=line_styles[1])

        # Highlight max values with different colors
        ax.scatter(max_x_1, max_y_1, color=highlight_colors[0], s=60,  label=f"{max_y_1:.5f}", marker=markers[0])
        ax.scatter(max_x_2, max_y_2, color=highlight_colors[1], s=60,  label=f"{max_y_2:.5f}", marker=markers[0])
        ax.scatter(max_x_3, max_y_3, color=highlight_colors[2], s=60,  label=f"{max_y_3:.5f}", marker=markers[0])
        ax.scatter(max_x_4, max_y_4, color=highlight_colors[0], s=60,  label=f"{max_y_4:.5f}", marker=markers[1])
        ax.scatter(max_x_5, max_y_5, color=highlight_colors[1], s=60,  label=f"{max_y_5:.5f}", marker=markers[1])
        ax.scatter(max_x_6, max_y_6, color=highlight_colors[2], s=60,  label=f"{max_y_6:.5f}", marker=markers[1])

        # Set title, legend, and grid
        ax.set_title(titles[i])
        ax.legend()
        # ax.grid(True)
    # Adjust layout
    fig.suptitle(f'Comprehensive benchmark for {tool_name} in {cafa}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Increase the top margin
    plt.savefig(f"{comp_path}/comprehensive_benchmark_by_aspect_{tool_name}_{cafa}.png", format="png")
        
    # No data version
    curve_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    highlight_colors = ["#004c99", "#800000", "#005c00"]
    line_styles = ["-", "--"]
    markers = ["o", "^"] 
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array for easy iteration
    # NK curves
    y_data_1 = [precision_across_aspects_NK[0], recall_across_aspects_NK[0], FDR_across_aspects_NK[0], F1score_across_aspects_NK[0]]
    y_data_2 = [precision_across_aspects_NK[1], recall_across_aspects_NK[1], FDR_across_aspects_NK[1], F1score_across_aspects_NK[1]]
    y_data_3 = [precision_across_aspects_NK[2], recall_across_aspects_NK[2], FDR_across_aspects_NK[2], F1score_across_aspects_NK[2]]
    # LK curves
    y_data_4 = [precision_across_aspects_LK[0], recall_across_aspects_LK[0], FDR_across_aspects_LK[0], F1score_across_aspects_LK[0]]
    y_data_5 = [precision_across_aspects_LK[1], recall_across_aspects_LK[1], FDR_across_aspects_LK[1], F1score_across_aspects_LK[1]]
    y_data_6 = [precision_across_aspects_LK[2], recall_across_aspects_LK[2], FDR_across_aspects_LK[2], F1score_across_aspects_LK[2]]
    label = [f"Aspect M NK", f"Aspect C NK", f"Aspect P NK", f"Aspect M LK", f"Aspect C LK", f"Aspect P LK"]
    # Loop through each subplot
    for i, ax in enumerate(axes):
        # Get data for both curves
        y1 = y_data_1[i]
        y2 = y_data_2[i]
        y3 = y_data_3[i]
        y4 = y_data_4[i]
        y5 = y_data_5[i]
        y6 = y_data_6[i]

        # Find max values
        max_idx_1 = np.argmax(y1)
        max_x_1, max_y_1 = thresholds[max_idx_1], y1[max_idx_1]

        max_idx_2 = np.argmax(y2)
        max_x_2, max_y_2 = thresholds[max_idx_2], y2[max_idx_2]

        max_idx_3 = np.argmax(y3)
        max_x_3, max_y_3 = thresholds[max_idx_3], y3[max_idx_3]

        max_idx_4 = np.argmax(y4)
        max_x_4, max_y_4 = thresholds[max_idx_4], y4[max_idx_4]

        max_idx_5 = np.argmax(y5)
        max_x_5, max_y_5 = thresholds[max_idx_5], y5[max_idx_5]

        max_idx_6 = np.argmax(y6)
        max_x_6, max_y_6 = thresholds[max_idx_6], y6[max_idx_6]

        # Plot both curves
        ax.plot(thresholds, y1, label=label[0], color=curve_colors[0], linestyle=line_styles[0])
        ax.plot(thresholds, y2, label=label[1], color=curve_colors[1], linestyle=line_styles[0])
        ax.plot(thresholds, y3, label=label[2], color=curve_colors[2], linestyle=line_styles[0])
        ax.plot(thresholds, y4, label=label[3], color=curve_colors[0], linestyle=line_styles[1])
        ax.plot(thresholds, y5, label=label[4], color=curve_colors[1], linestyle=line_styles[1])
        ax.plot(thresholds, y6, label=label[5], color=curve_colors[2], linestyle=line_styles[1])

        # Highlight max values with different colors
        ax.scatter(max_x_1, max_y_1, color=highlight_colors[0], s=60,  marker=markers[0])
        ax.scatter(max_x_2, max_y_2, color=highlight_colors[1], s=60,  marker=markers[0])
        ax.scatter(max_x_3, max_y_3, color=highlight_colors[2], s=60,  marker=markers[0])
        ax.scatter(max_x_4, max_y_4, color=highlight_colors[0], s=60,  marker=markers[1])
        ax.scatter(max_x_5, max_y_5, color=highlight_colors[1], s=60,  marker=markers[1])
        ax.scatter(max_x_6, max_y_6, color=highlight_colors[2], s=60,  marker=markers[1])

        # Set title, legend, and grid
        ax.set_title(titles[i])
        ax.legend()
        # ax.grid(True)
    # Adjust layout
    fig.suptitle(f'Comprehensive benchmark for {tool_name} in {cafa}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Increase the top margin
    plt.savefig(f"{comp_path}/comprehensive_benchmark_by_aspect_{tool_name}_{cafa}_nodata.png", format="png")
    
    
    report_dir_name = f"report_by_aspect_{tool_name}_{current_datetime}"
    report_path = os.path.join(dir_tree['btp_dir_path'] + f"/benchmark_{cafa}/by_aspect/reports_by_aspect", report_dir_name)
    if not os.path.exists(report_path):
        os.makedirs(report_path)
    
    report_name = f"{report_path}/report_{tool_name}_{cafa}_by_aspect.txt"
    with open(report_name, "w") as file:

        file.write(f"Max precision NK aspect M: {max(precision_across_aspects_NK[0])}\n")
        file.write(f"Max precision NK aspect C: {max(precision_across_aspects_NK[1])}\n")
        file.write(f"Max precision NK aspect P: {max(precision_across_aspects_NK[2])}\n")
        file.write(f"Max recall NK aspect M: {max(recall_across_aspects_NK[0])}\n")
        file.write(f"Max recall NK aspect C: {max(recall_across_aspects_NK[1])}\n")
        file.write(f"Max recall NK aspect P: {max(recall_across_aspects_NK[2])}\n")
        file.write(f"Max F1score NK aspect M: {max(F1score_across_aspects_NK[0])}\n")
        file.write(f"Max F1score NK aspect C: {max(F1score_across_aspects_NK[1])}\n")
        file.write(f"Max F1score NK aspect P: {max(F1score_across_aspects_NK[2])}\n")
        file.write(f"Max FDR NK aspect M: {max(FDR_across_aspects_NK[0])}\n")
        file.write(f"Max FDR NK aspect C: {max(FDR_across_aspects_NK[1])}\n")
        file.write(f"Max FDR NK aspect P: {max(FDR_across_aspects_NK[2])}\n")
        file.write(f"GPs with no predictions NK aspect M: {gp_across_aspects_NK[0]}\n\n") 
        file.write(f"GPs with no predictions NK aspect C: {gp_across_aspects_NK[1]}\n\n")
        file.write(f"GPs with no predictions NK aspect P: {gp_across_aspects_NK[2]}\n\n")
        
        file.write(f"Max precision LK aspect M: {max(precision_across_aspects_LK[0])}\n")
        file.write(f"Max precision LK aspect C: {max(precision_across_aspects_LK[1])}\n")
        file.write(f"Max precision LK aspect P: {max(precision_across_aspects_LK[2])}\n")
        file.write(f"Max recall LK aspect M: {max(recall_across_aspects_LK[0])}\n")
        file.write(f"Max recall LK aspect C: {max(recall_across_aspects_LK[1])}\n")
        file.write(f"Max recall LK aspect P: {max(recall_across_aspects_LK[2])}\n")
        file.write(f"Max F1score LK aspect M: {max(F1score_across_aspects_LK[0])}\n")
        file.write(f"Max F1score LK aspect C: {max(F1score_across_aspects_LK[1])}\n")
        file.write(f"Max F1score LK aspect P: {max(F1score_across_aspects_LK[2])}\n")
        file.write(f"Max FDR LK aspect M: {max(FDR_across_aspects_LK[0])}\n")
        file.write(f"Max FDR LK aspect C: {max(FDR_across_aspects_LK[1])}\n")
        file.write(f"Max FDR LK aspect P: {max(FDR_across_aspects_LK[2])}\n")
        file.write(f"GPs with no predictions LK aspect M: {gp_across_aspects_LK[0]}\n\n")
        file.write(f"GPs with no predictions LK aspect C: {gp_across_aspects_LK[1]}\n\n")
        file.write(f"GPs with no predictions LK aspect P: {gp_across_aspects_LK[2]}\n\n")
            
        
        file.write(f"Steps: {steps}\n")
        file.write(f"min thresh: {min_thresh}\n")
        file.write(f"max thresh: {max_thresh}\n")
        file.write("Thresholds: \n" + ", ".join(map(str, thresholds)) + "\n\n")
        
        # print comprehensive lists
        file.write(f"\nNK results\n")
        file.write("Precision in NK: \n" + ", ".join(map(str, precision_across_aspects_NK[0])) + "\n")
        file.write("Precision in NK: \n" + ", ".join(map(str, precision_across_aspects_NK[1])) + "\n")
        file.write("Precision in NK: \n" + ", ".join(map(str, precision_across_aspects_NK[2])) + "\n")
        file.write("Recall in NK: \n" + ", ".join(map(str, recall_across_aspects_NK[0])) + "\n")
        file.write("Recall in NK: \n" + ", ".join(map(str, recall_across_aspects_NK[1])) + "\n")
        file.write("Recall in NK: \n" + ", ".join(map(str, recall_across_aspects_NK[2])) + "\n")
        file.write("FDR in NK: \n" + ", ".join(map(str, FDR_across_aspects_NK[0])) + "\n")
        file.write("FDR in NK: \n" + ", ".join(map(str, FDR_across_aspects_NK[1])) + "\n")
        file.write("FDR in NK: \n" + ", ".join(map(str, FDR_across_aspects_NK[2])) + "\n")
        file.write("F1score in NK: \n" + ", ".join(map(str, F1score_across_aspects_NK[0])) + "\n")
        file.write("F1score in NK: \n" + ", ".join(map(str, F1score_across_aspects_NK[1])) + "\n")
        file.write("F1score in NK: \n" + ", ".join(map(str, F1score_across_aspects_NK[2])) + "\n")
        file.write(f"\nLK results\n")
        file.write("Precision in LK: \n" + ", ".join(map(str, precision_across_aspects_LK[0])) + "\n")
        file.write("Precision in LK: \n" + ", ".join(map(str, precision_across_aspects_LK[1])) + "\n")
        file.write("Precision in LK: \n" + ", ".join(map(str, precision_across_aspects_LK[2])) + "\n")
        file.write("Recall in LK: \n" + ", ".join(map(str, recall_across_aspects_LK[0])) + "\n")
        file.write("Recall in LK: \n" + ", ".join(map(str, recall_across_aspects_LK[1])) + "\n")
        file.write("Recall in LK: \n" + ", ".join(map(str, recall_across_aspects_LK[2])) + "\n")
        file.write("FDR in LK: \n" + ", ".join(map(str, FDR_across_aspects_LK[0])) + "\n")
        file.write("FDR in LK: \n" + ", ".join(map(str, FDR_across_aspects_LK[1])) + "\n")
        file.write("FDR in LK: \n" + ", ".join(map(str, FDR_across_aspects_LK[2])) + "\n")
        file.write("F1score in LK: \n" + ", ".join(map(str, F1score_across_aspects_LK[0])) + "\n")
        file.write("F1score in LK: \n" + ", ".join(map(str, F1score_across_aspects_LK[1])) + "\n")
        file.write("F1score in LK: \n" + ", ".join(map(str, F1score_across_aspects_LK[2])) + "\n")

    print(f"Report saved as {report_name}")


def bench_by_aspect(model_path_C5, model_path_C4, model_name, stepsize, cafa, dir_tree):
    
    if cafa == "C4":
        cafa_types = ["C4"]
    elif cafa == "C5":
        cafa_types = ["C5"]
    elif cafa == "both":
        cafa_types = ["C5", "C4"]
    else:
        cafa_types = ["C5", "C4"]

    print(f"Proceding with a benchmark with {1/stepsize} steps")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Model tested:")
    print(model_name)
    print("Cafa tested:")
    print(cafa_types)

    print(f"Current model: {model_name}")
    for cafa_type in cafa_types:
        print(f"Current cafa: {cafa_type}")
        if cafa_type == 'C5':
            compute_benchmark_by_cafa(model_path_C5, model_name, cafa_type, stepsize, current_datetime, dir_tree)
        elif cafa_type == 'C4':
            compute_benchmark_by_cafa(model_path_C4, model_name, cafa_type, stepsize, current_datetime, dir_tree)
