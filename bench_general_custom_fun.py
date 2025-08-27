import pandas as pd
import numpy as np
from sklearn.metrics import auc
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import sys
import os
import argparse
from datetime import datetime
import json

start_time = time.time()


def make_ROC(fpr, tpr, precision, tool_name, cafa, data_type, path):
    roc_auc = auc(fpr, tpr)
    # Find the index of the highest precision
    best_idx = np.argmax(precision)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label='Max Precision', zorder=5)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # Annotate the best precision point
    plt.annotate(f'Precision = {precision[best_idx]:.2f}', 
                (fpr[best_idx], tpr[best_idx]),
                textcoords="offset points", xytext=(10, -15), ha='left', color='red')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{data_type} ROC Curve for {tool_name} in {cafa}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/ROC_{data_type}_{tool_name}_{cafa}.png", format="png")
    
    
def make_multiROC(fpr_NK, tpr_NK, precision_NK, fpr_LK, tpr_LK, precision_LK, fpr_general, tpr_general, precision_general, tool_name, cafa, path):
    
    # Compute AUC
    auc1 = auc(fpr_NK, tpr_NK)
    auc2 = auc(fpr_LK, tpr_LK)
    auc3 = auc(fpr_general, tpr_general)
    
    # Best precision points
    idx1 = np.argmax(precision_NK)
    idx2 = np.argmax(precision_LK)
    idx3 = np.argmax(precision_general)
    
    # Plot
    plt.figure(figsize=(10, 7))

    # Curve 1
    plt.plot(fpr_NK, tpr_NK, label=f'NK (AUC = {auc1:.2f})', color='blue')
    plt.scatter(fpr_NK[idx1], tpr_NK[idx1], color='blue', edgecolors='black', zorder=5)
    plt.annotate(f'P={precision_NK[idx1]:.2f}', (fpr_NK[idx1], tpr_NK[idx1]), textcoords="offset points", xytext=(10,-10), color='blue')

    # Curve 2
    plt.plot(fpr_LK, tpr_LK, label=f'LK (AUC = {auc2:.2f})', color='green')
    plt.scatter(fpr_LK[idx2], tpr_LK[idx2], color='green', edgecolors='black', zorder=5)
    plt.annotate(f'P={precision_LK[idx2]:.2f}', (fpr_LK[idx2], tpr_LK[idx2]), textcoords="offset points", xytext=(10,-10), color='green')

    # Curve 3
    plt.plot(fpr_general, tpr_general, label=f'general (AUC = {auc3:.2f})', color='red')
    plt.scatter(fpr_general[idx3], tpr_general[idx3], color='red', edgecolors='black', zorder=5)
    plt.annotate(f'P={precision_general[idx3]:.2f}', (fpr_general[idx3], tpr_general[idx3]), textcoords="offset points", xytext=(10,-10), color='red')

    # Diagonal line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Comprehensive ROC Curve for {tool_name} in {cafa}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/comprehensive_ROC_{tool_name}_{cafa}.png", format="png")
    
    
def make_PRC(precision, recall, tool_name, cafa, data_type, path):
    precision = np.array(precision)
    recall = np.array(recall)
    
    prc_auc = auc(recall, precision)
    # Find the index of the highest precision
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero

    # Find index of maximum F1 score
    max_f1_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[max_f1_idx]

    # Plot PRC curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PRC curve (AUC = {prc_auc:.2f})', color='blue')
    plt.scatter(recall[max_f1_idx], precision[max_f1_idx], color='red', s=100, 
                label=f'Max F1 = {max_f1:.2f}', edgecolors='black', zorder=5)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{data_type} PRC Curve for {tool_name} in {cafa}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/PRC_{data_type}_{tool_name}_{cafa}.png", format="png")
    
    
def make_multiPRC(precision_NK, recall_NK, precision_LK, recall_LK, precision_general, recall_general, tool_name, cafa, path):
    precision_NK = np.array(precision_NK)
    recall_NK = np.array(recall_NK)
    precision_LK = np.array(precision_LK)
    recall_LK = np.array(recall_LK)
    precision_general = np.array(precision_general)
    recall_general = np.array(recall_general)
    
    # Compute AUC
    auc1 = auc(recall_NK, precision_NK)
    auc2 = auc(recall_LK, precision_LK)
    auc3 = auc(recall_general, precision_general)
    
    f1_scores_NK = 2 * (precision_NK * recall_NK) / (precision_NK + recall_NK + 1e-8)
    f1_scores_LK = 2 * (precision_LK * recall_LK) / (precision_LK + recall_LK + 1e-8)
    f1_scores_general = 2 * (precision_general * recall_general) / (precision_general + recall_general + 1e-8)
    
    # Find index of maximum F1 score
    max_f1_idx_NK = np.argmax(f1_scores_NK)
    max_f1_idx_LK = np.argmax(f1_scores_LK)
    max_f1_idx_general = np.argmax(f1_scores_general)
    max_f1_NK = f1_scores_NK[max_f1_idx_NK]
    max_f1_LK = f1_scores_LK[max_f1_idx_LK]
    max_f1_general = f1_scores_general[max_f1_idx_general]
    
    # Plot
    plt.figure(figsize=(10, 7))

    # Curve 1
    plt.plot(recall_NK, precision_NK, label=f'NK (AUC = {auc1:.2f})', color='blue')
    plt.scatter(recall_NK[max_f1_idx_NK], precision_NK[max_f1_idx_NK], label=f'NK F1max = {max_f1_NK:.2f}', color='blue', edgecolors='black', zorder=5)
    
    # Curve 2
    plt.plot(recall_LK, precision_LK, label=f'LK (AUC = {auc2:.2f})', color='green')
    plt.scatter(recall_LK[max_f1_idx_LK], precision_LK[max_f1_idx_LK], label=f'LK F1max = {max_f1_LK:.2f}', color='green', edgecolors='black', zorder=5)

    # Curve 3
    plt.plot(recall_general, precision_general, label=f'general (AUC = {auc3:.2f})', color='red')
    plt.scatter(recall_general[max_f1_idx_LK], precision_general[max_f1_idx_LK], label=f'general F1max = {max_f1_general:.2f}', color='red', edgecolors='black', zorder=5)
    
    # Labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Comprehensive PRC Curve for {tool_name} in {cafa}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/comprehensive_PRC_{tool_name}_{cafa}.png", format="png")


def compute_benchmark_by_cafa(path, model, cafa, steps, current_datetime, IC_dict, dir_tree):

    tool_name = model
        
    if cafa == "C4":
        cafa_column = "CAFA4_ID"
    else:
        cafa_column = "Query_ID"
        
    print(cafa_column)
        
    tool_predictions_address = path
    
    tool_predictions = pd.read_csv(tool_predictions_address, sep="\t")
    tool_predictions["GO_ID"] = tool_predictions["GO_ID"].str.replace("_", ":") 
    tool_predictions['IC'] = tool_predictions['GO_ID'].map(IC_dict)
    default_IC = 10
    print(tool_predictions.head())
    # retrieve the Information Content
    

    # Read the file and get the dataframe
    knowledge_type = "NK"   # NK or LK
    ground_truth_address = dir_tree['gt_dir_path'] + f"/ground_truth_{knowledge_type}_{cafa}.tsv"
    ground_truth = pd.read_csv(ground_truth_address, sep="\t") 
    # Sort alphabetically (the algorithm expects alphabetical order)
    ground_truth = ground_truth.sort_values(by=[cafa_column, "GO_ID"], ascending=[True, True])
    gt_grouped = ground_truth.groupby(cafa_column)

    # Uniamo LK and NK sullo stesso grafico? magari con colori diversi?
    # thresholder

    tot_recall_NK = []
    tot_precision_NK = []
    tot_F1score_NK = []
    tot_FDR_NK = []
    tot_FPR_NK = []
    tot_TPR_NK = []
    tot_S_measure_NK = []

    
    min_thresh = 0
    max_thresh = 1
    thresholds = np.arange(min_thresh, max_thresh, steps)

    print(f"Benchmark evaluator for {tool_name} in {cafa}")

    # NK cycle
    for threshold in tqdm(thresholds, desc=f"Processing NK thresholds"):
        total_TP_NK = 0
        total_TN_NK = 0
        total_FP_NK = 0
        total_FN_NK = 0
        total_P_NK = 0
        total_S_NK = 0
        gp_no_pred_NK = 0  # GPs with no predictions, poco utile, si accumula in continuazione
        
        tp_thresh = tool_predictions[tool_predictions["Score"]>threshold]
        tp_grouped = tp_thresh.groupby('Query_ID')
        tp_not_thresh = tool_predictions[tool_predictions["Score"]<=threshold]
        tp_grouped_not_thresh = tp_not_thresh.groupby('Query_ID')
        
        for group_id, group_gt in gt_grouped:  # Get a group of rows selected by ID
            if group_id in tp_grouped.groups:  # Ensure the ID exists in both DataFrames
                group_tp = tp_grouped.get_group(group_id)  # Retrieve predictions regarding the specific ID
                GO_pred = group_tp['GO_ID']
            else:  # If not behave as if there are no predictions
                GO_pred = []
                gp_no_pred_NK += 1
            
            if group_id in tp_grouped_not_thresh.groups:
                group_tp_not_thresh = tp_grouped_not_thresh.get_group(group_id)
                GO_not_pred = group_tp_not_thresh['GO_ID']
                GO_not_pred = set(GO_not_pred)
                TN_predictions = list(GO_not_pred - GO_gt) # Prediction excluded that are not in the gt
            else:
                GO_not_pred = []
                TN_predictions = []

            GO_pred = set(GO_pred)
            GO_gt = group_gt['GO_ID']
            GO_gt = set(GO_gt)
            
            # Salviamo tutti questi dati in un dizionario per ogni gp? (magari con la threshold come chiave)
            TP_predictions = list(GO_pred & GO_gt)  # The TP predictions are the shared ones
            FP_predictions = list(GO_pred - GO_gt)  # Predictions that are not in the gt
            FN_predictions = list(GO_gt - GO_pred)  # Predictions missed from the gt
            FP_IC_values = np.array([IC_dict.get(prediction, default_IC) for prediction in FP_predictions])
            FN_IC_values = np.array([IC_dict.get(prediction, default_IC) for prediction in FN_predictions])
            S_value = np.sqrt(np.power(np.sum(FP_IC_values), 2) + np.power(np.sum(FN_IC_values), 2))
            
            total_S_NK += S_value
            total_TP_NK += len(TP_predictions)
            total_TN_NK += len(TN_predictions)
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
        if (total_TN_NK + total_FP_NK) == 0:
            FPR = 0
        else:
            FPR = total_FP_NK/(total_TN_NK + total_FP_NK)
        if (total_TP_NK + total_FN_NK) == 0:
            TPR = 0
        else:
            TPR = total_TP_NK/(total_TP_NK + total_FN_NK)
        
        tot_precision_NK.append(precision)
        tot_recall_NK.append(recall)
        tot_F1score_NK.append(F1_score)
        tot_FDR_NK.append(FDR)
        tot_FPR_NK.append(FPR)
        tot_TPR_NK.append(TPR)
        tot_S_measure_NK.append(total_S_NK)

    NK_time = time.time()
    elapsed_time = NK_time - start_time  # End timer NK
    print(f"Elapsed time: {elapsed_time:.3f} seconds to process NK cycle")

    print(f"Max precision {knowledge_type}: {max(tot_precision_NK)}")
    print(f"Max recall {knowledge_type}: {max(tot_recall_NK)}")
    print(f"Max F1score {knowledge_type}: {max(tot_F1score_NK)}")
    print(f"Max FDR {knowledge_type}: {max(tot_FDR_NK)}")
    print(f"Min S measure {knowledge_type}: {min(tot_S_measure_NK)}")
    print(f"GP with no predictions: {gp_no_pred_NK}")


    # Read the file and get the dataframe
    knowledge_type = "LK"   # NK or LK
    ground_truth_address = dir_tree['gt_dir_path'] + f"/ground_truth_LK_KK_{cafa}.tsv"
    ground_truth = pd.read_csv(ground_truth_address, sep="\t") 
    # Sort alphabetically (the algorithm expects alphabetical order)
    ground_truth = ground_truth.sort_values(by=[cafa_column, "GO_ID"], ascending=[True, True])
    gt_grouped = ground_truth.groupby(cafa_column)
    # tp_grouped = tool_predictions.groupby('Query_ID')

    tot_recall_LK = []
    tot_precision_LK = []
    tot_F1score_LK = []
    tot_FDR_LK = []
    tot_FPR_LK = []
    tot_TPR_LK = []
    tot_S_measure_LK = []


    print(f"Benchmark evaluator for {tool_name} in {cafa}")

    # LK cycle
    for threshold in tqdm(thresholds, desc=f"Processing LK thresholds"):
        total_TP_LK = 0
        total_TN_LK = 0
        total_FP_LK = 0
        total_FN_LK = 0
        total_P_LK = 0
        total_S_LK = 0
        gp_no_pred_LK = 0  # GPs with no predictions
        
        tp_thresh = tool_predictions[tool_predictions["Score"]>threshold]
        tp_not_thresh = tool_predictions[tool_predictions["Score"]<=threshold]
        tp_grouped = tp_thresh.groupby('Query_ID')
        tp_grouped_not_thresh = tp_not_thresh.groupby('Query_ID')
        
        for group_id, group_gt in gt_grouped:  # Get a group of rows selected by ID
            if group_id in tp_grouped.groups:  # Ensure the ID exists in both DataFrames
                group_tp = tp_grouped.get_group(group_id)  # Retrieve predictions regarding the specific ID
                GO_pred = group_tp['GO_ID']
            else:  # If not behave as if there are no predictions
                GO_pred = []
                gp_no_pred_LK += 1
            
            if group_id in tp_grouped_not_thresh.groups:
                group_tp_not_thresh = tp_grouped_not_thresh.get_group(group_id)
                GO_not_pred = group_tp_not_thresh['GO_ID']
                GO_not_pred = set(GO_not_pred)
                TN_predictions = list(GO_not_pred - GO_gt) # Prediction excluded that are not in the gt
            else:
                GO_not_pred = []
                TN_predictions = []
                
            GO_pred = set(GO_pred)
            GO_gt = group_gt['GO_ID']
            GO_gt = set(GO_gt)
            
            TP_predictions = list(GO_pred & GO_gt)  # The TP predictions are the shared ones
            FP_predictions = list(GO_pred - GO_gt)  # Predictions that are not in the gt
            FN_predictions = list(GO_gt - GO_pred)  # Predictions missed from the gt
            FP_IC_values = np.array([IC_dict.get(prediction, default_IC) for prediction in FP_predictions])
            FN_IC_values = np.array([IC_dict.get(prediction, default_IC) for prediction in FN_predictions])
            S_value = np.sqrt(np.power(np.sum(FP_IC_values), 2) + np.power(np.sum(FN_IC_values), 2))
            
            total_S_LK += S_value
            total_TP_LK += len(TP_predictions)
            total_TN_LK += len(TN_predictions)
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
        if (total_TN_LK + total_FP_LK) == 0:
            FPR = 0
        else:
            FPR = total_FP_LK/(total_TN_LK + total_FP_LK)
        if (total_TP_LK + total_FN_LK) == 0:
            TPR = 0
        else:
            TPR = total_TP_LK/(total_TP_LK + total_FN_LK)
        
        tot_precision_LK.append(precision)
        tot_recall_LK.append(recall)
        tot_F1score_LK.append(F1_score)
        tot_FDR_LK.append(FDR)
        tot_FPR_LK.append(FPR)
        tot_TPR_LK.append(TPR)
        tot_S_measure_LK.append(total_S_LK)

    LK_time = time.time()
    elapsed_time = LK_time - NK_time  # End timer LK
    tot_elapsed_time = LK_time - start_time
    print(f"Elapsed time: {elapsed_time:.3f} seconds to process LK cycle, {tot_elapsed_time:.3f} in total")

    print(f"Max precision {knowledge_type}: {max(tot_precision_LK)}")
    print(f"Max recall {knowledge_type}: {max(tot_recall_LK)}")
    print(f"Max F1score {knowledge_type}: {max(tot_F1score_LK)}")
    print(f"Min S measure {knowledge_type}: {min(tot_S_measure_LK)}")
    print(f"Max FDR {knowledge_type}: {max(tot_FDR_LK)}")


    # General cycle
    # Read the file and get the dataframe
    knowledge_type = "general"
    ground_truth_address = dir_tree['gt_dir_path'] + f"/ground_truth_{knowledge_type}_{cafa}.tsv"
    ground_truth = pd.read_csv(ground_truth_address, sep="\t") 
    # Sort alphabetically (the algorithm expects alphabetical order)
    ground_truth = ground_truth.sort_values(by=[cafa_column, "GO_ID"], ascending=[True, True])
    gt_grouped = ground_truth.groupby(cafa_column)

    tot_recall_general = []
    tot_precision_general = []
    tot_F1score_general = []
    tot_FDR_general = []
    tot_FPR_general = []
    tot_TPR_general = []
    tot_S_measure_general = []


    thresholds = np.arange(min_thresh, max_thresh, steps)

    print(f"Benchmark evaluator for {tool_name} in {cafa}")

    # general cycle
    for threshold in tqdm(thresholds, desc=f"Processing general thresholds"):
        total_TP_general = 0
        total_TN_general = 0
        total_FP_general = 0
        total_FN_general = 0
        total_P_general = 0
        total_S_general = 0
        gp_no_pred_general = 0  # GPs with no predictions
        tp_thresh = tool_predictions[tool_predictions["Score"]>threshold]
        tp_not_thresh = tool_predictions[tool_predictions["Score"]<=threshold]
        tp_grouped = tp_thresh.groupby('Query_ID')
        tp_grouped_not_thresh = tp_not_thresh.groupby('Query_ID')
        
        for group_id, group_gt in gt_grouped:  # Get a group of rows selected by ID
            if group_id in tp_grouped.groups:  # Ensure the ID exists in both DataFrames
                group_tp = tp_grouped.get_group(group_id)  # Retrieve predictions regarding the specific ID
                GO_pred = group_tp['GO_ID']
            else:  # If not behave as if there are no predictions
                GO_pred = []
                gp_no_pred_general += 1
            
            if group_id in tp_grouped_not_thresh.groups:
                group_tp_not_thresh = tp_grouped_not_thresh.get_group(group_id)
                GO_not_pred = group_tp_not_thresh['GO_ID']
                GO_not_pred = set(GO_not_pred)
                TN_predictions = list(GO_not_pred - GO_gt) # Prediction excluded that are not in the gt
            else:
                GO_not_pred = []
                TN_predictions = []

                
            GO_pred = set(GO_pred)
            GO_gt = group_gt['GO_ID']
            GO_gt = set(GO_gt)
            
            TP_predictions = list(GO_pred & GO_gt)  # The TP predictions are the shared ones
            FP_predictions = list(GO_pred - GO_gt)  # Predictions that are not in the gt
            FN_predictions = list(GO_gt - GO_pred)  # Predictions missed from the gt
            FP_IC_values = np.array([IC_dict.get(prediction, default_IC) for prediction in FP_predictions])
            FN_IC_values = np.array([IC_dict.get(prediction, default_IC) for prediction in FN_predictions])
            S_value = np.sqrt(np.power(np.sum(FP_IC_values), 2) + np.power(np.sum(FN_IC_values), 2))
            
            total_S_general += S_value
            total_TP_general += len(TP_predictions)
            total_TN_general += len(TN_predictions)
            total_FP_general += len(FP_predictions)
            total_FN_general += len(FN_predictions)
            total_P_general += len(GO_gt)
                
        if (total_TP_general + total_FP_general) == 0:  # 0 div check precision
            precision = 0
        else:
            precision = total_TP_general/(total_TP_general + total_FP_general)
        if total_P_general == 0:  # 0 div check recall
            recall = 0
        else:
            recall = total_TP_general/total_P_general  
        if (precision + recall) == 0:  # 0 div check F1 score
            F1_score = 0
        else:
            F1_score = 2*(precision * recall)/(precision + recall)
        if (total_TP_general + total_FP_general) == 0: # 0 div check FDR
            FDR = 0
        else:
            FDR = total_FP_general / (total_TP_general + total_FP_general)
        if (total_TN_general + total_FP_general) == 0:
            FPR = 0
        else:
            FPR = total_FP_general/(total_TN_general + total_FP_general)
        if (total_TP_general + total_FN_general) == 0:
            TPR = 0
        else:
            TPR = total_TP_general/(total_TP_general + total_FN_general)
        
        tot_precision_general.append(precision)
        tot_recall_general.append(recall)
        tot_F1score_general.append(F1_score)
        tot_FDR_general.append(FDR)
        tot_FPR_general.append(FPR)
        tot_TPR_general.append(TPR)
        tot_S_measure_general.append(total_S_general)

    general_time = time.time()
    elapsed_time = general_time - start_time  # End timer general
    print(f"Elapsed time: {elapsed_time:.3f} seconds to process general cycle")


    print(f"Max precision {knowledge_type}: {max(tot_precision_general)}")
    print(f"Max recall {knowledge_type}: {max(tot_recall_general)}")
    print(f"Max F1score {knowledge_type}: {max(tot_F1score_general)}")
    print(f"Max FDR {knowledge_type}: {max(tot_FDR_general)}")
    print(f"Min S measure {knowledge_type}: {min(tot_S_measure_general)}")
    print(f"GP with no predictions: {gp_no_pred_general}")
    
        
    # ROC graphs
    ROC_dir_name = f"ROC_{tool_name}_{current_datetime}"
    ROC_path = os.path.join(dir_tree['btp_dir_path'] + f"/benchmark_{cafa}/ROC_{cafa}", ROC_dir_name)
    if not os.path.exists(ROC_path):
        os.makedirs(ROC_path)
        
    make_ROC(tot_FPR_NK, tot_TPR_NK, tot_precision_NK, tool_name, cafa, "NK", ROC_path)
    make_ROC(tot_FPR_LK, tot_TPR_LK, tot_precision_LK, tool_name, cafa, "LK", ROC_path)
    make_ROC(tot_FPR_general, tot_TPR_general, tot_precision_general, tool_name, cafa, "general", ROC_path)
    
    make_multiROC(tot_FPR_NK, tot_TPR_NK, tot_precision_NK, tot_FPR_LK, tot_TPR_LK, tot_precision_LK, 
                  tot_FPR_general, tot_TPR_general, tot_precision_general, tool_name, cafa, ROC_path)
    
    
    # PRC graphs
    PRC_dir_name = f"PRC_{tool_name}_{current_datetime}"
    PRC_path = os.path.join(dir_tree['btp_dir_path'] + f"/benchmark_{cafa}/PRC_{cafa}", PRC_dir_name)
    if not os.path.exists(PRC_path):
        os.makedirs(PRC_path)
                
    make_PRC(tot_precision_NK, tot_recall_NK, tool_name, cafa, "NK", PRC_path)
    make_PRC(tot_precision_LK, tot_recall_LK, tool_name, cafa, "LK", PRC_path)
    make_PRC(tot_precision_general, tot_recall_general, tool_name, cafa, "general", PRC_path)
    
    make_multiPRC(tot_precision_NK, tot_recall_NK, tot_precision_LK, tot_recall_LK, 
                  tot_precision_general, tot_recall_general, tool_name, cafa, PRC_path)
        
    

    titles = [f"Precision  for {tool_name}", 
            f"Recall  for {tool_name}", 
            f"S measure  for {tool_name}", 
            f"F1Score  for {tool_name}"
    ]
    
    NK_dir_name = f"NK_{tool_name}_{current_datetime}"
    NK_path = os.path.join(dir_tree['btp_dir_path'] + f"/benchmark_{cafa}/benchmark_NK_{cafa}", NK_dir_name)
    if not os.path.exists(NK_path):
        os.makedirs(NK_path)

    # Only NK and LK
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array for easy iteration
    y_data = [tot_precision_NK, tot_recall_NK, tot_S_measure_NK, tot_F1score_NK]
    # Loop through each subplot
    for i, ax in enumerate(axes):
        y = y_data[i]  # Select y data
        
        if i == 2:
            # For S_measure, highlight the minimum
            extremum_idx = np.argmin(y)
        else:
            # For others, highlight the maximum
            extremum_idx = np.argmax(y)
        max_x = thresholds[extremum_idx]  # X value of max
        max_y = y[extremum_idx]  # Y value of max

        ax.plot(thresholds, y)
        ax.scatter(max_x, max_y, color='red', s=100, label=f"{max_y:.5f}")  # Red dot for max
        ax.set_title(titles[i])  # Set title
        ax.legend()
        # ax.grid(True)
    fig.suptitle(f'NK benchmark for {tool_name} in {cafa}', y=1.5)
    plt.savefig(f"{NK_path}/NK_benchmark_{tool_name}_{cafa}.png", format="png")
    
    LK_dir_name = f"LK_{tool_name}_{current_datetime}"
    LK_path = os.path.join(dir_tree['btp_dir_path'] + f"/benchmark_{cafa}/benchmark_LK_{cafa}", LK_dir_name)
    if not os.path.exists(LK_path):
        os.makedirs(LK_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array for easy iteration
    y_data = [tot_precision_LK, tot_recall_LK, tot_S_measure_LK, tot_F1score_LK]
    # Loop through each subplot
    for i, ax in enumerate(axes):
        y = y_data[i]  # Select y data
        
        if i == 2:
            # For S_measure, highlight the minimum
            extremum_idx = np.argmin(y)
        else:
            # For others, highlight the maximum
            extremum_idx = np.argmax(y)
        max_x = thresholds[extremum_idx]  # X value of max
        max_y = y[extremum_idx]  # Y value of max

        ax.plot(thresholds, y)
        ax.scatter(max_x, max_y, color='red', s=100, label=f"{max_y:.5f}")  # Red dot for max
        ax.set_title(titles[i])  # Set title
        ax.legend()
        # ax.grid(True)
    fig.suptitle(f'LK benchmark for {tool_name} in {cafa}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Increase the top margin
    plt.savefig(f"{LK_path}/LK_benchmark_{tool_name}_{cafa}.png", format="png")


    curve_colors = ["blue", "green"]
    highlight_colors = ["red", "orange"]
    
    comprehensive_dir_name = f"comprehensive_{tool_name}_{current_datetime}"
    comprehensive_path = os.path.join(dir_tree['btp_dir_path'] + f"/benchmark_{cafa}/comprehensive_benchmark_{cafa}", comprehensive_dir_name)
    if not os.path.exists(comprehensive_path):
        os.makedirs(comprehensive_path)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array for easy iteration
    y_data_1 = [tot_precision_NK, tot_recall_NK, tot_S_measure_NK, tot_F1score_NK]
    y_data_2 = [tot_precision_LK, tot_recall_LK, tot_S_measure_LK, tot_F1score_LK]
    label = [["precision NK", "precision LK"], ["recall NK", "recall LK"], ["S measure NK", "S measure LK"], ["F1score NK", "F1score LK"]]
    # Loop through each subplot
    for i, ax in enumerate(axes):
        ys = [y_data_1[i], y_data_2[i]]

        for j, y in enumerate(ys):
            if i == 2:  # For S-measure, highlight minimum
                idx = np.argmin(y)
            else:       # For others, highlight maximum
                idx = np.argmax(y)

            x_val, y_val = thresholds[idx], y[idx]

            # Plot the curve
            ax.plot(thresholds, y, label=label[i][j], color=curve_colors[j])
            # Highlight the point
            ax.scatter(x_val, y_val, color=highlight_colors[j], s=100, label=f"{y_val:.5f}")

        ax.set_title(titles[i])
        ax.legend()
        # ax.grid(True)
    # Adjust layout
    plt.tight_layout()
    fig.suptitle(f'Comprehensive benchmark for {tool_name} in {cafa}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Increase the top margin
    plt.savefig(f"{comprehensive_path}/NK_LK_benchmark_{tool_name}_{cafa}.png", format="png")
    #######################################################################
    
    general_dir_name = f"general_{tool_name}_{current_datetime}"
    general_path = os.path.join(dir_tree['btp_dir_path'] + f"/benchmark_{cafa}/benchmark_general_{cafa}", general_dir_name)
    if not os.path.exists(general_path):
        os.makedirs(general_path)
    
    # General evaluation
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array for easy iteration
    y_data = [tot_precision_general, tot_recall_general, tot_S_measure_general, tot_F1score_general]
    # Loop through each subplot
    for i, ax in enumerate(axes):
        y = y_data[i]  # Select y data
        if i == 2:
            # For S_measure, highlight the minimum
            extremum_idx = np.argmin(y)
        else:
            # For others, highlight the maximum
            extremum_idx = np.argmax(y)
        max_x = thresholds[extremum_idx]  # X value of max
        max_y = y[extremum_idx]  # Y value of max

        ax.plot(thresholds, y)
        ax.scatter(max_x, max_y, color='red', s=100, label=f"{max_y:.5f}")  # Red dot for max
        ax.set_title(titles[i])  # Set title
        ax.legend()
        # ax.grid(True)
    fig.suptitle(f'General benchmark for {tool_name} in {cafa}', y=1.5)
    plt.savefig(f"{general_path}/general_benchmark_{tool_name}_{cafa}.png", format="png")


    curve_colors = ["blue", "green", "black"]
    highlight_colors = ["red", "orange", "yellow"]
    

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array for easy iteration
    y_data_1 = [tot_precision_NK, tot_recall_NK, tot_S_measure_NK, tot_F1score_NK]
    y_data_2 = [tot_precision_LK, tot_recall_LK, tot_S_measure_LK, tot_F1score_LK]
    y_data_3 = [tot_precision_general, tot_recall_general, tot_S_measure_general, tot_F1score_general]
    label = [["precision NK", "precision LK", "precision all"], ["recall NK", "recall LK", "recall all"], ["S measure NK", "S measure LK", "S measure all"], ["F1score NK", "F1score LK", "F1score all"]]
    # Loop through each subplot
    for i, ax in enumerate(axes):
        # Select y-values for this subplot
        ys = [y_data_1[i], y_data_2[i], y_data_3[i]]
        
        # Loop through each curve (NK, LK, general)
        for j, y in enumerate(ys):
            if i == 2:  # S-measure: use minimum
                idx = np.argmin(y)
            else:       # Others: use maximum
                idx = np.argmax(y)

            x_val, y_val = thresholds[idx], y[idx]

            # Plot the curve
            ax.plot(thresholds, y, label=label[i][j], color=curve_colors[j])
            # Highlight the point
            ax.scatter(x_val, y_val, color=highlight_colors[j], s=100, label=f"{y_val:.5f}")

        ax.set_title(titles[i])
        ax.legend()
        # ax.grid(True)
    # Adjust layout
    plt.tight_layout()
    fig.suptitle(f'Overall benchmark for {tool_name} in {cafa}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Increase the top margin
    # comprehensive_path
    plt.savefig(f"{comprehensive_path}/comp_general_benchmark_{tool_name}_{cafa}.png", format="png")
    #####################################
   
    
    report_dir_name = f"general_report_{tool_name}_{current_datetime}"
    report_dir_name = os.path.join(dir_tree['btp_dir_path'] + f"/benchmark_{cafa}/reports", report_dir_name)
    if not os.path.exists(report_dir_name):
        os.makedirs(report_dir_name)
    
    report_name = f"{report_dir_name}/general_report_{tool_name}_{cafa}.txt"
    with open(report_name, "w") as file:
        file.write(f"Max precision NK: {max(tot_precision_NK)}\n")
        file.write(f"Max recall NK: {max(tot_recall_NK)}\n")
        file.write(f"Max F1score NK: {max(tot_FDR_NK)}\n")
        file.write(f"Max FDR NK: {max(tot_F1score_NK)}\n")
        print(f"Min S measure NK: {min(tot_S_measure_NK)}")
        file.write(f"GPs with no predictions NK: {gp_no_pred_NK}\n\n")
        
        file.write(f"Max precision LK: {max(tot_precision_LK)}\n")
        file.write(f"Max recall LK: {max(tot_recall_LK)}\n")
        file.write(f"Max F1score LK: {max(tot_F1score_LK)}\n")
        file.write(f"Max FDR LK: {max(tot_FDR_LK)}\n")
        print(f"Min S measure LK: {min(tot_S_measure_LK)}")
        file.write(f"GPs with no predictions LK: {gp_no_pred_LK}\n\n")
        
        file.write(f"Max precision general: {max(tot_precision_general)}\n")
        file.write(f"Max recall general: {max(tot_recall_general)}\n")
        file.write(f"Max F1score general: {max(tot_F1score_general)}\n")
        file.write(f"Max FDR general: {max(tot_FDR_general)}\n")
        print(f"Min S measure general: {min(tot_S_measure_general)}")
        file.write(f"GPs with no predictions general: {gp_no_pred_general}\n\n")
            
        
        file.write(f"Steps: {steps}\n")
        file.write(f"min thresh: {min_thresh}\n")
        file.write(f"max thresh: {max_thresh}\n")
        file.write("Thresholds: \n" + ", ".join(map(str, thresholds)) + "\n\n")
        
        # print comprehensive lists
        file.write(f"\nNK results\n")
        file.write("Precision in NK: \n" + ", ".join(map(str, tot_precision_NK)) + "\n")
        file.write("Recall in NK: \n" + ", ".join(map(str, tot_recall_NK)) + "\n")
        file.write("FDR in NK: \n" + ", ".join(map(str, tot_FDR_NK)) + "\n")
        file.write("S measure in NK: \n" + ", ".join(map(str, tot_S_measure_NK)) + "\n")
        file.write("F1score in NK: \n" + ", ".join(map(str, tot_F1score_NK)) + "\n")
        file.write(f"\nLK results\n")
        file.write("Precision in LK: \n" + ", ".join(map(str, tot_precision_LK)) + "\n")
        file.write("Recall in LK: \n" + ", ".join(map(str, tot_recall_LK)) + "\n")
        file.write("FDR in LK: \n" + ", ".join(map(str, tot_FDR_LK)) + "\n")
        file.write("S measure in LK: \n" + ", ".join(map(str, tot_S_measure_LK)) + "\n")
        file.write("F1score in LK: \n" + ", ".join(map(str, tot_F1score_LK)) + "\n")
        file.write(f"\nGeneral results\n")
        file.write("Precision in general: \n" + ", ".join(map(str, tot_precision_general)) + "\n")
        file.write("Recall in general: \n" + ", ".join(map(str, tot_recall_general)) + "\n")
        file.write("FDR in general: \n" + ", ".join(map(str, tot_FDR_general)) + "\n")
        file.write("S measure in general: \n" + ", ".join(map(str, tot_S_measure_general)) + "\n")
        file.write("F1score in general: \n" + ", ".join(map(str, tot_F1score_general)) + "\n")

    print(f"Report saved as {report_name}")


def bench_general(model_path_C5, model_path_C4, model_name, stepsize, cafa, dir_tree):
    
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
    
    # Open and load the JSON file
    with open(dir_tree['owl_dir_path'] + '/dict_ics.json', 'r') as file:
        IC_dict = json.load(file)
        
    IC_dict = {key.replace('_', ':'): value for key, value in IC_dict.items()}

    print(f"Current model: {model_name}")
    for cafa_type in cafa_types:
        print(f"Current cafa: {cafa_type}")
        if cafa_type == 'C5':
            compute_benchmark_by_cafa(model_path_C5, model_name, cafa_type, stepsize, current_datetime, IC_dict, dir_tree)
        elif cafa_type == 'C4':
            compute_benchmark_by_cafa(model_path_C4, model_name, cafa_type, stepsize, current_datetime, IC_dict, dir_tree)
