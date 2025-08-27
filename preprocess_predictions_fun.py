import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import json
from owlLibrary3 import GoOwl


def colon_to_underscore(go_string):
    if ':' in go_string:  # Change colon to underscore
        return go_string.replace(':', '_')
    return go_string  # Return unchanged if ':' is not present



def substitute_obsolete_go(data_df, owl_data_dir):
    # Substitution and removal of obslete goas from the old annotation for compatibility purposes
    
    # The dicts use the standard GO_0123456 NOT GO:0123456!
    start_time = time.time()
    data_df['GO_ID'] = data_df['GO_ID'].str.replace(':', '_')

    # # Create a dictionary mapping obsolete GOs to lists of new GOs
    
    with open(owl_data_dir + "/depr.json", "r") as file:
        depr_dict = json.load(file)
        
    updated_rows = []
    # Process the rows of the main dataframe
    for index, row in data_df.iterrows():
        go_term = row['GO_ID']
        if go_term in depr_dict:
            if depr_dict[go_term] == "unreferenced":
                continue
            # If the GO term is obsolete, create a new row for each replacement
            for new_go in depr_dict[go_term]:
                new_row = row.copy()
                new_row['GO_ID'] = new_go
                updated_rows.append(new_row)
        else:
            # If not obsolete, keep the row as is
            updated_rows.append(row)

    # Create a new dataframe from the updated rows
    updated_df_nodepr = pd.DataFrame(updated_rows)
    
    
    with open(owl_data_dir + "/obs.json", "r") as file:
        obs_dict = json.load(file)

    updated_rows = []
    # Process the rows of the updated dataframe to remove the obsolete gos
    for index, row in updated_df_nodepr.iterrows():
        go_term = row['GO_ID']
        if go_term in obs_dict:
            for new_go in obs_dict[go_term]:
                new_row = row.copy()
                new_row['GO_ID'] = new_go
                updated_rows.append(new_row)
        else:
            # If not obsolete, keep the row as is
            updated_rows.append(row)
            
    updated_df_final = pd.DataFrame(updated_rows)
                 
    # Restore the previous state before saving the tsv
    updated_df_final['GO_ID'] = updated_df_final['GO_ID'].str.replace('_', ':')

    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds.")

    return updated_df_final
    

def propagated_existing_annotations_fast(owl_file, df):
    start_time = time.time()

    # --- 1. Setup and Pre-computation ---
    owl = GoOwl(owl_file, goa_file="", by_ontology=True)


    # --- Caching Ancestors (Major Optimization) ---
    print("Caching GO term ancestors...")
    unique_go_ids = df['GO_ID'].unique()
    ancestor_cache = {}
    root_nodes = {'GO_0005575', 'GO_0008150', 'GO_0003674'}
    for go_id in unique_go_ids:
        go_id_under = colon_to_underscore(go_id)
        # Get ancestors and filter out the root nodes immediately
        ancestors = owl.get_ancestors_id(go_id_under, by_ontology=True, valid_edges=True)
        ancestor_cache[go_id] = [anc for anc in ancestors if anc not in root_nodes]
    print(f"Cached ancestors for {len(unique_go_ids)} unique GO terms.")
    
    # --- 2. Collect Data in Lists (Eliminates `concat` in a loop) ---
    # Prepare lists to hold the data for our new DataFrame
    propagated_rows = []
    # Prepare static columns to avoid repeated lookups inside the loop
    base_columns = ["Query_ID", "Ontology", "Score"]


    print("Propagating annotations...")
    # Iterate using `to_dict('records')` which is much faster than iterrows
    for row in df.to_dict('records'):
        current_go_id = row['GO_ID']
        
        # Prepare a base dictionary for this row's data to avoid repetition
        base_row_data = {col: row.get(col) for col in base_columns}

        # Add the original annotation
        original_row = base_row_data.copy()
        original_row['GO_ID'] = current_go_id
        propagated_rows.append(original_row)

        # Add all cached ancestors
        ancestors = ancestor_cache.get(current_go_id, [])
        for ancestor_id in ancestors:
            ancestor_row = base_row_data.copy()
            # Replace GO_ID with the ancestor's ID
            ancestor_row['GO_ID'] = ancestor_id.replace('_', ':') # Store in standard format
            propagated_rows.append(ancestor_row)

    print(f"Cleaning {len(propagated_rows)} collected rows before DataFrame creation...")


    # print(propagated_rows)
    # --- 3. Create DataFrame in One Go (Highly Optimized) ---
    print(f"Creating final DataFrame from {len(propagated_rows)} propagated rows...")
    complete_df = pd.DataFrame(propagated_rows)

    # The original GO_ID might have underscores, standardize them all now
    complete_df['GO_ID'] = complete_df['GO_ID'].str.replace('_', ':')
    
    # --- 4. Final Processing (This part was already well-vectorized) ---
    print("Sorting and dropping duplicates...")
    
    # Sort by Query_ID, GO_ID, and then by priority to bring the best match to the top
    complete_df.sort_values(by=['Query_ID', 'GO_ID', 'Score'], ascending=[True, True, False], inplace=True)

    # Now drop duplicates on the protein-term pair, keeping the first (which is the highest priority)
    complete_df.drop_duplicates(keep='first', inplace=True)


    print("Reordering columns to place 'GO_ID' second...")
    cols = complete_df.columns.tolist() # Get current columns as a list
    if 'GO_ID' in cols:
        cols.remove('GO_ID')          # Remove GO_ID from its current position
        cols.insert(1, 'GO_ID')       # Insert GO_ID at index 1 (the second position)
        complete_df = complete_df[cols] # Re-index the DataFrame with the new column order
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds.")
    return complete_df

##############################################################################
######################### PREPROCESSING REQUIREMENTS #########################
##############################################################################
# The file containing the predictions must be a tsv file with 4 columns
# 'Query_ID': Contains the ID of the protein itself
# 'GO_ID': The GO ID of the prediction, formatted as such GO_0123456 (the _ is mandatory???)
# 'Ontology': The type of aspect of the GO choose between P/M/C (Si può rendere più generalizzato)
# 'Score': A numerical value between 0 and 1. If the prediction tool doesn't use this type of 
# value it must be normalized before being accepted!

def preprocess(model_path, model_name, propagate, owl_file, dir_tree):

    df = pd.read_csv(model_path, sep='\t')

    if 'Aspect' in df.columns:
        df.rename(columns={'Aspect': 'Ontology'}, inplace=True)

    # Replace '_' for compatibility
    df = df[['Query_ID', 'GO_ID', 'Ontology', 'Score']]
    df["GO_ID"] = df["GO_ID"].str.replace("_", ":") 
    # The aspect triplet needs to be C/M/P!
    # Triplet descriptive
    df["Ontology"] = df["Ontology"].str.replace("Cellular Component", "C") 
    df["Ontology"] = df["Ontology"].str.replace("Molecular Function", "M") 
    df["Ontology"] = df["Ontology"].str.replace("Biological Process", "P")
    # Triplet CC/MF/BP
    df["Ontology"] = df["Ontology"].str.replace("CC", "C") 
    df["Ontology"] = df["Ontology"].str.replace("MF", "M") 
    df["Ontology"] = df["Ontology"].str.replace("BP", "P") 
    # Triplet C/F/B
    df["Ontology"] = df["Ontology"].str.replace("F", "M") 
    df["Ontology"] = df["Ontology"].str.replace("B", "P")

    # Remove obsolete GOs
    df = substitute_obsolete_go(df, dir_tree['owl_dir_path'])  # New and improved version
    print('[INFO] --- Obsolete annotations substituted ---')
    
    if propagate:
        df = propagated_existing_annotations_fast(owl_file, df)
    print('[INFO] --- Annotations propagated ---')

    # Split between the CAFA4 and CAFA5 datasets
    cafa_types = []
    column_ID = "Query_ID"
    pattern_C4 = r"^T[0-9]{5,}"  # C4 ids always start with T and some numbers

    match_C4 = df[column_ID].str.match(pattern_C4, na=False).any()
    match_C5 = (~df[column_ID].str.match(pattern_C4, na=False)).any()

    # Check the content of the predictions
    if match_C4:
        cafa_types.append("C4")
    if match_C5:
        cafa_types.append("C5")


    for cafa in cafa_types:
        if cafa == "C5":
            filtered_df = df[~df[column_ID].str.match(pattern_C4, na=False)]
        elif cafa == "C4":
            filtered_df = df[df[column_ID].str.match(pattern_C4, na=False)]
        else:
            print("[ERROR]  --- Unrecognized CAFA in preprocessing (This should never happen!) ---")
            
        output_file = dir_tree['prep_preds_dir_path'] + f'/preds_{model_name}_{cafa}_preprocessed.tsv'
        aliases_nx = dir_tree['prep_data_dir_path'] + f'/aliases_{cafa}_nx.tsv'  

        # Preprocess the Query ID through aliasing
        aliases = pd.read_csv(aliases_nx, sep='\t')
        id_map = dict(zip(aliases['match_IDs'], aliases['Query_ID']))
        # Replace IDs in filtered_df using the map
        filtered_df['Query_ID'] = filtered_df['Query_ID'].map(id_map).fillna(filtered_df['Query_ID']).astype(str)
        
        # Save the filtered DataFrame
        filtered_df = filtered_df.sort_values(by=['Query_ID', 'GO_ID', 'Score'], ascending=[True, True, False])
        filtered_df = filtered_df.drop_duplicates(subset=['Query_ID', 'GO_ID'], keep='first')  # Keep the one with the highest score
        filtered_df.to_csv(output_file, index=False, sep='\t')

        print(f'[INFO] --- {cafa} data processed ---')


    print("[SUCCESS] --- Preprocessed data saved successfully! ---")

