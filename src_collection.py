## Collection of all the scripts used for the pipeline

import pandas as pd
import time
import networkx as nx
from owlLibrary3 import GoOwl
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt
from owlready2 import get_ontology
from rdflib import Graph
import json
from tqdm import tqdm
import subprocess
import os
from pathlib import Path


def ensure_directory_exists(directory_path):
    """Check if a directory exists and create it if it doesn't."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def colon_to_underscore(go_string):
    if ':' in go_string:  # Change colon to underscore
        return go_string.replace(':', '_')
    return go_string  # Return unchanged if ':' is not present


def underscore_to_colon(go_string):
    if '_' in go_string:  # Change underscore to colon
        return go_string.replace('_', ':')
    return go_string  # Return unchanged if '_' is not present


def filter_diamond(cafa, filename, outfile, further_inv_file='', check_further=False, k=150, to_clean=False, add_header=False):
    print(f'Clean: {to_clean}')
    if to_clean:
        if add_header:
            df = pd.read_csv(filename, sep='\t', header=None, names=[
            'Query_ID', 'Subject_ID', 'Percentage_matches', 'Alignment_length', 'Mismatches', 'Gap_openings', 
            'Start_of_alignment_in_query', 'End_of_alignment_in_query', 'Start_of_alignment_in_subject', 
            'End_of_alignment_in_subject', 'Expected_value', 'Bit_score'], comment="#")
        else:
            df = pd.read_csv(filename, sep='\t', comment="#")
        print('Pre filtering:')
        print(df.head())

        # Extract the ID and Taxon value from 'Subject_ID' (extracted from pipes)
        df['match_IDs'] = df['Subject_ID'].str.split('|').str[1]
        df['Taxon'] = df['Subject_ID'].str.split('|').str[2]
        
        first_row = df.iloc[0]
        # Loop through the items in the Series
        print('Manual column printing:')
        print('-----------------------------')
        for column_name, value in first_row.items():
            print(f"Column '{column_name}': {value}")
        print('-----------------------------\n\n')
    else:
        if add_header:
            df = pd.read_csv(filename, sep='\t', header=None, names=[
            'Query_ID', 'Subject_ID', 'Percentage_matches', 'Alignment_length', 'Mismatches', 'Gap_openings', 
            'Start_of_alignment_in_query', 'End_of_alignment_in_query', 'Start_of_alignment_in_subject', 
            'End_of_alignment_in_subject', 'Expected_value', 'Bit_score'], comment="#")
        else:
            df = pd.read_csv(filename, sep='\t', comment="#")
        print('Pre filtering:')
        print(df.head())

    # Apply the filtering conditions, identical sequences with no gaps or mismatches
    df = df[(df['Percentage_matches'] >= 100) & (df['Mismatches'] == 0) & 
            (df['Start_of_alignment_in_query'] == df['Start_of_alignment_in_subject']) & 
            (df['End_of_alignment_in_query'] == df['End_of_alignment_in_subject'])]
    
    df = df.rename(columns={"Query_ID": "CAFA4_ID"})
    # Generate the Query_ID column by propagating the first Subject_ID that is found
    df['Query_ID'] = df.groupby('CAFA4_ID')['match_IDs'].transform('first')


    print('After filtering:')
    print(df.head())
    df.to_csv(outfile, sep='\t', index=False)

    if cafa == 'C5':
        value_counts = df['Query_ID'].value_counts() # Count the occurrences of each unique value
    elif cafa == 'C4':
        value_counts = df['CAFA4_ID'].value_counts() # Count the occurrences of each unique value
    print(value_counts)

    if check_further:
        is_empty = False
        more_than_k = value_counts[value_counts >= k] # Filter to find values with counts greater than k
        print(more_than_k)
        if more_than_k.empty:
            is_empty = True
        
        if not is_empty:            
            IDs_more_than_k = value_counts[value_counts >= k].index.tolist()  # Create the list that requires further investigation
            print(IDs_more_than_k)
            print(f"Values with more than {k} appearances:")
            print(more_than_k)
        
            with open(further_inv_file, "w") as file:
                for item in IDs_more_than_k:
                    file.write(f"{item}\n")
                    
        return is_empty
    

def make_db(uniprot_abs_path, db_path):
    subprocess.run(["bash", "makedb.sh", uniprot_abs_path, db_path])


# SCRIPTS FOR CAFA 5
def diamond_management_C5(dir_tree, uniprot_merged_path, cores=32):    

    chunked_C5_dir = os.path.join(dir_tree['pc5_dir_path'], "chunked_results_cafa5") 
    ensure_directory_exists(chunked_C5_dir)
    dmnd_chunks_path = dir_tree['dmnd_data_path'] + '/chunked_cafa5'
    subprocess.run(["bash", "diamond_chunked_cafa5.sh", dmnd_chunks_path, uniprot_merged_path, chunked_C5_dir, str(cores)])

    ####   Concat all the pieces of diamond search to a single file   ####
    # File to save the concatenated result
    concat_file = chunked_C5_dir + '/concat_diamond_cafa5.tsv'
    
    concat_file = chunked_C5_dir + '/concat_diamond_cafa5.tsv'
    diamond_columns = [
    'Query_ID', 'Subject_ID', 'Percentage_matches', 'Alignment_length', 'Mismatches', 'Gap_openings', 
    'Start_of_alignment_in_query', 'End_of_alignment_in_query', 'Start_of_alignment_in_subject', 
    'End_of_alignment_in_subject', 'Expected_value', 'Bit_score']
    combined_dataframes = []
    combined_df = pd.DataFrame()
    for i in range(1, 11):
        filename = chunked_C5_dir + f"/cafa5_k150_out_chunk_{i}.tsv"
        df = pd.read_csv(filename, sep='\t', comment='#', header=None, names=diamond_columns, skiprows=1)
        combined_dataframes.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(combined_dataframes, ignore_index=True)

    # Write the combined TSV
    combined_df.to_csv(concat_file, sep='\t', index=False)

    print(f"Concatenated file saved to {concat_file}")
    print(combined_df.head(5))

    filtered_concat = chunked_C5_dir + '/filtered_concat_cafa5.tsv'
    further_inv_IDs = chunked_C5_dir + '/further_investigation_cafa5_IDs.tsv'
    empty_inv_1 = filter_diamond('C5', concat_file, filtered_concat, further_inv_IDs, True, 150, True)
    
    
    if not empty_inv_1:  # Check if there is no further need to expand the search
        # Takes all the proteins that requires further investigations and fetches the data from
        # the CAFA5 (or CAFA4) fasta to create a query file for diamond

        # Define file paths
        fasta_file = "diamond_data/testsuperset_cafa5.fasta" 
        query_file_fur_inv = chunked_C5_dir + "/cafa5_inv_10000_query.fasta" 

        # Read access IDs from the txt file
        with open(further_inv_IDs, 'r') as txt:
            access_ids = {line.strip() for line in txt if line.strip()} 

        # Filter the FASTA file
        filtered_records = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            # Check if the access ID is in the txt list
            if record.id in access_ids:
                filtered_records.append(record)

        # Write the filtered records to an output FASTA file
        SeqIO.write(filtered_records, query_file_fur_inv, "fasta")

        print(f"Filtered {len(filtered_records)} records. Saved to {query_file_fur_inv}.")
        
        # Subprocessing bash per la query aggiuntiva
        
        ## Round 2: search for 10000 hits in the Uniprot DB
    
        queries_1 = 10000
        further_inv_dmnd = chunked_C5_dir + '/further_inv_cafa5.tsv'
        
        subprocess.run(["bash", "diamond_chunked_fur_inv_cafa.sh", query_file_fur_inv, uniprot_merged_path, further_inv_dmnd, str(queries_1), str(cores)])
        
        filtered_inv1 = chunked_C5_dir + '/further_investigation_cafa5_inv1.tsv'
        further_inv_IDs_2 = chunked_C5_dir + '/further_investigation_cafa5_IDs_2.tsv'
        empty_inv_2 = filter_diamond('C5', further_inv_dmnd, filtered_inv1, further_inv_IDs_2, True, queries_1, True, True)
    
        if not empty_inv_2:
            query_file_fur_inv2 = chunked_C5_dir + f"/cafa5_inv_1M_query.fasta" 
            # Read access IDs from the txt file
            with open(further_inv_IDs_2, 'r') as txt:
                access_ids = {line.strip() for line in txt if line.strip()} 

            # Filter the FASTA file
            filtered_records = []
            for record in SeqIO.parse(fasta_file, "fasta"):
                # Check if the access ID is in the txt list
                if record.id in access_ids:
                    filtered_records.append(record)

            # Write the filtered records to an output FASTA file
            SeqIO.write(filtered_records, query_file_fur_inv2, "fasta")

            print(f"Filtered {len(filtered_records)} records. Saved to {query_file_fur_inv2}.")
            
            ## Round 3: search for 1m hits in the Uniprot DB
            queries_2 = 1000000
            further_inv_dmnd_2 = chunked_C5_dir + '/further_inv_cafa5_2.tsv'
            subprocess.run(["bash", "diamond_chunked_fur_inv_cafa.sh", query_file_fur_inv2, uniprot_merged_path, further_inv_dmnd_2, str(queries_2), str(cores)])
            
            filtered_inv2 = chunked_C5_dir + '/further_investigation_cafa5_inv2.tsv'
            filter_diamond('C5', further_inv_dmnd_2, filtered_inv2, "", False, queries_2, True, True)

    
    benchmark_c5_dir = os.path.join(dir_tree['pc5_dir_path'], "benchmarker_cafa5")
    ensure_directory_exists(benchmark_c5_dir)
    filtered_complete_cafa5 = benchmark_c5_dir + '/filtered_chunked_cafa5_1.tsv'
    # Open the output file for writing
    if empty_inv_1:
        df1 = pd.read_csv(filtered_concat, sep='\t')
        combined_df = df1
    elif empty_inv_2:
        df1 = pd.read_csv(filtered_concat, sep='\t')
        df2 = pd.read_csv(filtered_inv1, sep='\t')
        combined_df = pd.concat([df1, df2])
    else:
        df1 = pd.read_csv(filtered_concat, sep='\t')
        df2 = pd.read_csv(filtered_inv1, sep='\t')
        df3 = pd.read_csv(filtered_inv2, sep='\t')
        combined_df = pd.concat([df1, df2, df3])
    
    # Remove duplicate rows
    unique_df = combined_df.drop_duplicates()

    # Sort rows alphabetically by the first column
    sorted_df = unique_df.sort_values(by=unique_df.columns[0])

    # Save the result to a new TSV file
    sorted_df.to_csv(filtered_complete_cafa5, sep='\t', index=False)
    
    

def find_aliases_from_filtered_diamond_2_C5(filename, outfile):
    print("Running: find_aliases_from_filtered_diamond_2_C5 ...")
    # Retrieve all the aliases aligned using the diamond search
    df = pd.read_csv(filename, sep='\t')

    result= df[['Query_ID', 'match_IDs']]
    result.to_csv(outfile, sep='\t', index=False)



def go_retriever_diamond_chunks_3_C5_fast(goa, filename, outfile, chunk_size):
    print("Running FAST version: go_retriever_diamond_chunks_3_C5 ...")
    start_time = time.time()

    # --- 1. PRE-COMPUTATION & SETUP ---
    print(f"Reading DIAMOND results from {filename}...")
    df_diamond = pd.read_csv(filename, sep='\t')

    query_ids = set(df_diamond['Query_ID'].dropna().unique())
    match_ids = set(df_diamond['match_IDs'].dropna().unique())
    target_ids_set = query_ids | match_ids
    print(f"Created a consolidated set of {len(target_ids_set)} unique target protein IDs.")

    # Setup for reading the large GOA file
    columns = ['DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
               'DB_Reference', 'Evidence_Code', 'With_or_From', 'Aspect',
               'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type',
               'Taxon_and_Interacting_Taxon', 'Date', 'Assigned_by',
               'Annotation_extension', 'Gene_Product_Form_ID']
    dtype_dict = {col: "str" for col in columns}
    df_iterator_goa = pd.read_csv(goa, sep='\t', names=columns, dtype=dtype_dict, chunksize=chunk_size,
                                  compression='gzip', comment="!", low_memory=False)
    
    # List to hold the small, relevant chunks of annotations
    relevant_chunks = []

    # --- 2. FILTER-IN-A-LOOP (Fast and Memory-Efficient) ---
    print("Scanning GOA file for relevant annotations...")
    for i, df_chunk in enumerate(df_iterator_goa):
        iter_start_time = time.time()
        
        # Apply initial filters (Qualifier is more likely to have 'NOT' than GO_ID)
        df_chunk = df_chunk[df_chunk['DB'] == 'UniProtKB']
        df_chunk = df_chunk[~df_chunk['Qualifier'].str.contains('NOT', na=False)]
        
        # This is the most important step: keep only rows matching our consolidated set of IDs.
        relevant_annotations = df_chunk[df_chunk['DB_Object_ID'].isin(target_ids_set)]
        
        # If we found any matches in this chunk, add them to our list
        if not relevant_annotations.empty:
            relevant_chunks.append(relevant_annotations)
        
        print(f"Chunk {i+1}: Found {len(relevant_annotations)} relevant annotations. Time: {time.time() - iter_start_time:.2f}s")
        
    # --- 3. ASSEMBLE, MERGE, AND PROCESS ONCE ---
    if not relevant_chunks:
        print("No matching annotations found in the GOA file. Exiting.")
        return

    print("Concatenating all relevant annotations...")
    # Create one single DataFrame of all found annotations
    all_annotations = pd.concat(relevant_chunks, ignore_index=True)
    # Drop duplicates at the source to make the final merge faster
    all_annotations.drop_duplicates(inplace=True)

    print("Performing final merges...")
    # Perform two merges, but only against the pre-filtered annotation table.
    match_data1 = df_diamond[['Query_ID', 'match_IDs']].merge(
        all_annotations, left_on='Query_ID', right_on='DB_Object_ID', how='inner'
    )
    match_data2 = df_diamond[['Query_ID', 'match_IDs']].merge(
        all_annotations, left_on='match_IDs', right_on='DB_Object_ID', how='inner'
    )

    print("Combining and cleaning final results...")
    # Combine the two sets of matches
    result = pd.concat([match_data1, match_data2], ignore_index=True)
    
    # Drop any fully duplicated rows that might have resulted from the concat
    result.drop_duplicates(inplace=True)

    # We can now drop the join key 'DB_Object_ID' if it's not needed in the final output
    if 'DB_Object_ID' in result.columns:
        result.drop(columns=['DB_Object_ID'], inplace=True)

    # --- 4. SAVE FINAL OUTPUT ONCE ---
    print(f"Sorting and saving final results to {outfile}...")
    # Sort the final dataset just once
    result.sort_values(by='Query_ID', inplace=True, kind='mergesort') # 'mergesort' is stable
    
    # Save the complete, sorted file just once
    result.to_csv(outfile, sep="\t", index=False, header=True, compression='gzip')

    time_total = time.time() - start_time
    print(f"Total time required: {time_total:.2f}s")
    

   

def double_evco_tagger_chunks_4_C5(df_new, df_old, outfile, chunk_size):  # vwersione v1 o v2???
    print("Running: double_evco_tagger_chunks_4_C5 ...")
    columns = ['Query_ID', 'GO_ID', 'Evidence_Code', 'DB_Object_Symbol', 'DB_Object_Type', 
        'Taxon_and_Interacting_Taxon', 'Date', 'match_IDs',
    ]

    # Create a dictionary for dtypes
    dtype_dict = {col: "str" for col in columns}  # Set all columns to string
    df_iterator_2024 = pd.read_csv(df_new, dtype=dtype_dict, chunksize=chunk_size, sep='\t')  # DataFrame with EvCo_2024 (new), chunked
    df_iterator_2023 = pd.read_csv(df_old, dtype=dtype_dict, chunksize=chunk_size, sep='\t')  # DataFrame with EvCo_2023 (old), chunked

    start_time = time.time()
    for j, df_chunk_2024 in enumerate(df_iterator_2024):
        start_iter_2024 = time.time()
        print(f"macro chunk {j}")
        df_chunk_2024.rename(columns={'Evidence_Code': 'EvCo_2024'}, inplace=True)


        for i, df_chunk_2023 in enumerate(df_iterator_2023):
            start_iter = time.time()
            # Set writing mode to append after first chunk
            print(f"chunk {i}")
            mode = 'w' if i == 0 and j==0 else 'a'
            print(df_chunk_2023.head()) if i == 0 and j==0 else ''
            print(df_chunk_2024.head()) if i == 0 and j==0 else ''
            # Add header if it is the first chunk
            if i == 0 and j == 0:
                header = True
            else:
                header = False

            df_chunk_2023.rename(columns={'Evidence_Code': 'EvCo_2023'}, inplace=True)

            # 2023
            num_rows_2023, num_columns_2023 = df_chunk_2023.shape
            print(f"DF old")
            print(f"Number of rows: {num_rows_2023}")
            print(f"Number of columns: {num_columns_2023}")

            # Merge DataFrames to find exact matches
            merged_df = pd.merge(df_chunk_2024, df_chunk_2023[['Query_ID', 'GO_ID', 'EvCo_2023']], on=['Query_ID', 'GO_ID'], how='outer')
            print(f"Time required for merging: {time.time()-start_iter}s")
            
            num_rows_merged, num_columns = merged_df.shape
            print(f"Number of original rows: {num_rows_merged}")
            
            # Add logic for finer control
            def resolve_evco_2023(row, lookup_df):
                if not pd.isna(row['EvCo_2023']):  # If exact match found return the same Evidence Code
                    return row['EvCo_2023']
                elif row['Query_ID'] in lookup_df['Query_ID'].values:  # If ID exists but GO does not match
                    return 'GO_NF'  # GO Not Found
                else:  # If ID does not exist
                    return 'GP_NF'  # Gene Product Not Found

            # Apply the custom logic
            merged_df['EvCo_2023'] = merged_df.apply(lambda row: resolve_evco_2023(row, df_chunk_2023), axis=1)

            # Print the final DataFrame with EvCo_2024 and controlled EvCo_2023
            print(merged_df[['Query_ID', 'GO_ID', 'EvCo_2024', 'EvCo_2023']].head())
            print(merged_df.head())
            merged_df.to_csv(outfile, sep="\t", index=False, header=header, mode=mode)
            time_iter = time.time() - start_iter
            print(f"Time required for this iteration: {time_iter}s")
            # Count the occurrences of each unique value in EvCo_2024
            evco_2024_counts = merged_df['EvCo_2024'].value_counts()

            # Count the occurrences of each unique value in EvCo_2023
            evco_2023_counts = merged_df['EvCo_2023'].value_counts()

            print(f"Counts for EvCo_new: {evco_2024_counts} in iteration {i}")
            print(f"\nCounts for EvCo_old: {evco_2023_counts} in iteration {i}")
        
        time_iter_2024 = time.time() - start_iter_2024
        print(f"Time required for this macro iteration: {time_iter_2024}s")

    time_total = time.time() - start_time
    print(f"Total time required: {time_total}s")
    # Print the results

        

def benchmark_selection_5_subcat_C5(filename, outfile, fig_hard, fig_overall, df_hard, df_overall):
    print("Running: benchmark_selection_5_subcat_C5 ...")
    df_tagged = pd.read_csv(filename, sep='\t')  # DataFrame with EvCo_2024

    # Define the unreliable source codes
    unreliable_sources = ['ND', 'ISS', 'IEA']
    extended = ['ND', 'ISS', 'IEA', 'GP_NF', 'GO_NF']
    hard_evidence = ['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP']
    philo_evidence = ['IBA', 'IBD', 'IKR', 'IRD']
    comput_evidence = ['ISO', 'ISA', 'ISM', 'IGC', 'RCA']
    author_evidence = ['TAS', 'IC']
    extendend_evidence = ['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP', 
                          'IBA', 'IBD', 'IKR', 'IRD', 'ISO', 'ISA', 'ISM', 'IGC', 'RCA', 'TAS', 'IC']
    print(df_tagged.head())
    print(df_tagged.columns)
    # Filter out rows with unreliable sources
    filtered_df = df_tagged

    # Define the knowledge_type column based on conditions
    def determine_knowledge_type(row):
        evco_2023 = row['EvCo_2023']
        evco_2024 = row['EvCo_2024']

        if evco_2023 == 'GP_NF' and evco_2024 not in unreliable_sources:
            return 'NK'  # Gene Product previously unknown, but now experimentally tested (NK)
        elif evco_2023 == 'GO_NF' and evco_2024 not in unreliable_sources:
            return 'LK1'  # Gene function previously unknown for this GP but now experimentally verified (LK type 1)
        elif evco_2024 in extendend_evidence and evco_2023 in unreliable_sources:
            return 'LK2'  # Gene function previously known for this GP but now experimentally verified (LK type 2)
        elif row['DB_Object_Symbol'] == 'Heredited':
            return 'Desumed'  # Desumed information
        elif pd.isna(evco_2024) and not pd.isna(evco_2023):
            if evco_2023 in hard_evidence:
                return 'hard_missing'
            elif evco_2023 in philo_evidence:
                return 'philo_missing' 
            elif evco_2023 in comput_evidence:
                return 'comput_missing' 
            elif evco_2023 in author_evidence:
                return 'author_missing' 
            elif evco_2023 == 'ISS':
                return 'iss_missing'
            else:
                return 'misc_missing'
        # elif evco_2024 in unreliable_sources and not pd.isna(evco_2023):
        elif evco_2024 in unreliable_sources and evco_2023 not in unreliable_sources:
            return 'declassed'
        elif evco_2024 in unreliable_sources and evco_2023 in unreliable_sources:
            return 'unreliable'
        # elif evco_2024 not in unreliable_sources and evco_2023 not in unreliable_sources:
        elif evco_2024 not in extended and evco_2023 not in extended:
            return 'KK'   # Previously verified knowledge (KK)
        elif pd.isna(evco_2024) == True and pd.isna(evco_2023) == False or evco_2024 == None and evco_2023 != None:
            return 6  # Deprecated annotation
        else:
            return None  # Default case, if no condition matches (It should never happen!)

    # Apply the function to each row
    filtered_df['knowledge_type'] = filtered_df.apply(determine_knowledge_type, axis=1)
    filtered_df.to_csv(f'{outfile}.test', sep="\t", index=False)
    filtered_df = filtered_df.loc[filtered_df['knowledge_type'] != 'unreliable']
    filtered_df = filtered_df.loc[filtered_df['knowledge_type'] != 'declassed']

    # Count rows for each knowledge_type value
    knowledge_type_counts = filtered_df['knowledge_type'].value_counts()

    # Print the counts
    print("Counts for each knowledge_type:")
    print(knowledge_type_counts)

    hard_count = (filtered_df['knowledge_type'] == 'hard_missing').sum()
    print(f"Number of rows with type 'hard_missing': {hard_count}")

    declassed_count = (filtered_df['knowledge_type'] == 'declassed').sum()
    print(f"Number of rows with type 'declassed': {declassed_count}")

    hard_evidence_count = filtered_df['EvCo_2023'].isin(hard_evidence).sum()
    print(f"Number of rows with hard evidence in old: {hard_evidence_count}")
    print(f"ratio: {hard_count/hard_evidence_count}")

    hardish_evidence = ['hard_missing', 'author_missing']
    hard_missing_counts = filtered_df[filtered_df['knowledge_type'].isin(hardish_evidence)].groupby('Query_ID').size()

    # Calculate average and median
    average_hard_missing = hard_missing_counts.mean()
    median_hard_missing = hard_missing_counts.median()

    print("---------- HARD ----------")
    print(f"Average 'hard_missing' count per Query_ID: {average_hard_missing}")
    print(f"Median 'hard_missing' count per Query_ID: {median_hard_missing}")

    plt.figure(figsize=(8, 5))
    plt.hist(hard_missing_counts, bins=range(1, hard_missing_counts.max() + 2), edgecolor='black', alpha=0.7)
    # Labels and title
    plt.xlabel("Number of missing hard Annotations per ID")
    plt.ylabel("Frequency (Number of IDs)")
    plt.title("Distribution of missing hard Annotations per ID old-new")
    plt.xticks(range(1, hard_missing_counts.max() + 1))  # Ensure integer x-axis ticks
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Save the figure
    plt.savefig(fig_hard, dpi=300, bbox_inches='tight')


    overall_evidence = ['hard_missing', 'author_missing', 'philo_missing', 'declassed']
    overall_missing_counts = filtered_df[filtered_df['knowledge_type'].isin(overall_evidence)].groupby('Query_ID').size()

    # Calculate average and median
    average_overall_missing = overall_missing_counts.mean()
    median_overall_missing = overall_missing_counts.median()
    print("---------- OVERALL ----------")
    print(f"Average overall count per Query_ID: {average_overall_missing}")
    print(f"Median overall count per Query_ID: {median_overall_missing}")

    top_20_overall_missing_ids = overall_missing_counts.nlargest(20)
    print("Top 20 IDs with the most overall annotations:")
    print(top_20_overall_missing_ids)


    plt.figure(figsize=(8, 5))
    plt.hist(overall_missing_counts, bins=range(1, overall_missing_counts.max() + 2), edgecolor='black', alpha=0.7)
    # Labels and title
    plt.xlabel("Number of overall missing Annotations per ID")
    plt.ylabel("Frequency (Number of IDs)")
    plt.title("Distribution of overall missing Annotations per ID old-new")
    plt.xticks(range(1, overall_missing_counts.max() + 1))  # Ensure integer x-axis ticks
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Save the figure
    plt.savefig(fig_overall, dpi=300, bbox_inches='tight')

    # Save the updated DataFrame to a new CSV
    overall_missing_counts.to_csv(df_overall, sep='\t', header=True)
    hard_missing_counts.to_csv(df_hard, sep='\t', header=True)
    filtered_df.to_csv(outfile, sep="\t", index=False)
    
    print(f"Lenght dataframe {len(filtered_df)}")




# SCRIPTS FOR CAFA 4

def diamond_management_C4(dir_tree, uniprot_merged_path, cores=32):
    
    chunked_C4_dir = os.path.join(dir_tree['pc4_dir_path'], "chunked_results_cafa4") 
    ensure_directory_exists(chunked_C4_dir)
    
    dmnd_chunks_path = dir_tree['dmnd_data_path'] + '/chunked_cafa4'
    subprocess.run(["bash", "diamond_chunked_cafa4.sh", dmnd_chunks_path, uniprot_merged_path, chunked_C4_dir, str(cores)])

    ####   Concat all the pieces of diamond search to a single file   ####
    # File to save the concatenated result
    concat_file = chunked_C4_dir + '/concat_diamond_cafa4.tsv'
    
    concat_file = chunked_C4_dir + '/concat_diamond_cafa4.tsv'
    diamond_columns = [
    'Query_ID', 'Subject_ID', 'Percentage_matches', 'Alignment_length', 'Mismatches', 'Gap_openings', 
    'Start_of_alignment_in_query', 'End_of_alignment_in_query', 'Start_of_alignment_in_subject', 
    'End_of_alignment_in_subject', 'Expected_value', 'Bit_score']
    combined_dataframes = []
    combined_df = pd.DataFrame()
    for i in range(1, 11):
        filename = chunked_C4_dir + f"/cafa4_k150_out_chunk_{i}.tsv"
        df = pd.read_csv(filename, sep='\t', comment='#', header=None, names=diamond_columns, skiprows=1)
        combined_dataframes.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(combined_dataframes, ignore_index=True)

    # Write the combined TSV
    combined_df.to_csv(concat_file, sep='\t', index=False)

    print(f"Concatenated file saved to {concat_file}")
    print(combined_df.head(5))

    filtered_concat = chunked_C4_dir + '/filtered_concat_cafa4.tsv'
    further_inv_IDs = chunked_C4_dir + '/further_investigation_cafa4_IDs.tsv'
    empty_inv_1 = filter_diamond('C4', concat_file, filtered_concat, further_inv_IDs, True, 150, True)


    if not empty_inv_1:  # Check if there is no further need to expand the search
        # Takes all the proteins that requires further investigations and fetches the data from
        # the CAFA5 (or CAFA4) fasta to create a query file for diamond

        # Define file paths
        fasta_file = "diamond_data/superset_cafa4.fasta" 
        query_file_fur_inv = chunked_C4_dir + "/cafa4_inv_10000_query.fasta" 

        # Read access IDs from the txt file
        with open(further_inv_IDs, 'r') as txt:
            access_ids = {line.strip() for line in txt if line.strip()} 

        # Filter the FASTA file
        filtered_records = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            # Check if the access ID is in the txt list
            if record.id in access_ids:
                filtered_records.append(record)

        # Write the filtered records to an output FASTA file
        SeqIO.write(filtered_records, query_file_fur_inv, "fasta")

        print(f"Filtered {len(filtered_records)} records. Saved to {query_file_fur_inv}.")
        
        ## Round 2: search for 10000 hits in the Uniprot DB
        queries_1 = 10000
        further_inv_dmnd = chunked_C4_dir + '/further_inv_cafa4.tsv'
        
        subprocess.run(["bash", "diamond_chunked_fur_inv_cafa.sh", query_file_fur_inv, uniprot_merged_path, further_inv_dmnd, str(queries_1), str(cores)])
        
        filtered_inv1 = chunked_C4_dir + '/further_investigation_cafa4_inv1.tsv'
        further_inv_IDs_2 = chunked_C4_dir + '/further_investigation_cafa4_IDs_2.tsv'
        empty_inv_2 = filter_diamond('C4', further_inv_dmnd, filtered_inv1, further_inv_IDs_2, True, queries_1, True, True)
    
    if not empty_inv_2:
        query_file_fur_inv2 = chunked_C4_dir + "/cafa4_inv_1M_query.fasta" 
        # Read access IDs from the txt file
        with open(further_inv_IDs_2, 'r') as txt:
            access_ids = {line.strip() for line in txt if line.strip()} 

        # Filter the FASTA file
        filtered_records = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            # Check if the access ID is in the txt list
            if record.id in access_ids:
                filtered_records.append(record)

        # Write the filtered records to an output FASTA file
        SeqIO.write(filtered_records, query_file_fur_inv2, "fasta")

        print(f"Filtered {len(filtered_records)} records. Saved to {query_file_fur_inv2}.")
        
        ## Round 3: search for 1m hits in the Uniprot DB
        queries_2 = 1000000
        further_inv_dmnd_2 = chunked_C4_dir + '/further_inv_cafa4_2.tsv'
        subprocess.run(["bash", "diamond_chunked_fur_inv_cafa.sh", query_file_fur_inv2, uniprot_merged_path, further_inv_dmnd_2, str(queries_2), str(cores)])
        
        filtered_inv2 = chunked_C4_dir + '/further_investigation_cafa4_inv2.tsv'
        filter_diamond('C4', further_inv_dmnd_2, filtered_inv2, "", False, queries_2, True, True)
    
    benchmark_c4_dir = os.path.join(dir_tree['pc4_dir_path'], "benchmarker_cafa4")
    ensure_directory_exists(benchmark_c4_dir)
    filtered_complete_cafa4 = benchmark_c4_dir + '/filtered_chunked_cafa4_1.tsv'
    # Open the output file for writing
    if empty_inv_1:
        df1 = pd.read_csv(filtered_concat, sep='\t')
        combined_df = df1
    elif empty_inv_2:
        df1 = pd.read_csv(filtered_concat, sep='\t')
        df2 = pd.read_csv(filtered_inv1, sep='\t')
        combined_df = pd.concat([df1, df2])
    else:
        df1 = pd.read_csv(filtered_concat, sep='\t')
        df2 = pd.read_csv(filtered_inv1, sep='\t')
        df3 = pd.read_csv(filtered_inv2, sep='\t')
        combined_df = pd.concat([df1, df2, df3])
    
    # Remove duplicate rows
    unique_df = combined_df.drop_duplicates()

    # Sort rows alphabetically by the first column
    sorted_df = unique_df.sort_values(by=unique_df.columns[0])

    # Save the result to a new TSV file
    sorted_df.to_csv(filtered_complete_cafa4, sep='\t', index=False)
    
    

def find_aliases_from_filtered_diamond_2_C4(filename, outfile):
    print("Running: find_aliases_from_filtered_diamond_2_C4 ...")
    # Retrieve all the aliases aligned using the diamond search
    df = pd.read_csv(filename, sep='\t')

    result= df[['Query_ID', 'match_IDs', 'CAFA4_ID']]  # Keep the custom IDs given from CAFA, it may prove useful
    result.to_csv(outfile, sep='\t', index=False)



def go_retriever_diamond_chunks_3_C4_fast(goa, filename, outfile, chunk_size):
    print("Running FAST version: go_retriever_diamond_chunks_3_C4 ...")
    start_time = time.time()

    # --- 1. PRE-COMPUTATION & SETUP ---
    print(f"Reading DIAMOND results from {filename}...")
    df_diamond = pd.read_csv(filename, sep='\t')

    target_ids_set = set(df_diamond['match_IDs'].unique())
    print(f"Created a set of {len(target_ids_set)} unique target protein IDs.")

    # Setup for reading the large GOA file
    columns = ['DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
               'DB_Reference', 'Evidence_Code', 'With_or_From', 'Aspect',
               'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type',
               'Taxon_and_Interacting_Taxon', 'Date', 'Assigned_by',
               'Annotation_extension', 'Gene_Product_Form_ID']
    dtype_dict = {col: "str" for col in columns}
    df_iterator_goa = pd.read_csv(goa, sep='\t', names=columns, dtype=dtype_dict, chunksize=chunk_size,
                                  compression='gzip', comment="!", low_memory=False)

    useful_columns = ['DB_Object_ID', 'GO_ID', 'Evidence_Code', 'DB_Object_Symbol',
                      'Aspect', 'DB_Object_Type', 'Taxon_and_Interacting_Taxon', 'Date']
    
    # List to hold the small, relevant chunks of annotations
    relevant_chunks = []

    # --- 2. FILTER-IN-A-LOOP (Instead of Merge-in-a-loop) ---
    print("Scanning GOA file for relevant annotations...")
    for i, df_chunk in enumerate(df_iterator_goa):
        iter_start_time = time.time()
        
        # Apply initial filters
        df_chunk = df_chunk[df_chunk['DB'] == 'UniProtKB']
        df_chunk = df_chunk[~df_chunk['GO_ID'].str.contains('not', na=False)]
        
        relevant_annotations = df_chunk[df_chunk['DB_Object_ID'].isin(target_ids_set)]
        
        # If we found any matches in this chunk, add them to our list
        if not relevant_annotations.empty:
            relevant_chunks.append(relevant_annotations[useful_columns])
        
        print(f"Chunk {i+1}: Found {len(relevant_annotations)} relevant annotations. Time: {time.time() - iter_start_time:.2f}s")
        
    # --- 3. ASSEMBLE, MERGE, AND PROCESS ONCE ---
    if not relevant_chunks:
        print("No matching annotations found in the GOA file. Saving an empty result file.")
        # Create an empty DataFrame with the correct final columns and save it
        final_cols = df_diamond.columns.tolist() + [col for col in useful_columns if col != 'DB_Object_ID']
        empty_df = pd.DataFrame(columns=final_cols)
        empty_df.to_csv(outfile, sep="\t", index=False, header=True, compression='gzip')
        return

    print("Concatenating all relevant annotations...")
    # Create one single DataFrame of all found annotations
    all_annotations = pd.concat(relevant_chunks, ignore_index=True)
    # Drop duplicates at the source to make the final merge faster
    all_annotations.drop_duplicates(inplace=True)

    print("Performing final merge...")
    # Now, perform the merge operation just ONCE.
    result = df_diamond.merge(all_annotations, left_on='match_IDs', right_on='DB_Object_ID', how='left')

    # Drop rows where the merge failed to find a GO annotation
    result.dropna(subset=['GO_ID'], inplace=True)

    # --- 4. SAVE FINAL OUTPUT ONCE ---
    print(f"Sorting and saving final results to {outfile}...")
    # Sort the final dataset just once
    result.sort_values(by='Query_ID', inplace=True, kind='mergesort') # 'mergesort' is stable
    
    # Save the complete, sorted file just once
    result.to_csv(outfile, sep="\t", index=False, header=True, compression='gzip')

    time_total = time.time() - start_time
    print(f"Total time required: {time_total:.2f}s")


    
def double_evco_tagger_chunks_4_C4(df_new, df_old, outfile, chunk_size):
    print("Running: double_evco_tagger_chunks_4_C4 ...")

    columns = ['Query_ID', 'GO_ID', 'Evidence_Code', 'DB_Object_Symbol', 'DB_Object_Type', 
        'Taxon_and_Interacting_Taxon', 'Date', 'match_IDs', 'CAFA4_ID'
    ]
    # Create a dictionary for dtypes
    dtype_dict = {col: "str" for col in columns}  # Set all columns to string
    chunk_size = chunk_size
    
    df_iterator_2024 = pd.read_csv(df_new, dtype=dtype_dict, chunksize=chunk_size, sep='\t')
    df_iterator_2019 = pd.read_csv(df_old, dtype=dtype_dict, chunksize=chunk_size, sep='\t')

    # EvCo_2024 (new), EvCo_2019 (old)
    start_time = time.time()
    for j, df_chunk_2024 in enumerate(df_iterator_2024):
        start_iter_2024 = time.time()
        print(f"macro chunk {j}")
        df_chunk_2024.rename(columns={'Evidence_Code': 'EvCo_2024'}, inplace=True)


        for i, df_chunk_2019 in enumerate(df_iterator_2019):
            start_iter = time.time()
            # Set writing mode to append after first chunk
            print(f"chunk {i}")
            mode = 'w' if i == 0 and j==0 else 'a'
            print(df_chunk_2019.head()) if i == 0 and j==0 else ''
            print(df_chunk_2024.head()) if i == 0 and j==0 else ''
            # Add header if it is the first chunk
            if i == 0 and j == 0:
                header = True
            else:
                header = False

            df_chunk_2019.rename(columns={'Evidence_Code': 'EvCo_2019'}, inplace=True)

            # 2019
            num_rows_2019, num_columns_2019 = df_chunk_2019.shape
            print(f"DF old")
            print(f"Number of rows: {num_rows_2019}")
            print(f"Number of columns: {num_columns_2019}")

            # Merge DataFrames to find exact matches
            merged_df = pd.merge(df_chunk_2024, df_chunk_2019[['Query_ID', 'GO_ID', 'EvCo_2019']], on=['Query_ID', 'GO_ID'], how='outer')
            print(f"Time required for merging: {time.time()-start_iter}s")
            
            num_rows_merged, num_columns = merged_df.shape
            print(f"Number of original rows: {num_rows_merged}")
            
            # Add logic for finer control
            def resolve_evco_2019(row, lookup_df):
                if not pd.isna(row['EvCo_2019']):  # If exact match found return the same Evidence Code
                    return row['EvCo_2019']
                # elif row['match_IDs'] in lookup_df['match_IDs'].values:  # If ID exists but GO does not match
                elif row['Query_ID'] in lookup_df['Query_ID'].values:  # If ID exists but GO does not match
                    return 'GO_NF'  # GO Not Found
                else:  # If ID does not exist, and the EvCo is not found the gene product did not exist
                    return 'GP_NF'  # Gene Product Not Found

            # Apply the custom logic
            merged_df['EvCo_2019'] = merged_df.apply(lambda row: resolve_evco_2019(row, df_chunk_2019), axis=1)

            # Print the final DataFrame with EvCo_2024 and controlled EvCo_2019
            print(merged_df[['Query_ID', 'GO_ID', 'EvCo_2024', 'EvCo_2019']].head())
            print(merged_df.head())
            merged_df.to_csv(outfile, sep="\t", index=False, header=header, mode=mode)
            
            time_iter = time.time() - start_iter
            print(f"Time required for this iteration: {time_iter}s")
            # Count the occurrences of each unique value in EvCo_2024
            evco_2024_counts = merged_df['EvCo_2024'].value_counts()

            # Count the occurrences of each unique value in EvCo_2019 
            evco_2019_counts = merged_df['EvCo_2019'].value_counts()

            print(f"Counts for EvCo_new: {evco_2024_counts} in iteration {i}")
            print(f"\nCounts for EvCo_old: {evco_2019_counts} in iteration {i}")
        
        time_iter_2024 = time.time() - start_iter_2024
        print(f"Time required for this macro iteration: {time_iter_2024}s")

    time_total = time.time() - start_time
    print(f"Total time required: {time_total}s")

    
    

def benchmark_selection_5_subcat_C4(filename, outfile, fig_hard, fig_overall, df_hard, df_overall):  # vwersione v1 o v2???
    print("Running: benchmark_selection_5_subcat_C4 ...")    
    
    df_tagged = pd.read_csv(filename, sep='\t')  # DataFrame with EvCo_new

    # Define the unreliable source codes
    unreliable_sources = ['ND', 'ISS', 'IEA']
    extended = ['ND', 'ISS', 'IEA', 'GP_NF', 'GO_NF']
    hard_evidence = ['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP']
    philo_evidence = ['IBA', 'IBD', 'IKR', 'IRD']
    comput_evidence = ['ISO', 'ISA', 'ISM', 'IGC', 'RCA']
    author_evidence = ['TAS', 'IC']
    print(df_tagged.head())
    print(df_tagged.columns)
    # Filter out rows with unreliable sources
    filtered_df = df_tagged

    # Define the knowledge_type column based on conditions
    def determine_knowledge_type(row):
        evco_2019 = row['EvCo_2019']
        evco_2024 = row['EvCo_2024']

        if evco_2019 == 'GP_NF' and evco_2024 not in unreliable_sources:
            return 'NK'  # Gene Product previously unknown, but now experimentally tested (NK)
        elif evco_2019 == 'GO_NF' and evco_2024 not in unreliable_sources:
            return 'LK1'  # Gene function previously unknown for this GP but now experimentally verified (LK type 1)
        elif evco_2024 not in unreliable_sources and evco_2019 in unreliable_sources:
            return 'LK2'  # Gene function previously known for this GP but now experimentally verified (LK type 2)
        elif row['DB_Object_Symbol'] == 'Heredited':
            return 'Desumed'  # Desumed information
        elif pd.isna(evco_2024) and not pd.isna(evco_2019):
            if evco_2019 in hard_evidence:
                return 'hard_missing'
            elif evco_2019 in philo_evidence:
                return 'philo_missing' 
            elif evco_2019 in comput_evidence:
                return 'comput_missing' 
            elif evco_2019 in author_evidence:
                return 'author_missing' 
            elif evco_2019 == 'ISS':
                return 'iss_missing'
            else:
                return 'misc_missing' 
        # elif evco_2024 in unreliable_sources and not pd.isna(evco_2019):
        elif evco_2024 in unreliable_sources and evco_2019 not in unreliable_sources:
            return 'declassed'
        elif evco_2024 in unreliable_sources and evco_2019 in unreliable_sources:
            return 'unreliable'
        elif evco_2024 not in extended and evco_2019 not in extended:
            return 'KK'   # Previously verified knowledge (KK)
        elif pd.isna(evco_2024) == True and pd.isna(evco_2019) == False or evco_2024 == None and evco_2019 != None:
            return 6  # Deprecated annotation
        else:
            return None  # Default case, if no condition matches (It should never happen!)

    # Apply the function to each row
    filtered_df['knowledge_type'] = filtered_df.apply(determine_knowledge_type, axis=1)

    # Count rows for each knowledge_type value
    knowledge_type_counts = filtered_df['knowledge_type'].value_counts()

    # Print the counts
    print("Counts for each knowledge_type:")
    print(knowledge_type_counts)

    hard_count = (filtered_df['knowledge_type'] == 'hard_missing').sum()
    print(f"Number of rows with type 'hard_missing': {hard_count}")

    declassed_count = (filtered_df['knowledge_type'] == 'declassed').sum()
    print(f"Number of rows with type 'declassed': {declassed_count}")

    hard_evidence_count = filtered_df['EvCo_2019'].isin(hard_evidence).sum()
    print(f"Number of rows with hard evidence in 2019: {hard_evidence_count}")
    print(f"ratio: {hard_count/hard_evidence_count}")

    hardish_evidence = ['hard_missing', 'author_missing']
    hard_missing_counts = filtered_df[filtered_df['knowledge_type'].isin(hardish_evidence)].groupby('Query_ID').size()

    # Calculate average and median
    average_hard_missing = hard_missing_counts.mean()
    median_hard_missing = hard_missing_counts.median()
    print("---------- HARD ----------")
    print(f"Average 'hard_missing' count per Query_ID: {average_hard_missing}")
    print(f"Median 'hard_missing' count per Query_ID: {median_hard_missing}")

    # top_20_hard_missing_ids = hard_missing_counts.nlargest(20)

    plt.figure(figsize=(8, 5))
    plt.hist(hard_missing_counts, bins=range(1, hard_missing_counts.max() + 2), edgecolor='black', alpha=0.7)
    # Labels and title
    plt.xlabel("Number of missing hard Annotations per ID")
    plt.ylabel("Frequency (Number of IDs)")
    plt.title("Distribution of missing hard Annotations per ID old-new")
    plt.xticks(range(1, hard_missing_counts.max() + 1))  # Ensure integer x-axis ticks
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Save the figure
    plt.savefig(fig_hard, dpi=300, bbox_inches='tight')

    overall_evidence = ['hard_missing', 'author_missing', 'philo_missing', 'declassed']
    overall_missing_counts = filtered_df[filtered_df['knowledge_type'].isin(overall_evidence)].groupby('Query_ID').size()

    # Calculate average and median
    average_overall_missing = overall_missing_counts.mean()
    median_overall_missing = overall_missing_counts.median()
    print("---------- OVERALL ----------")
    print(f"Average overall count per Query_ID: {average_overall_missing}")
    print(f"Median overall count per Query_ID: {median_overall_missing}")

    top_20_overall_missing_ids = overall_missing_counts.nlargest(20)
    print("Top 20 IDs with the most overall annotations:")
    print(top_20_overall_missing_ids)

    plt.figure(figsize=(8, 5))
    plt.hist(overall_missing_counts, bins=range(1, overall_missing_counts.max() + 2), edgecolor='black', alpha=0.7)
    # Labels and title
    plt.xlabel("Number of overall missing Annotations per ID")
    plt.ylabel("Frequency (Number of IDs)")
    plt.title("Distribution of overall missing Annotations per ID old-new")
    plt.xticks(range(1, overall_missing_counts.max() + 1))  # Ensure integer x-axis ticks
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Save the figure
    plt.savefig(fig_overall, dpi=300, bbox_inches='tight')

    # Save the updated DataFrame to a new CSV
    overall_missing_counts.to_csv(df_overall, sep='\t', header=True)
    hard_missing_counts.to_csv(df_hard, sep='\t', header=True)
    filtered_df.to_csv(outfile, sep="\t", index=False)
    # print(filtered_df.head()) 
    print(f"Lenght dataframe {len(filtered_df)}")

        
        
    
    
# GENERAL PURPOSE SCRIPTS


def remove_duplicates_and_roots_3_1(filename, outfile):
    print("Running: remove_duplicates_and_roots_3_1 ...")
    # Load the data
    df = pd.read_csv(filename, sep='\t', compression='gzip')
    print(df.head(5))
    print(df.columns.tolist())
    if 'DB_Object_ID' in df.columns.tolist():
        df = df.drop(columns=['DB_Object_ID'])
    # Drop duplicates
    df = df.drop_duplicates() 

    roots = ["GO_0005575", "GO_0008150", "GO_0003674"]
    df = df[~df['GO_ID'].isin(roots)]

    # Save the cleaned DataFrame to a gzipped TSV file
    df.to_csv(outfile, sep="\t", index=False)       
    
    
#######################################################
def go_quality_selection_chunks_v2_3_5(filename, outfile, chunk_size):
    print("Running: go_quality_selection_chunks_v2_3_5 ...")
    # Used to keep only the most recent experimental data
    df_iterator = pd.read_csv(filename, chunksize=chunk_size, sep='\t')
    for j, df_chunk in enumerate(df_iterator):
        print(f"Chunk {j}")
        mode = 'w' if j == 0 else 'a'
        if j == 0:
            header = True
            print(df_chunk.head())
            print(df_chunk.columns)
        else:
            header = False
        values_to_remove = ['IEA', 'ND', 'ISS']

        # Convert 'Date' column to datetime for comparison
        df_chunk['Date'] = pd.to_datetime(df_chunk['Date'], format='%Y%m%d')

        # Define the filtering logic for each group
        # Each group contains only one type of GO_ID
        # Remove all the GOs without experimental evidence and just keep the most recent experimental one
        # if that is not possible, then keep the most recent one even if is not experimental 
        def filter_group(group):
            # Sort the group by date (latest first)
            group = group.sort_values(by='Date', ascending=False)
            not_in_values = group[~group['Evidence_Code'].isin(values_to_remove)]
            if not_in_values.empty:
                # If no such row exists, return the latest row regardless
                return group.iloc[0]
            return not_in_values.iloc[0]

        filtered_rows = []
        # Group equal GO_IDs that appear in the same gene product
        for _, group in df_chunk.groupby(['Query_ID', 'GO_ID']):
            result = filter_group(group)
            if isinstance(result, pd.DataFrame):  # Ensure result is a DataFrame for append
                filtered_rows.append(result)
            else:  # If a single row is returned
                filtered_rows.append(result.to_frame().T)

        filtered_df = pd.concat(filtered_rows, ignore_index=True)
        
        # Save the modified DataFrame back to a TSV file
        filtered_df.to_csv(outfile, sep="\t", index=False, header=header, mode=mode)
        
        
    
#######################################################
def go_quality_selection_round2_3_6(filename, outfile):
    print("Running: go_quality_selection_round2_3_6 ...")
    # Rerun of the previous script without the chunking to remove remaining redundancy
    df = pd.read_csv(filename, sep='\t')
    print(df.head())
    
    values_to_remove = ['IEA', 'ND', 'ISS']

    df_sorted = df.sort_values('Date', ascending=False)

    # 2. Identify all rows that have the desired evidence codes
    valid_mask = ~df_sorted['Evidence_Code'].isin(values_to_remove)

    # 3. Get the first valid row for each group.
    #    Since the df is sorted, .first() gets the latest valid row.
    group_cols = ['Query_ID', 'GO_ID']
    valid_selections = df_sorted[valid_mask].groupby(group_cols).first()

    # 4. Get the absolute first row for each group (our fallback).
    fallback_selections = df_sorted.groupby(group_cols).first()

    # 5. Combine the results. Use valid_selections where they exist, otherwise fill with the fallback.
    filtered_df = valid_selections.combine_first(fallback_selections).reset_index()

    # Coerce date back to datetime if needed, as .combine_first can change dtypes
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

    # Save the modified DataFrame back to a TSV file
    filtered_df.to_csv(outfile, sep="\t", index=False)
    
    
    
    
def substitute_obsolete_go_3_61(filename, owl_data_dir, outfile):
    # Substitution and removal of obslete goas from the old annotation for compatibility purposes
    print("Running: substitute_obsolete_go_3_61 ...")
    
    data_df = pd.read_csv(filename, sep='\t')  # Load the main data fill
    
    # The dicts use the standard GO_0123456 NOT GO:0123456!
    data_df['GO_ID'] = data_df['GO_ID'].str.replace(':', '_')

    # # Create a dictionary mapping obsolete GOs to lists of new GOs
    
    with open(owl_data_dir + "depr.json", "r") as file:
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
    
    
    with open(owl_data_dir + "obs.json", "r") as file:
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
    updated_df_final.to_csv(outfile, sep='\t', index=False)



def remove_IEA_3_69(filename, outfile):
    df = pd.read_csv(filename, sep='\t')
    # List of types to remove
    values_to_remove = ['IEA', 'ND', 'ISS']
    # Filter the DataFrame
    df_filtered = df[~df['Evidence_Code'].isin(values_to_remove)]
    df_filtered.to_csv(outfile, sep='\t', index=False)
    


def propagated_existing_annotations_6_1_fast(owl_file, datafile, outfile):
    print("Running FAST version: propagated_existing_annotations_6_1 fast...")
    start_time = time.time()

    # --- 1. Setup and Pre-computation ---
    owl = GoOwl(owl_file, goa_file="", by_ontology=True)
    df = pd.read_csv(datafile, sep='\t')

    cafa4_check = 'CAFA4_ID' in df.columns

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
    base_columns = ["Query_ID", "EvCo_2024", "DB_Object_Symbol", "Aspect", "Taxon_and_Interacting_Taxon", "match_IDs"]
    if cafa4_check:
        base_columns.append("CAFA4_ID")

    print("Propagating annotations...")
    # Iterate using `to_dict('records')` which is much faster than iterrows
    for row in df.to_dict('records'):
        current_go_id = row['GO_ID']
        
        # Prepare a base dictionary for this row's data to avoid repetition
        base_row_data = {col: row.get(col) for col in base_columns}
        base_row_data['Date'] = row.get('Date', "2023-12-31 01:23:45")
        if pd.isna(base_row_data['Date']):
             base_row_data['Date'] = "2023-12-31 01:23:45"
        base_row_data['knowledge_type'] = "KK" if row.get('knowledge_type') == "Desumed" else row.get('knowledge_type', "Heredited")

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

    MISSING_VALUES = {None, np.nan, ''} 

    for row_dict in propagated_rows:
        # print(row_dict)
        current_value = row_dict.get('match_IDs')
        
        # Check if the current value is in our set of missing values.
        # We also check if stripping a string value makes it empty.
        is_missing = current_value in MISSING_VALUES
        if not is_missing and isinstance(current_value, str):
            is_missing = not current_value.strip() # True if string is only whitespace

        if is_missing:
            row_dict['match_IDs'] = 'Heredited'


    # --- 3. Create DataFrame in One Go (Highly Optimized) ---
    print(f"Creating final DataFrame from {len(propagated_rows)} propagated rows...")
    complete_df = pd.DataFrame(propagated_rows)

    # The original GO_ID might have underscores, standardize them all now
    complete_df['GO_ID'] = complete_df['GO_ID'].str.replace('_', ':')
    
    # --- 4. Final Processing (This part was already well-vectorized) ---
    print("Sorting and dropping duplicates...")
    # Drop duplicates first to reduce the amount of data to sort
    complete_df.drop_duplicates(subset=['Query_ID', 'GO_ID', 'knowledge_type'], inplace=True)

    priority_order = {"KK": 1, "LK1": 2, "NK": 3}  # Lower number = higher priority
    complete_df['priority'] = complete_df['knowledge_type'].map(priority_order)
    
    # Sort by Query_ID, GO_ID, and then by priority to bring the best match to the top
    complete_df.sort_values(by=['Query_ID', 'GO_ID', 'priority'], ascending=True, inplace=True)

    # Now drop duplicates on the protein-term pair, keeping the first (which is the highest priority)
    complete_df.drop_duplicates(subset=['Query_ID', 'GO_ID'], keep='first', inplace=True)
    complete_df.drop(columns=['priority'], inplace=True)

    condition_is_null = complete_df['Taxon_and_Interacting_Taxon'].isnull()
  
    # Condition 2: Check for empty or whitespace-only strings. 
    # The .str accessor safely handles any actual nulls (they won't cause an error).
    # It will return `True` for `''` or `'   '`, and `False` for non-empty strings.
    condition_is_empty_string = (complete_df['Taxon_and_Interacting_Taxon'].str.strip() == '')

    # Combine the conditions with a logical OR (`|`).
    # This creates a final boolean Series that is True if EITHER condition is met.
    final_condition = condition_is_null | condition_is_empty_string

    # Use .loc with the final, robust condition to assign the new value.
    print(f"Found {final_condition.sum()} rows with missing or empty Taxon. Filling with 'Heredited'.")
    complete_df.loc[final_condition, 'Taxon_and_Interacting_Taxon'] = 'Heredited'

    print("Reordering columns to place 'GO_ID' second...")
    cols = complete_df.columns.tolist() # Get current columns as a list
    if 'GO_ID' in cols:
        cols.remove('GO_ID')          # Remove GO_ID from its current position
        cols.insert(1, 'GO_ID')       # Insert GO_ID at index 1 (the second position)
        complete_df = complete_df[cols] # Re-index the DataFrame with the new column order

    # --- 5. Save Final Output ---  
    print(f"Saving final file to {outfile}...")
    complete_df.to_csv(outfile, sep='\t', index=False)
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds.")

    
    
def gp_filtering_6_theshold_check(filename, outfile):  
    print("Running: gp_filtering_6_theshold_check ...")
    
    df = pd.read_csv(filename, sep='\t')
    valid_ids = df[df['knowledge_type'].isin(['NK', 'LK1'])]['Query_ID'].unique()

    # Keep only the groups where ID is in valid_ids
    df_filtered = df[df['Query_ID'].isin(valid_ids)]

    missing_tags = ['philo_missing', 'hard_missing', 'comput_missing', 'author_missing', 'iss_missing', 'misc_missing', 'declassed']

    for threshold in np.arange(0, 1.1, 0.1).tolist():
        if threshold == 0:
            # Write the comment first, then append the DataFrame
            with open(outfile, 'w') as f:
                f.write(f"# Evaluation at threshold: {threshold}\n")
        else:
            with open(outfile, 'a') as f:
                f.write(f"# Evaluation at threshold: {threshold}\n")
        
        missing_percentage = df_filtered.groupby('Query_ID')['knowledge_type'].apply(lambda x: (x.isin(missing_tags)).mean())
        print(missing_percentage)
        # Remove IDs where more than threshold of the entries are 'missing'
        valid_ids = missing_percentage[missing_percentage <= threshold].index  # Keep only IDs with ≤threshold 'missing'
        df_filtered_threshold = df_filtered[df_filtered['Query_ID'].isin(valid_ids)]
        knowledge_type_counts = df_filtered_threshold['knowledge_type'].value_counts()

        # Print the counts
        print("Counts for each knowledge_type:")
        print(knowledge_type_counts)
        knowledge_type_counts.to_csv(outfile, sep='\t', header=True, mode='a')
        
        num_groups = df_filtered_threshold['Query_ID'].nunique()
        with open(outfile, 'a') as f:
                f.write(f"# Number of different Query_ID: {num_groups}\n\n")  # Write comment
        
    print(f"Total unfiltered Query_ID groups: {df['Query_ID'].nunique()}")        
    print(f"Total filtered Query_ID groups: {df_filtered['Query_ID'].nunique()}")
    
    
def gp_filtering_6(filename, ID_list, path_to_fasta, df_threshold, df_cleaned_name, outfile, preferred_threshold = 0.2):  
    
    print("Running: gp_filtering_6 ...")
    print(f"Preferred threshold: {preferred_threshold}")
    df = pd.read_csv(filename, sep='\t')
    print(df.head())
    print(df['EvCo_2024'].head())
    valid_ids = df[df['knowledge_type'].isin(['NK', 'LK1', 'LK2'])]['Query_ID'].unique()

    # Keep only the groups where ID is in valid_ids
    df_filtered = df[df['Query_ID'].isin(valid_ids)]

    missing_tags = ['philo_missing', 'hard_missing', 'comput_missing', 'author_missing', 'iss_missing', 'misc_missing', 'declassed']

    missing_percentage = df_filtered.groupby('Query_ID')['knowledge_type'].apply(lambda x: (x.isin(missing_tags)).mean())
    valid_ids = missing_percentage[missing_percentage <= preferred_threshold].index  # Keep only IDs with ≤threshold% 'missing'
    df_filtered_threshold = df_filtered[df_filtered['Query_ID'].isin(valid_ids)]

    if 'CAFA4_ID' in df_filtered_threshold.columns:
        ID_column = df_filtered_threshold['CAFA4_ID']
    else:
        ID_column = df_filtered_threshold['Query_ID']
    
    df_filtered_threshold.to_csv(df_threshold, sep='\t', index=False)
    df_cleaned = df_filtered_threshold[~df_filtered_threshold['knowledge_type'].isin(missing_tags)]
    df_cleaned.to_csv(df_cleaned_name, sep='\t', index=False)
        
    ID_column_cleaned = ID_column[ID_column != ""].drop_duplicates().reset_index(drop=True)
    ID_column_cleaned.to_csv(ID_list, index=False, header=False)

    # Load the sequence names into a set for fast lookup
    with open(ID_list, "r") as f:
        target_names = set(line.strip() for line in f if line.strip())
    with open(outfile, "w") as out_f:
        for record in SeqIO.parse(path_to_fasta, "fasta"):
            if record.id in target_names:
                SeqIO.write(record, out_f, "fasta")
    
    

def gp_filtering_6_2_definitive_extraction(filename, ID_list, path_to_fasta, of_6_2_seq, of_6_2_nokk):  
    print("Running: gp_filtering_6_2_definitive_extraction ...")
    df = pd.read_csv(filename, sep='\t')

    knowledge_type_counts_before = df['knowledge_type'].value_counts()
    print("Count before filtering:")
    print(knowledge_type_counts_before)

    df = df.groupby("Query_ID").filter(lambda group: not all(group["knowledge_type"] == "KK"))

    knowledge_type_counts_after = df['knowledge_type'].value_counts()
    print("Count after filtering:")
    print(knowledge_type_counts_after)


    if 'CAFA4_ID' in df.columns:
        ID_column = df['CAFA4_ID']
    else:
        ID_column = df['Query_ID']
        
    df.to_csv(of_6_2_nokk, sep='\t', index=False)
            
    ID_column_cleaned = ID_column[ID_column != ""].drop_duplicates().reset_index(drop=True)
    ID_column_cleaned.to_csv(ID_list, index=False, header=False)

    # Load the sequence names into a set for fast lookup
    with open(ID_list, "r") as f:
        target_names = set(line.strip() for line in f if line.strip())

    # Filter and save the matching sequences
    with open(of_6_2_seq, "w") as out_f:
        for record in SeqIO.parse(path_to_fasta, "fasta"):
            if record.id in target_names:
                SeqIO.write(record, out_f, "fasta")
    

    
def clean_fasta_7(input_fasta, output_fasta):
    print("Running: clean_fasta_7 ...")
    
    def extract_organism(header):
        """Extracts organism name from the FASTA header."""
        parts = header.split(" ", 1)  # Split on first space
        return parts[1] if len(parts) > 1 else "Unknown"  # Return organism name

    def remove_duplicate_fasta(input_fasta, output_fasta):
        unique_sequences = {}  # Dictionary to store unique sequences per organism
        total_sequences = 0  # Counter for input sequences

        for record in SeqIO.parse(input_fasta, "fasta"):
            total_sequences += 1  # Count each sequence
            organism = extract_organism(record.description)  # Extract organism name
            key = (organism, str(record.seq))  # Unique key: (organism, sequence)

            if key not in unique_sequences:
                unique_sequences[key] = record

        # Write unique sequences to the output file
        with open(output_fasta, "w") as out_fasta:
            SeqIO.write(unique_sequences.values(), out_fasta, "fasta")

        # Count unique sequences after deduplication
        unique_count = len(unique_sequences)

        # Print results
        print(f"Total sequences before deduplication: {total_sequences}")
        print(f"Total sequences after deduplication: {unique_count}")
        print(f"Duplicates removed: {total_sequences - unique_count}")
    remove_duplicate_fasta(input_fasta, output_fasta)
    


def create_deprecated_dict_and_graph(owl_file, goa_file, owl_data_dir):
    print("Running: create_deprecated_dict and the networkx graph...")
    # Creates a lot of dicts: also the get_gos_ic and the networkx net

    owl = GoOwl(owl_file, goa_file = goa_file, by_ontology = True)
    onto = get_ontology(owl_file).load()

    # Create the graph
    G = nx.Graph()

    # Add nodes (classes)
    for cls in onto.classes():
        G.add_node(cls.name)
    node_list = list(G.nodes)
    
    for node in node_list:
        children = owl.get_children(node, by_ontology=True, valid_edges=True)
        children_go = list(children.keys())
        if children_go:  # Ig the node has children
            for go in children_go:
                simgic = owl.compute_simgic(node, go)
                G.add_edge(node, go, weight=simgic)

    nx.write_edgelist(G, owl_data_dir + "graph_with_simgics_by_ontology.edgelist")   
    
    obs_depr = owl.get_obsolete_deprecated_list()
    depr_json = {key: list(value) for key, value in obs_depr[0].items()}
    obs_json = {key: list(value) for key, value in obs_depr[1].items()}
        
    with open(owl_data_dir + "depr.json", "w") as file:
        json.dump(dict(depr_json), file)  # Convert defaultdict to dict
    with open(owl_data_dir + "obs.json", "w") as file:
        json.dump(dict(obs_json), file)  # Convert defaultdict to dict
        
    dict_ics = owl.get_gos_ic()
    with open(owl_data_dir + "dict_ics.json", 'w') as json_file:
        json.dump(dict_ics, json_file)


   

########   ------------- GROUND TRUTH PRE-PROCESSING -------------   ########

def perfected_aliasing_nx(filename, outfile, cafa):
    print(f'perfected_aliasing_nx running for {cafa}')
    
    df = pd.read_csv(filename, sep='\t')
    
    # 1. Prepare the input data: ensure we have pairs for graph edges
    # We only need the two ID columns.
    df_edges = df[['Query_ID', 'match_IDs']]
    df_edges = df_edges.drop_duplicates()
    df_edges = df_edges.dropna() # Ensure no NaN IDs, which can cause issues

    print(f"Processing {len(df_edges)} unique edges...")

    # 2. Build the graph
    G = nx.Graph() # Undirected graph
    # Add edges from the DataFrame.
    # If Query_ID and match_IDs can be the same, this is fine.
    for _, row in tqdm(df_edges.iterrows(), total=len(df_edges), desc="Building graph"):
        G.add_edge(row['Query_ID'], row['match_IDs'])

    # 3. Find connected components
    # Each component is a set of IDs that are all aliases of each other.
    print("Finding connected components...")
    connected_components = list(nx.connected_components(G))

    # 4. Create the 'aliases' DataFrame (intermediate list format)
    aliases_list_data = []
    for component in tqdm(connected_components, desc="Formatting components"):
        if not component:
            continue
        
        # Sort component for deterministic representative and list order
        sorted_component = sorted(list(component)) 
        
        representative_id = sorted_component[0] # Choose the first (e.g., smallest) as the main Query_ID
        
        # The match_IDs are all other IDs in the component. If the component has only one ID, it's an alias of itself.
        match_ids_list = [item for item in sorted_component if item != representative_id]
        
        if not match_ids_list and len(sorted_component) == 1:
            # If it's a single-node component, the original logic implies it lists itself.
            match_ids_list = [representative_id]
            
        aliases_list_data.append({'Query_ID': representative_id, 'match_IDs': match_ids_list})

    aliases_intermediate_df = pd.DataFrame(aliases_list_data)

    # Ensure 'match_IDs' column can hold lists (especially empty lists)
    # This is often automatic if lists are present, but good for robustness.
    if 'match_IDs' in aliases_intermediate_df.columns:
        aliases_intermediate_df['match_IDs'] = aliases_intermediate_df['match_IDs'].apply(
            lambda x: x if isinstance(x, list) else [] # Ensure it's a list
        )
    else: # Handle case where aliases_intermediate_df might be empty or lack 'match_IDs'
        aliases_intermediate_df['match_IDs'] = pd.Series([[] for _ in range(len(aliases_intermediate_df))], dtype=object)

    aliases_long_df = aliases_intermediate_df.explode('match_IDs')
    print('\nAliases head (intermediate list format):')
    print(aliases_long_df.head(5))
    aliases_long_df = aliases_long_df.sort_values('Query_ID').reset_index(drop=True)
    aliases_long_df.to_csv(outfile, sep='\t', index=False)
    
    

def make_ID_list(fn_gt, list_name_nk, list_name_lk, cafa):
    print(f'make_ID_list running for {cafa}...')
    
    if cafa == "C5":
        column = 'Query_ID'
    elif cafa == "C4":
        column = 'CAFA4_ID'

    df = pd.read_csv(fn_gt, sep ="\t")

    NK_df = df[df["knowledge_type"] == "NK"]
    other_df = df[df["knowledge_type"] != "NK"]

    # Get unique IDs
    unique_ids_NK = NK_df[column].unique()
    unique_ids_other = other_df[column].unique()

    # Save to TSV files
    pd.DataFrame(unique_ids_NK, columns=[column]).to_csv(list_name_nk, sep="\t", index=False)
    pd.DataFrame(unique_ids_other, columns=[column]).to_csv(list_name_lk, sep="\t", index=False)    

    

def separate_knowledge_type(fn_gt, id_list_nk, of_nk, id_list_lk, of_lk, cafa):
    print(f'separate_knowledge_type running for {cafa}...')
    if cafa == "C5":
        id_column = 'Query_ID'
    elif cafa == "C4":
        id_column = 'CAFA4_ID'

    gt_df = pd.read_csv(fn_gt, sep="\t")  # The DataFrame containing all rows
    reference_df_nk = pd.read_csv(id_list_nk, sep="\t")  # The DataFrame with preferred_IDs
    filtered_df_nk = gt_df[gt_df[id_column].isin(reference_df_nk[id_column])]
    filtered_df_nk.to_csv(of_nk, sep="\t", index=False)
    
    # gt_df_lk = pd.read_csv(fn_lk, sep="\t")  # The DataFrame containing all rows
    reference_df_lk = pd.read_csv(id_list_lk, sep="\t")  # The DataFrame with preferred_IDs
    filtered_df_lk = gt_df[gt_df[id_column].isin(reference_df_lk[id_column])]
    filtered_df_lk.to_csv(of_lk, sep="\t", index=False)



def concat_gt(gt_nk, gt_lk, gt_general):
    print(f'concat_gt running...')
    
    df1 = pd.read_csv(gt_nk, sep = "\t")
    df2 = pd.read_csv(gt_lk, sep = "\t")

    # Step 1: Concatenate the two DataFrames
    df_combined = pd.concat([df1, df2], ignore_index=True)

    # Step 2: Sort based on columns
    df_sorted = df_combined.sort_values(by=['Query_ID', 'GO_ID']).reset_index(drop=True)
    df_sorted.to_csv(f"{gt_general}", index=False, sep="\t") 


    
def alias_preprocessing(filename, aliases_nx, outfile):
    print(f'alias_preprocessing running...')
    
    aliases = pd.read_csv(aliases_nx, sep='\t')

    # for type in type_list:
    df1 = pd.read_csv(filename, sep='\t')
    id_map = dict(zip(aliases['match_IDs'], aliases['Query_ID']))
    # Replace IDs in df1 using the map
    df1['Query_ID'] = df1['Query_ID'].map(id_map).fillna(df1['Query_ID']).astype(str)
    
    # Replace '_' for compatibility
    df1["GO_ID"] = df1["GO_ID"].str.replace("_", ":") 
    # The aspect triplet needs to be C/M/P!
    # Triplet CC/MF/BP
    df1["Aspect"] = df1["Aspect"].str.replace("CC", "C") 
    df1["Aspect"] = df1["Aspect"].str.replace("MF", "M") 
    df1["Aspect"] = df1["Aspect"].str.replace("BP", "P") 
    # Triplet C/F/B
    df1["Aspect"] = df1["Aspect"].str.replace("F", "M") 
    df1["Aspect"] = df1["Aspect"].str.replace("B", "P") 
    df1 = df1.sort_values(by=['Query_ID', 'GO_ID']).reset_index(drop=True)
    
    df1.to_csv(outfile, sep='\t', index=False)