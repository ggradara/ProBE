import pandas as pd
import numpy as np
import os
import src_collection
import sys

def ensure_directory_exists(directory_path):
    """Check if a directory exists and create it if it doesn't."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)



def pipeline_C5(uniprot_merged_path, cores, goa_new, goa_old, go_owl, preferred_threshold, 
                fasta_reference, dir_tree, stop_at_filtering = False, skip_to_filtering = False, chunk_size=500000):

    ###   -----------------------------------------------------------------   ###
    # Do the diamond search and then filter the results

    if not skip_to_filtering:

        chunked_C5_dir = os.path.join(dir_tree['pc5_dir_path'], "chunked_results_cafa5") 
        ensure_directory_exists(chunked_C5_dir)
        src_collection.diamond_management_C5(dir_tree, uniprot_merged_path, cores)


        ###   -----------------------------------------------------------------   ###
        # Retrieve all the aliases aligned using the diamond search
        benchmark_c5_dir = os.path.join(dir_tree['pc5_dir_path'], "benchmarker_cafa5")
        ensure_directory_exists(benchmark_c5_dir)
        benchmark_c5_dir = benchmark_c5_dir + "/"
        fn_2_aliases = benchmark_c5_dir + "filtered_chunked_cafa5_1.tsv"
        of_2_aliases = benchmark_c5_dir + "chunked_cafa5_aliases_2.tsv"

        src_collection.find_aliases_from_filtered_diamond_2_C5(fn_2_aliases, of_2_aliases)


        ###   -----------------------------------------------------------------   ###
        # Assign go terms to the gene products and perform preliminary filtering, for both goas
        # Assign recent data (by convention we call it 2025)
        fn_3_retriever = fn_2_aliases  # the starting data is the same for both
        of_3_retriever_2025 = benchmark_c5_dir + "chunked_go_2025_CAFA5_3.gz"

        src_collection.go_retriever_diamond_chunks_3_C5_fast(goa_new, fn_3_retriever, of_3_retriever_2025, chunk_size)

        # Assign older data (by convention we call it 2023)
        # the starting data is the same for both
        of_3_retriever_2023 = benchmark_c5_dir + "chunked_go_2023_CAFA5_3.gz"

        src_collection.go_retriever_diamond_chunks_3_C5_fast(goa_old, fn_3_retriever, of_3_retriever_2023, chunk_size)
            

        ###   -----------------------------------------------------------------   ###
        # Remove the duplicates and the roots in the dataframe, for both the dataframes
        # the file with duplicates is saved as gzip to save memory
        fn_3_1_nodup_2025 = of_3_retriever_2025
        of_3_1_nodup_2025 = benchmark_c5_dir + "chunked_go_2025_CAFA5_3_1_nodup.tsv"

        src_collection.remove_duplicates_and_roots_3_1(fn_3_1_nodup_2025, of_3_1_nodup_2025)

        fn_3_1_nodup_2023 = of_3_retriever_2023
        of_3_1_nodup_2023 = benchmark_c5_dir + "chunked_go_2023_CAFA5_3_1_nodup.tsv"

        src_collection.remove_duplicates_and_roots_3_1(fn_3_1_nodup_2023, of_3_1_nodup_2023)


        ###   -----------------------------------------------------------------   ###
        # Remove the annotations that are not experimental and keep only the most recent ones
        fn_3_5_2025 = of_3_1_nodup_2025
        of_3_5_2025 = benchmark_c5_dir + "chunked_go_2025_CAFA5_3_5.tsv"

        src_collection.go_quality_selection_chunks_v2_3_5(fn_3_5_2025, of_3_5_2025, chunk_size)

        fn_3_5_2023 = of_3_1_nodup_2023
        of_3_5_2023 = benchmark_c5_dir + "chunked_go_2023_CAFA5_3_5.tsv"

        src_collection.go_quality_selection_chunks_v2_3_5(fn_3_5_2023, of_3_5_2023, chunk_size)


        ###   -----------------------------------------------------------------   ###
        # Rerun of the previous script without the chunking to remove remaining redundancy
        fn_3_6_2025 = of_3_5_2025
        of_3_6_2025 = benchmark_c5_dir + "chunked_go_2025_CAFA5_3_6.tsv"

        src_collection.go_quality_selection_round2_3_6(fn_3_6_2025, of_3_6_2025)

        fn_3_6_2023 = of_3_5_2023
        of_3_6_2023 = benchmark_c5_dir + "chunked_go_2023_CAFA5_3_6.tsv"

        src_collection.go_quality_selection_round2_3_6(fn_3_6_2023, of_3_6_2023)


        ###   -----------------------------------------------------------------   ###
        # Substitution and removal of obslete goas from the old annotation for compatibility purposes

        owl_data_dir = dir_tree['owl_dir_path']
        ensure_directory_exists(owl_data_dir)
        owl_data_dir = owl_data_dir + "/"

        src_collection.create_deprecated_dict_and_graph(go_owl, goa_new, owl_data_dir)

        fn_3_61_2025 = of_3_6_2025  # To be sure we check for obsolete gos also in the new dataframe
        of_3_61_2025 = benchmark_c5_dir + "chunked_go_2025_CAFA5_3_61_noobs.tsv"

        src_collection.substitute_obsolete_go_3_61(fn_3_61_2025, owl_data_dir, of_3_61_2025)

        fn_3_61_2023 = of_3_6_2023
        of_3_61_2023 = benchmark_c5_dir + "chunked_go_2023_CAFA5_3_61_noobs.tsv"

        src_collection.substitute_obsolete_go_3_61(fn_3_61_2023, owl_data_dir, of_3_61_2023)


        ###   -----------------------------------------------------------------   ###
        # Trims the df from low simgic propagations keeping only the most informative annotations
        fn_3_69_2023 = of_3_61_2023
        of_3_69_2023 = benchmark_c5_dir + "chunked_go_2023_CAFA5_3_69_noIEA.tsv"

        src_collection.remove_IEA_3_69(fn_3_69_2023, of_3_69_2023)

        fn_3_69_2025 = of_3_61_2025
        of_3_69_2025 = benchmark_c5_dir + "chunked_go_2025_CAFA5_3_69_noIEA.tsv"

        src_collection.remove_IEA_3_69(fn_3_69_2025, of_3_69_2025)


        ###   -----------------------------------------------------------------   ###
        # Merges the two dataframes allowing for further comparisons down the pipeline
        # Tags the annotations comparing the two different dataframes via chunking
        # in particular this script looks for annotations that were not present in the past
        # the tags are: GO_NF for annotations taht were not available in the older dataframe and
        # GP_NF for gene products that were missing altogether

        df_new_4_2025 = of_3_69_2025
        df_old_4_2023 = of_3_69_2023
        of_4_evco_tagged = benchmark_c5_dir + "double_evco_CAFA5_4.tsv"

        src_collection.double_evco_tagger_chunks_4_C5(df_new_4_2025, df_old_4_2023, of_4_evco_tagged, chunk_size)


        ###   -----------------------------------------------------------------   ###
        # Properly tags each annotation after comparing it to the previous dataframe
        # NK for new gene products, LK1 for new annotation for known gene products
        # LK2 for experimental confirmetion of previously IEA annotations
        # KK previously known annotations

        benchmark_data_5_c5_dir = os.path.join(dir_tree['results_dir_path'], "knowledge_reports_cafa5")
        ensure_directory_exists(benchmark_data_5_c5_dir)
        benchmark_data_5_c5_dir = benchmark_data_5_c5_dir + "/"
        fig_hard_missing_5 = benchmark_data_5_c5_dir + "hard_missing_counts_C5.png"
        fig_overall_missing_5 = benchmark_data_5_c5_dir + "overall_missing_counts_C5.png"
        count_hard_missing_5 = benchmark_data_5_c5_dir + "hard_missing_counts_C5.tsv"
        count_overall_missing_5 = benchmark_data_5_c5_dir + "overall_missing_counts_C5.tsv"

        fn_5_selected = of_4_evco_tagged
        of_5_selected = benchmark_c5_dir + "df_with_knowledge_type_C5_5.tsv"

        src_collection.benchmark_selection_5_subcat_C5(fn_5_selected, of_5_selected, fig_hard_missing_5,
                                                        fig_overall_missing_5, count_hard_missing_5, count_overall_missing_5)


        ###   -----------------------------------------------------------------   ###
        # Compute the number of annotation available after culling the gp with a specific threshold of missing data

        fn_6_threshold_check = of_5_selected
        of_6_threshold_eval = benchmark_data_5_c5_dir + "threshold_eval_C5_6.txt"

        src_collection.gp_filtering_6_theshold_check(fn_6_threshold_check, of_6_threshold_eval)
        
        if stop_at_filtering:
            print("[STOP] threshold_eval_C5 computed, pipeline stopped as requested by 'stop_at_filtering'")
            sys.exit(1)        
    else:  
        # if the rest of the script is skipped initialize the required names
        benchmark_c5_dir = os.path.join(dir_tree['pc5_dir_path'], "benchmarker_cafa5")
        ensure_directory_exists(benchmark_c5_dir)
        benchmark_c5_dir = benchmark_c5_dir + "/"
        fn_2_aliases = benchmark_c5_dir + "filtered_chunked_cafa5_1.tsv"
        of_5_selected = benchmark_c5_dir + "df_with_knowledge_type_C5_5.tsv"

    ###   -----------------------------------------------------------------   ###
    # Apply the previously selected filtering threshold to clean the dataframe and retrieve the name of 
    # the gene products that will take part in the final benchmark
    fn_6_filtering = of_5_selected
    preferred_threshold_char = str(preferred_threshold).replace(".", "_") # make it influence the names before everything
    ID_list_6 = benchmark_c5_dir + f"ID_list_C5_6_{preferred_threshold_char}.tsv"
    path_to_fasta = dir_tree['dmnd_data_path'] + "/testsuperset_cafa5.fasta"
    df_threshold = benchmark_c5_dir + f"data_with_knowledge_type_C5_6_{preferred_threshold_char}.tsv"
    df_cleaned_name = benchmark_c5_dir + f"data_with_knowledge_type_cleaned_C5_6_{preferred_threshold_char}.tsv"
    of_6_filtering = benchmark_c5_dir + f"selected_sequences_C5_threshold_{preferred_threshold_char}.fasta" 
    src_collection.gp_filtering_6(fn_6_filtering, ID_list_6, path_to_fasta, df_threshold, df_cleaned_name, 
                                of_6_filtering, preferred_threshold)


    ###   -----------------------------------------------------------------   ###
    # Remove redundant information based on the date of discovery and priority system based on the knowledge type
    # NK annotation take full precedence, then LK then KK
    fn_6_1_nodup = df_cleaned_name
    of_6_1_nodup = benchmark_c5_dir + "data_with_knowledge_type_nodup_C5_6_1.tsv"
    src_collection.propagated_existing_annotations_6_1_fast(go_owl, fn_6_1_nodup, of_6_1_nodup)

    ###   -----------------------------------------------------------------   ###
    # Create the definitive fasta itself from the list of gene products
    # This fasta will contain all the gene products that will be considered in our benchmark
    prep_persist_dir = dir_tree['prep_data_dir_path']
    ensure_directory_exists(prep_persist_dir)
    prep_persist_dir = prep_persist_dir + "/"

    fn_6_2_nokk = of_6_1_nodup
    ID_list_6_2 = prep_persist_dir + f"ID_list_C5_threshold_{preferred_threshold_char}_nokk.tsv"
    of_6_2_nokk = benchmark_c5_dir + f"overall_data_C5_threshold_{preferred_threshold_char}_nokk.tsv"
    of_6_2_seq = prep_persist_dir + f"query_seq_C5_thr_{preferred_threshold_char}_nokk.fasta"

    src_collection.gp_filtering_6_2_definitive_extraction(fn_6_2_nokk, ID_list_6_2, path_to_fasta, of_6_2_seq, of_6_2_nokk)


    ###   -----------------------------------------------------------------   ###
    # Removes potential duplicates from the fasta file itself
    input_fasta_7 = of_6_2_seq
    output_fasta_7 = prep_persist_dir + f"fasta_seqs_for_pred_C5.fasta"
    src_collection.clean_fasta_7(input_fasta_7, output_fasta_7)


    ###  ----------------- GROUND TRUTH PREPROCESSING ----------------------  ###
    cafa = "C5"  # Useful for some preprocessing functions

    # Create the alias table through a networkx graph
    preprocessing_dir = dir_tree['prep_dir_path']
    ensure_directory_exists(preprocessing_dir)
    preprocessing_dir = preprocessing_dir + "/"
    fn_aliasing_nx = fn_2_aliases
    of_aliasing_nx = prep_persist_dir + f"aliases_C5_nx.tsv"  # This file must persist
    src_collection.perfected_aliasing_nx(fn_aliasing_nx, of_aliasing_nx, cafa)

    # Create the ID list used to separate the gt
    fn_gt = of_6_2_nokk
    list_name_nk = preprocessing_dir + "ID_list_NK_C5.tsv"
    list_name_lk = preprocessing_dir + "ID_list_KK_LK_C5.tsv"
    src_collection.make_ID_list(fn_gt, list_name_nk, list_name_lk, cafa)

    # Separate the gt into NK and LK_KK
    fn_gt_mixed = of_6_2_nokk
    id_list_nk = list_name_nk
    id_list_lk = list_name_lk
    of_gt_nk_separated = preprocessing_dir + f"prep_ground_truth_NK_C5.tsv"
    of_gt_lk_separated = preprocessing_dir + f"prep_ground_truth_LK_KK_C5.tsv"
    src_collection.separate_knowledge_type(fn_gt_mixed, id_list_nk, of_gt_nk_separated, 
                                        id_list_lk, of_gt_lk_separated, cafa)

    # Create the general gt from NK and LK
    fn_gt_nk_pre_alias = of_gt_nk_separated
    fn_gt_lk_pre_alias = of_gt_lk_separated
    of_gt_general = preprocessing_dir + f"prep_ground_truth_general_C5.tsv"
    src_collection.concat_gt(fn_gt_nk_pre_alias, fn_gt_lk_pre_alias, of_gt_general)

    # Alias preprocessing and creation of definitive gt
    ground_truth_dir = dir_tree['gt_dir_path']
    ensure_directory_exists(ground_truth_dir)
    ground_truth_dir = ground_truth_dir + "/"

    fn_gt_gen_pre_alias = of_gt_general
    aliases_nx = of_aliasing_nx
    of_gt_nk_def = ground_truth_dir + f'ground_truth_NK_C5.tsv'
    of_gt_lk_def = ground_truth_dir + f'ground_truth_LK_KK_C5.tsv'
    of_gt_gen_def = ground_truth_dir + f'ground_truth_general_C5.tsv'
    src_collection.alias_preprocessing(fn_gt_nk_pre_alias, aliases_nx, of_gt_nk_def)
    src_collection.alias_preprocessing(fn_gt_lk_pre_alias, aliases_nx, of_gt_lk_def)
    src_collection.alias_preprocessing(fn_gt_gen_pre_alias, aliases_nx, of_gt_gen_def)
