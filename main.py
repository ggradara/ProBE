import argparse
import yaml
import os
import sys
import time
import pipeline_C5_fun
import pipeline_C4_fun
import bench_by_aspect_custom_fun
import bench_general_custom_fun
import preprocess_predictions_fun
import src_collection
import shutil
import uuid

# ===================================================================
#                      Core Pipeline Functions
# ===================================================================
# These are your actual analysis functions. They take the final, merged config dictionary as their only argument.

def run_make_db(config: dict, dir_tree: dict):
    """
    Create the database necessary to run the diamond search.
    """
    print("\n[INFO] --- Starting Make DB ---")
    
    # --- Configuration Access ---
    shared_config = config.get('shared', {})
    make_db_config = config.get('make_db', {})
    db_name = make_db_config.get('db_name', "uniprot_merged_test.dmnd")
    db_source_rel_path = make_db_config.get('db_source', "uniprot_merged.fasta")
    
    if not db_source_rel_path:
        print("[ERROR] 'uniprot' not defined in pipeline_C5 section of config.", file=sys.stderr)
        sys.exit(1)
        
    base_path = shared_config.get('base_path', '/data')
    db_source_abs_path = os.path.join(base_path, db_source_rel_path)
    
    if not os.path.exists(db_source_abs_path):
        print(f"[ERROR] File not found: {db_source_abs_path}", file=sys.stderr)
        sys.exit(1)
        
    dmnd_db_dir_path = dir_tree['dmnd_db_dir_path']
    print(f'[INFO] --- {db_name} will be saved in {dmnd_db_dir_path} ---')
    db_path = os.path.join(dmnd_db_dir_path, db_name)

    src_collection.make_db(db_source_abs_path, db_path)

    print("[SUCCESS] --- Diamond DB created  ---")
   
    
def run_pc5(config: dict, dir_tree: dict):
    """
    Performs the analysis for the pipeline C5.
    """
    print("\n[INFO] --- Starting Pipeline C5 ---")
    start_time = time.time()
    
    # --- Configuration Access ---
    shared_config = config.get('shared', {})
    chunk_size = shared_config.get('chunk_size', 1000000)

    pc5_config = config.get('pipeline_C5', {})
    preferred_threshold = pc5_config.get('preferred_threshold', 0.2)
    cores = pc5_config.get('cores', 32)
    stop_at_filtering = pc5_config.get('stop_at_filtering', False)
    skip_to_filtering = pc5_config.get('skip_to_filtering', False)
    # Future-proofing to allow for independent or new fastas to use as the base of the benchmark, don't hold your breath
    fasta_reference = pc5_config.get('fasta_reference', "internal")
    
    
    if stop_at_filtering and skip_to_filtering:
        print("[ERROR] Conflicting instructions, stop_at_filtering and skip_to_filtering can't be both true", file=sys.stderr)
        sys.exit(1)

    # --- Input Validation ---
    old_goa_C5_rel_path = pc5_config.get('old_goa_C5')
    if not old_goa_C5_rel_path:
        print("[ERROR] 'old_goa_C5' not defined in pipeline_C5 section of config.", file=sys.stderr)
        sys.exit(1)
        
    new_goa_rel_path = shared_config.get('new_goa')
    if not new_goa_rel_path:
        print("[ERROR] 'new_goa' not defined in pipeline_C5 section of config.", file=sys.stderr)
        sys.exit(1)
        
    go_owl_rel_path = shared_config.get('go_owl')
    if not go_owl_rel_path:
        print("[ERROR] 'go_owl' not defined in pipeline_C5 section of config.", file=sys.stderr)
        sys.exit(1)
        
    make_db_config = config.get('make_db', {})
    dmnd_db_rel_path = make_db_config.get('db_name')
    if not dmnd_db_rel_path:
        print("[ERROR] 'db_name' not defined in make_db section of config.", file=sys.stderr)
        sys.exit(1)
        
    # Construct full path inside the container
    base_path = shared_config.get('base_path', '/data')
    old_goa_C5_abs_path = os.path.join(base_path, old_goa_C5_rel_path)
    dmnd_db_rel_path = 'dmnd_dbs/' + dmnd_db_rel_path
    dmnd_db_abs_path = os.path.join(base_path, dmnd_db_rel_path)
    new_goa_abs_path = os.path.join(base_path, new_goa_rel_path)
    go_owl_abs_path = os.path.join(base_path, go_owl_rel_path)


    if not os.path.exists(old_goa_C5_abs_path):
        print(f"[ERROR] File not found: {old_goa_C5_abs_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(dmnd_db_abs_path):
        print(f"[ERROR] File not found: {dmnd_db_abs_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(new_goa_abs_path):
        print(f"[ERROR] File not found: {new_goa_abs_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(go_owl_abs_path):
        print(f"[ERROR] File not found: {go_owl_abs_path}", file=sys.stderr)
        sys.exit(1)
        
    # Evaluate the provided fasta reference
    if fasta_reference == "internal":
        fasta_reference = "diamond_data/testsuperset_cafa5.fasta"
    else:
        if not fasta_reference:
            print("[ERROR] 'fasta_reference' not defined in pipeline_C5 section of config.", file=sys.stderr)
            sys.exit(1)
        fasta_reference = os.path.join(base_path, fasta_reference)
        if not os.path.exists(fasta_reference):
            print(f"[ERROR] File not found: {fasta_reference}", file=sys.stderr)
            sys.exit(1)

    pipeline_C5_fun.pipeline_C5(dmnd_db_abs_path, cores, new_goa_abs_path, old_goa_C5_abs_path, go_owl_abs_path, 
                                preferred_threshold, fasta_reference, dir_tree, stop_at_filtering, skip_to_filtering, chunk_size)
    
    print("[SUCCESS] --- Pipeline C5 finished ---")
    run_time = time.time()
    elapsed_time = (run_time - start_time)/60
    print(f"Elapsed time: {elapsed_time:.3f} minutes to process Pipeline C5")
    
    

def run_pc4(config: dict, dir_tree: dict):
    """
    Performs the analysis for the pipeline C4.
    """
    print("\n[INFO] --- Starting Pipeline C4 ---")
    start_time = time.time()
    
    # --- Configuration Access ---
    shared_config = config.get('shared', {})
    chunk_size = shared_config.get('chunk_size', 1000000)

    pc4_config = config.get('pipeline_C4', {})
    preferred_threshold = pc4_config.get('preferred_threshold', 0.2)
    cores = pc4_config.get('cores', 32)
    stop_at_filtering = pc4_config.get('stop_at_filtering', False)
    skip_to_filtering = pc4_config.get('skip_to_filtering', False)
    # Future-proofing to allow for independent or new fastas to use as the base of the benchmark, don't hold your breath
    fasta_reference = pc4_config.get('fasta_reference', "internal")
    
    if stop_at_filtering and skip_to_filtering:
        print("[ERROR] Conflicting instructions, stop_at_filtering and skip_to_filtering can't be both true", file=sys.stderr)
        sys.exit(1)

    # --- Input Validation ---
    old_goa_C4_rel_path = pc4_config.get('old_goa_C4')
    if not old_goa_C4_rel_path:
        print("[ERROR] 'old_goa_C4' not defined in pipeline_C4 section of config.", file=sys.stderr)
        sys.exit(1)
        
    new_goa_rel_path = shared_config.get('new_goa')
    if not new_goa_rel_path:
        print("[ERROR] 'new_goa' not defined in pipeline_C4 section of config.", file=sys.stderr)
        sys.exit(1)
        
    go_owl_rel_path = shared_config.get('go_owl')
    if not go_owl_rel_path:
        print("[ERROR] 'go_owl' not defined in pipeline_C4 section of config.", file=sys.stderr)
        sys.exit(1)
        
    make_db_config = config.get('make_db', {})
    dmnd_db_rel_path = make_db_config.get('db_name')
    if not dmnd_db_rel_path:
        print("[ERROR] 'db_name' not defined in make_db section of config.", file=sys.stderr)
        sys.exit(1)
        
    # Construct full path inside the container
    base_path = shared_config.get('base_path', '/data')
    old_goa_C4_abs_path = os.path.join(base_path, old_goa_C4_rel_path)
    new_goa_abs_path = os.path.join(base_path, new_goa_rel_path)
    go_owl_abs_path = os.path.join(base_path, go_owl_rel_path)
    dmnd_db_rel_path = 'dmnd_dbs/' + dmnd_db_rel_path
    dmnd_db_abs_path = os.path.join(base_path, dmnd_db_rel_path)
    

    if not os.path.exists(old_goa_C4_abs_path):
        print(f"[ERROR] File not found: {old_goa_C4_abs_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(dmnd_db_abs_path):
        print(f"[ERROR] File not found: {dmnd_db_abs_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(new_goa_abs_path):
        print(f"[ERROR] File not found: {new_goa_abs_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(go_owl_abs_path):
        print(f"[ERROR] File not found: {go_owl_abs_path}", file=sys.stderr)
        sys.exit(1)
        
    # Evaluate the provided fasta reference
    if fasta_reference == "internal":
        fasta_reference = "diamond_data/superset_cafa4.fasta"
    else:
        if not fasta_reference:
            print("[ERROR] 'fasta_reference' not defined in pipeline_C4 section of config.", file=sys.stderr)
            sys.exit(1)
        fasta_reference = os.path.join(base_path, fasta_reference)
        if not os.path.exists(fasta_reference):
            print(f"[ERROR] File not found: {fasta_reference}", file=sys.stderr)
            sys.exit(1)

    pipeline_C4_fun.pipeline_C4(dmnd_db_abs_path, cores, new_goa_abs_path, old_goa_C4_abs_path, go_owl_abs_path, 
                                preferred_threshold, fasta_reference, dir_tree, stop_at_filtering, skip_to_filtering, chunk_size)
    
    print("[SUCCESS] --- Pipeline C4 finished ---")
    run_time = time.time()
    elapsed_time = (run_time - start_time)/60  
    print(f"Elapsed time: {elapsed_time:.3f} minutes to process Pipeline C4")
    

    
def run_bench_by_aspect(config: dict, dir_tree: dict):
    """
    Performs the benchmarking dividing by aspect.
    """
    print("\n[INFO] --- Starting Benchmarking by aspect ---")
    start_time = time.time()
    
    # --- Configuration Access ---   
    benchmarking_config = config.get('benchmarking', {})
    model_name = benchmarking_config.get('model_name', 'custom_model')
    stepsize = benchmarking_config.get('stepsize', 0.01)
    
    cafa = benchmarking_config.get('cafa', 'both')
    
    # --- Input Validation ---
    model_path_rel_path_C5 = benchmarking_config.get('model_path_C5')
    if not model_path_rel_path_C5:
        print("[WARNING] 'model_path_C5' not defined in benchmarking section of config.")

    model_path_rel_path_C4 = benchmarking_config.get('model_path_C4')
    if not model_path_rel_path_C4:
        print("[WARNING] 'model_path_C4' not defined in benchmarking section of config.")

        
    model_path_abs_path_C5 = os.path.join(dir_tree['prep_data_dir_path'], model_path_rel_path_C5)

    # If the data is required send the error
    if cafa == 'C5' or cafa == 'both' or cafa == 'c5':
        if not os.path.exists(model_path_abs_path_C5):
            print(f"[ERROR] File not found: {model_path_abs_path_C5}", file=sys.stderr)
            sys.exit(1)

    model_path_abs_path_C4 = os.path.join(dir_tree['prep_data_dir_path'], model_path_rel_path_C4)
    
    if cafa == 'C4' or cafa == 'both' or cafa == 'c4':
        if not os.path.exists(model_path_abs_path_C4):
            print(f"[ERROR] File not found: {model_path_abs_path_C4}", file=sys.stderr)
            sys.exit(1)
        
    bench_by_aspect_custom_fun.bench_by_aspect(model_path_abs_path_C5, model_path_abs_path_C4, model_name, stepsize, cafa, dir_tree)
    
    print("[SUCCESS] --- Benchmarking by aspect finished ---")
    run_time = time.time()
    elapsed_time = (run_time - start_time)/60  
    print(f"Elapsed time: {elapsed_time:.3f} minutes to process the aspect sensitive benchmark")
    
    
    
    
def run_bench_general(config: dict, dir_tree: dict):
    """
    Performs the analysis for the general benchmarking.
    """
    print("\n[INFO] --- Starting General benchmarking ---")
    start_time = time.time()
    
    # --- Configuration Access ---
    benchmarking_config = config.get('benchmarking', {})
    model_name = benchmarking_config.get('model_name', 'custom_model')
    stepsize = benchmarking_config.get('stepsize', 0.01)
    cafa = benchmarking_config.get('cafa', 'both')

    # --- Input Validation ---
    model_path_rel_path_C5 = benchmarking_config.get('model_path_C5')
    if not model_path_rel_path_C5:
        print("[WARNING] 'model_path_C5' not defined in benchmarking section of config.")

    model_path_rel_path_C4 = benchmarking_config.get('model_path_C4')
    if not model_path_rel_path_C4:
        print("[WARNING] 'model_path_C4' not defined in benchmarking section of config.")

        
    model_path_abs_path_C5 = os.path.join(dir_tree['prep_data_dir_path'], model_path_rel_path_C5)

    # If the data is required send the error
    if cafa == 'C5' or cafa == 'both' or cafa == 'c5':
        if not os.path.exists(model_path_abs_path_C5):
            print(f"[ERROR] File not found: {model_path_abs_path_C5}", file=sys.stderr)
            sys.exit(1)

    model_path_abs_path_C4 = os.path.join(dir_tree['prep_data_dir_path'], model_path_rel_path_C4)
    
    if cafa == 'C4' or cafa == 'both' or cafa == 'c4':
        if not os.path.exists(model_path_abs_path_C4):
            print(f"[ERROR] File not found: {model_path_abs_path_C4}", file=sys.stderr)
            sys.exit(1)
        
    bench_general_custom_fun.bench_general(model_path_abs_path_C5, model_path_abs_path_C4, model_name, stepsize, cafa, dir_tree)
    
    print("[SUCCESS] --- General benchmarking finished ---")
    run_time = time.time()
    elapsed_time = (run_time - start_time)/60  
    print(f"Elapsed time: {elapsed_time:.3f} minutes to process the general benchmark")

    
    
def run_preprocess(config: dict, dir_tree: dict):
    """
    Performs the preprocess analysis.
    """
    print("\n[INFO] --- Starting preprocessing ---")
    start_time = time.time()
    
    # --- Configuration Access ---
    shared_config = config.get('shared', {})
    preprocessing_config = config.get('preprocessing', {})
    model_name = preprocessing_config.get('model_name', 'custom_model')
    propagate = preprocessing_config.get('propagate', True)
    
    # --- Input Validation ---
    model_path_rel_path = preprocessing_config.get('model_path')
    if not model_path_rel_path:
        print("[ERROR] 'model_path' not defined in preprocessing section of config.", file=sys.stderr)
        sys.exit(1)
    
    owl_file_rel_path = shared_config.get('go_owl')
    if not owl_file_rel_path:
        print("[ERROR] 'go_owl' not defined in shared section of config.", file=sys.stderr)
        sys.exit(1)
        
    # Construct full path inside the container
    base_path = shared_config.get('base_path', '/data')
    model_path_abs_path = os.path.join(base_path, model_path_rel_path)

    if not os.path.exists(model_path_abs_path):
        print(f"[ERROR] File not found: {model_path_abs_path}", file=sys.stderr)
        sys.exit(1)
        
    owl_file_abs_path = os.path.join(base_path, owl_file_rel_path)

    if not os.path.exists(owl_file_abs_path):
        print(f"[ERROR] File not found: {owl_file_abs_path}", file=sys.stderr)
        sys.exit(1)
    
    preprocess_predictions_fun.preprocess(model_path_abs_path, model_name, propagate, owl_file_abs_path, 
                                          dir_tree)
    
    print(f"[SUCCESS] --- Preprocessing for {model_name} predictions finished ---")
    run_time = time.time()
    elapsed_time = (run_time - start_time)/60  
    print(f"Elapsed time: {elapsed_time:.3f} minutes to process Preprocessing Pipeline")



# ===================================================================
#                          Directory Setup
# ===================================================================
def ensure_directory_exists(directory_path):
    """Check if a directory exists and create it if it doesn't."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def directory_setup(config: dict):
    shared_config = config.get('shared', {})
    base_path = shared_config.get('base_path', '/data')
    results_dir = shared_config.get('run_path', 'results')
    dmnd_data_dir = shared_config.get('dmnd_data_path', 'diamond_data')
    run_tag = shared_config.get('run_tag', False)
    
    # Create the temp directory and the result directory
    tmp_dir = 'tmp'
    if run_tag:
        run_id = run_tag  # Use the provided tag for debugging
        print(f"[INFO] Using provided run_tag: {run_id}")
    else:
        run_id = str(uuid.uuid4())[:8]  # Create a unique identifier for the specific run
    tmp_dir_path = os.path.join(base_path, f"tmp/{tmp_dir}_{run_id}")
    ensure_directory_exists(tmp_dir_path)
    generic_res_dir_path = os.path.join(base_path, results_dir)
    results_dir_path = os.path.join(generic_res_dir_path, f"results_{run_id}")
    if os.path.exists(results_dir_path):
        print('[WARNING] --- Same "run_path" config option as other previous pipeline runs ---')
    ensure_directory_exists(results_dir_path)
    
    # Populate the temp directory
    pc5_dir_path = os.path.join(tmp_dir_path, 'pipeline_cafa5')
    ensure_directory_exists(pc5_dir_path)
    pc4_dir_path = os.path.join(tmp_dir_path, 'pipeline_cafa4')
    ensure_directory_exists(pc4_dir_path)
    prep_dir_path = os.path.join(tmp_dir_path, 'preprocessing_data')
    ensure_directory_exists(prep_dir_path)

    # Create the results directories
    dmnd_db_dir_path = os.path.join(base_path, 'dmnd_dbs')
    ensure_directory_exists(dmnd_db_dir_path)    
    # Even if it's not part of the results, I need it to be persistent and tied to the individual run
    owl_dir_path = os.path.join(base_path, 'owl_data')
    ensure_directory_exists(owl_dir_path)   
    gt_dir_path = os.path.join(results_dir_path, 'ground_truth')
    ensure_directory_exists(gt_dir_path)
    prep_data_dir_path = os.path.join(results_dir_path, 'preprocessing_data')
    ensure_directory_exists(prep_data_dir_path)
    prep_preds_dir_path = os.path.join(results_dir_path, 'preprocessed_preds')
    ensure_directory_exists(prep_preds_dir_path)
    btp_dir_path = os.path.join(results_dir_path, 'benchmark_results')
    ensure_directory_exists(btp_dir_path)
    
    # Creation of the dir tree, useful to tell to the functions where to put what
    dir_tree = {}
    # Populate with tmp dirs
    dir_tree['tmp_dir_path'] = tmp_dir_path
    dir_tree['dmnd_data_path'] = dmnd_data_dir
    dir_tree['pc5_dir_path'] = pc5_dir_path
    dir_tree['pc4_dir_path'] = pc4_dir_path
    dir_tree['prep_dir_path'] = prep_dir_path
    # Populate with results dirs
    dir_tree['results_dir_path'] = results_dir_path
    dir_tree['owl_dir_path'] = owl_dir_path
    dir_tree['dmnd_db_dir_path'] = dmnd_db_dir_path
    dir_tree['gt_dir_path'] = gt_dir_path
    dir_tree['prep_data_dir_path'] = prep_data_dir_path   # Necessary data required for the preprocessing
    dir_tree['prep_preds_dir_path'] = prep_preds_dir_path  # Preprocessed tool predictions
    dir_tree['btp_dir_path'] = btp_dir_path

    return dir_tree




# ===================================================================
#                  Main Command-Line Interface Logic
# ===================================================================

def main():
    
    # --- Define ALL possible arguments in argparse ---
    
    # --- Top-level parser ---
    parser = argparse.ArgumentParser(
        description="A multi-modal benchmarking pipeline. Choose a functionality to run (e.g., run_pc4, run_pc5).",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    # This creates the sub-command system. The chosen command will be stored in `args.command`.
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available functionalities")
    
    # --- Parent parser for shared arguments ---
    # Arguments defined here will be available for ALL sub-commands.
    # This avoids repeating `--config` and override arguments everywhere.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the config.yaml file (outside the container).'
    )
    
    
    # These arguments are OPTIONAL and will override the config file
    parser.add_argument('--cores', type=int, default=None, help='Override the number of cores from the config file.')
    parser.add_argument('--stepsize', type=float, default=None, help='Override the benchmark stepsize from the config file.')
    parser.add_argument('--cafa', type=str, default=None, help='Choose witch cafa to use to run the benchmark, "C4", "C5" or "both" (Overrides config file).')
    parser.add_argument('--preferred_threshold', type=float, default=None, help='Override the number of cores from the config file.')
    parser.add_argument('--stop_at_filtering', type=bool, default=None, help='Stop the pipeline before the filtering step. (Allows to evaluate the data before setting a threshold)')
    parser.add_argument('--skip_to_filtering', type=bool, default=None, help='Start the pipeline at the filtering step, skipping everything else.')
    parser.add_argument('--propagate', type=bool, default=None, help='Propagate the GO annotation of a predictive tool.')

    # ============================ PIPELINE FUNCTIONALITIES ============================
    # MAKE_DB, DIAMOND_SEARCH, PC4, PC5, PREPROCESS, BENCHMARK
    
    # --- Parser for make_db sub-command ---
    parser_make_db = subparsers.add_parser(
        'make_db', 
        parents=[parent_parser], # Inherit shared arguments
        help='Make the diamond DB from the Uniprot data',
        description='Uses the uniprot.fasta to create a db accessible for the diamond search.'
    )
    parser_make_db.set_defaults(func=run_make_db) # Link sub-command to its function

    # --- Parser for pc4 sub-command ---
    parser_pc4 = subparsers.add_parser(
        'pc4', 
        parents=[parent_parser], # Inherit shared arguments
        help='Run the pipeline for the creation of the ground truth for the CAFA4 data.',
        description='Creates and runs the preprocessing of the ground truth for the benchmark from the CAFA4 data.'
    )
    parser_pc4.set_defaults(func=run_pc4) # Link sub-command to its function
    
    # --- Parser for pc5 sub-command ---
    parser_pc5 = subparsers.add_parser(
        'pc5', 
        parents=[parent_parser], # Inherit shared arguments
        help='Run the pipeline for the creation of the ground truth for the CAFA5 data.',
        description='Creates and runs the preprocessing of the ground truth for the benchmark from the CAFA5 data.'
    )
    parser_pc5.set_defaults(func=run_pc5) # Link sub-command to its function
    
    # --- Parser for preprocess sub-command ---
    parser_preprocess = subparsers.add_parser(
        'preprocess', 
        parents=[parent_parser], # Inherit shared arguments
        help='Run the preprocessing for the tool predictions.',
        description='Preprocesses the data of the predictions required to perform the benchmark.'
    )
    parser_preprocess.set_defaults(func=run_preprocess) # Link sub-command to its function
    
    # --- Parser for bench_by_aspect sub-command ---
    parser_bench_by_aspect = subparsers.add_parser(
        'bench_by_aspect', 
        parents=[parent_parser], # Inherit shared arguments
        help='Run the benchmark separating the data by aspect.',
        description='Computes the metrics and the performances comparing the predictions to the ground truth.'
    )
    parser_bench_by_aspect.set_defaults(func=run_bench_by_aspect) # Link sub-command to its function
    
    # --- Parser for bench_general sub-command ---
    parser_bench_general = subparsers.add_parser(
        'bench_general', 
        parents=[parent_parser], # Inherit shared arguments
        help='Run the benchmark keeping all the aspects together.',
        description='Computes the metrics and the performances comparing the predictions to the ground truth.'
    )
    parser_bench_general.set_defaults(func=run_bench_general) # Link sub-command to its function


    # --- Argument Parsing and Config Loading ---
    args = parser.parse_args()

    # --- Load the configuration from the YAML file ---
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # --- Merge CLI overrides into the config ---
    # Create a dictionary of parameters that were actually provided on the command line
    cli_overrides = {key: value for key, value in vars(args).items() if value is not None and key not in ['command', 'config', 'func']}

    if cli_overrides:
        print("[INFO] Applying command-line overrides...")
        # Safely get the configuration section for the current command.
        func_config_section = config.get(args.command, {})
        
        # Now, loop through the overrides and update this specific section
        for key, value in cli_overrides.items():
            print(f"  - In section '{args.command}', setting '{key}' to '{value}'")
            func_config_section[key] = value
            
        # Finally, put the updated section back into the main config.
        # This handles the case where the section was missing to begin with.
        config[args.command] = func_config_section
            
    # ===================================================================
    #                Verbose Output Options
    # ===================================================================
    # Print the final, effective configuration to standard error.
    # Use stderr so it doesn't pollute standard output if the user is piping results to another program
        
    print("="*60, file=sys.stderr)
    print("          FINAL EFFECTIVE CONFIGURATION FOR THIS RUN", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    print(yaml.dump(config), file=sys.stderr)
    print("="*60, file=sys.stderr)
    # ===================================================================

    # Run the dir setup
    try:
        dir_tree = directory_setup(config)  # Lancia un errore per le cartelle
        print("[SUCCESS] Directory structure setup is complete.")
    except Exception as e:
        # Catch any other unexpected error during setup
        print("\n" + "="*60, file=sys.stderr)
        print("          FATAL SETUP ERROR: UNEXPECTED ISSUE", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"An unexpected error occurred during the setup phase: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Execute the chosen function ---
    # `args.func` holds the actual function to run, pass it the final, config dictionary.
    try:
        args.func(config, dir_tree)
    except Exception as e:
        print("\n" + "="*60, file=sys.stderr)
        print("          PIPELINE ERROR: UNEXPECTED FAILURE", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"An unexpected error occurred during the analysis: {e}", file=sys.stderr)
    finally:
        # Get the cleanup setting from the config. Default to `False` (meaning we DO clean up).
        cleanup_enabled = not config.get('shared', {}).get('keep_tmp', True)
        tmp_dir = dir_tree['tmp_dir_path']
        if cleanup_enabled:
            print(f"[INFO] `keep_tmp` is false. Cleaning up temporary directory: {tmp_dir}")
            try:
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)
                    print("[INFO] Temporary directory successfully removed.")
            except OSError as e:
                print(f"[WARNING] Error deleting directory {tmp_dir}. Error: {e}", file=sys.stderr)
        else:
            print(f"[INFO] `keep_tmp` is true. Leaving temporary files at: {tmp_dir}")


if __name__ == "__main__":
    main() 