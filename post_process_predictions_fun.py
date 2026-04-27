import pandas as pd
from tqdm import tqdm
import pandas as pd
import csv
import os
from collections import defaultdict
from typing import Optional, List, Dict, Set

def ensure_directory_exists(directory_path):
    """Check if a directory exists and create it if it doesn't."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def correct_IDs(alias_path, gt_path, out_path):
    # 1. Load the alias and data files
    # Using sep='\t' since they are .tsv files
    alias_df = pd.read_csv(alias_path, sep='\t')
    data_df = pd.read_csv(gt_path, sep='\t')

    for index, row in tqdm(alias_df.iterrows()):
        # Access specific columns in the current row
        target_alias = row['match_IDs']
        # if target_alias in missing_ids:
        replacement_main = row['Query_ID']
        
        # Access and update the 'ID' column in data_df where it matches the alias
        # .loc[row_condition, column_label] is used for direct access
        data_df.loc[data_df['Query_ID'] == target_alias, 'Query_ID'] = replacement_main

    # 4. Save the updated data to a new file
    # data_df.to_csv('gtg_C5_corrected.tsv', sep='\t', index=False)
    data_df.to_csv(out_path, sep='\t', index=False)
    print("Substitution complete. The updated file is saved as 'updated_data.tsv'.")



def restructure_go_data(aliases_path, annotations_path, output_path):
    # --- Step 1: Build aliases map ---
    aliases = defaultdict(set)

    with open(aliases_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            representative = row[0].strip()
            aliased = row[1].strip()
            
            # Prevent the header row from becoming a protein entry
            if representative.lower() == 'query_id':
                continue
                
            if representative and aliased:
                aliases[representative].add(aliased)

    # --- Step 2: Parse annotations ---
    # Robust aspect mapping (handles F, M, C, P or if they are already MF, CC, BP)
    aspect_map = {
        'F': 'MF', 'M': 'MF', 'MF': 'MF', 
        'C': 'CC', 'CC': 'CC', 
        'P': 'BP', 'BP': 'BP'
    }

    annotations = defaultdict(lambda: defaultdict(list))

    with open(annotations_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            query_id = row['Query_ID'].strip()
            go_id = row['GO_ID'].strip()
            evco = row['EvCo_2024'].strip()
            aspect_raw = row['Aspect'].strip().upper()
            date = row['Date'].strip()
            kt = row['knowledge_type'].strip()

            aspect = aspect_map.get(aspect_raw)
            if not aspect:
                continue

            annotation_str = f"{go_id}|{evco}|{date}|{kt}"
            annotations[query_id][aspect].append(annotation_str)

    # --- Step 3: Collect all unique representative ACCIDs ---
    all_representatives = set(aliases.keys())
    all_aliased_ids = {alias for alias_set in aliases.values() for alias in alias_set}
    all_query_ids = set(annotations.keys())
    
    # Query IDs that are not in aliases are standalone representatives
    standalone = all_query_ids - all_representatives - all_aliased_ids
    all_representatives.update(standalone)

    # Helper function to deduplicate by GO_ID (keeps the most recent annotation)
    def add_unique_go_term(target_dict, ann_str):
        parts = ann_str.split('|')
        go_id = parts[0]
        date = parts[2] if len(parts) > 2 else ""
        
        if go_id in target_dict:
            existing_parts = target_dict[go_id].split('|')
            existing_date = existing_parts[2] if len(existing_parts) > 2 else ""
            if date > existing_date:
                target_dict[go_id] = ann_str
        else:
            target_dict[go_id] = ann_str

    # --- Step 4: Write output ---
    written_count = 0
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Representative_ACCID', 'Aliases', 'MF_Annotations', 'CC_Annotations', 'BP_Annotations'])

        for rep in sorted(all_representatives):
            alias_set = aliases.get(rep, set())
            aliases_str = ','.join(sorted(alias_set)) if alias_set else ''

            mf_dict = {}
            cc_dict = {}
            bp_dict = {}

            # 1. Get annotations for the representative itself
            if rep in annotations:
                for ann in annotations[rep].get('MF', []): add_unique_go_term(mf_dict, ann)
                for ann in annotations[rep].get('CC', []): add_unique_go_term(cc_dict, ann)
                for ann in annotations[rep].get('BP', []): add_unique_go_term(bp_dict, ann)

            # 2. Get annotations for all associated aliases
            for alias in alias_set:
                if alias in annotations:
                    for ann in annotations[alias].get('MF', []): add_unique_go_term(mf_dict, ann)
                    for ann in annotations[alias].get('CC', []): add_unique_go_term(cc_dict, ann)
                    for ann in annotations[alias].get('BP', []): add_unique_go_term(bp_dict, ann)

            mf_str = ','.join([mf_dict[k] for k in sorted(mf_dict.keys())])
            cc_str = ','.join([cc_dict[k] for k in sorted(cc_dict.keys())])
            bp_str = ','.join([bp_dict[k] for k in sorted(bp_dict.keys())])

            if not mf_str and not cc_str and not bp_str:
                continue

            writer.writerow([rep, aliases_str, mf_str, cc_str, bp_str])
            written_count += 1

    print(f"Done. {written_count} fully annotated representative ACCIDs written to {output_path}")



def split_by_knowledge_type(input_file, nk_output_file, lk_output_file, lk_kk_output_file):
    # 1. Load the restructured data
    # Use dtype=str to ensure pandas doesn't accidentally convert any IDs to numbers
    df = pd.read_csv(input_file, sep='\t', dtype=str).fillna('')
    
    # 2. Define a helper function to filter the joined annotation strings
    def filter_annots(annot_str, valid_types):
        if not annot_str:
            return ''
        
        # Split by comma to get individual annotations (e.g., GO:123|IBA|2024|NK)
        annotations = annot_str.split(',')
        filtered_annotations = []
        
        for ann in annotations:
            parts = ann.split('|')
            # The knowledge_type is the 4th item (index 3) in the string
            if len(parts) >= 4:
                k_type = parts[3].strip()
                if k_type in valid_types:
                    filtered_annotations.append(ann)
                    
        # Rejoin the filtered annotations with a comma
        return ','.join(filtered_annotations)

    # 3. Create two independent copies of the dataframe for our two outputs
    df_nk = df.copy()
    df_lk = df.copy()
    df_lk_kk = df.copy()
    
    annot_cols = ['MF_Annotations', 'CC_Annotations', 'BP_Annotations']
    
    # Apply the filter for 'NK'
    for col in annot_cols:
        df_nk[col] = df_nk[col].apply(lambda x: filter_annots(x, ['NK']))
        
    # Apply the filter for 'LK1' and 'LK2'
    for col in annot_cols:
        df_lk[col] = df_lk[col].apply(lambda x: filter_annots(x, ['LK1', 'LK2']))

    # Apply the filter for 'LK1' and 'LK2' and 'KK'
    for col in annot_cols:
        df_lk_kk[col] = df_lk_kk[col].apply(lambda x: filter_annots(x, ['LK1', 'LK2', 'KK']))
        
    # 4. Cleanup: Remove rows that have no annotations left after filtering
    # If a representative had ONLY 'NK' annotations, it will be completely empty in the LK file.
    # It is usually best to drop these empty rows.
    nk_has_data = (df_nk['MF_Annotations'] != '') | (df_nk['CC_Annotations'] != '') | (df_nk['BP_Annotations'] != '')
    df_nk = df_nk[nk_has_data]
                     
    lk_has_data = (df_lk['MF_Annotations'] != '') | (df_lk['CC_Annotations'] != '') | (df_lk['BP_Annotations'] != '')
    df_lk = df_lk[lk_has_data]

    lk_kk_has_data = (df_lk_kk['MF_Annotations'] != '') | (df_lk_kk['CC_Annotations'] != '') | (df_lk_kk['BP_Annotations'] != '')
    df_lk_kk = df_lk_kk[lk_kk_has_data]
                     
    df_nk = df_nk.sort_values(by='Representative_ACCID', ascending=True)
    df_lk = df_lk.sort_values(by='Representative_ACCID', ascending=True)
    df_lk_kk = df_lk_kk.sort_values(by='Representative_ACCID', ascending=True)

    # 5. Export to the new TSV files
    df_nk.to_csv(nk_output_file, sep='\t', index=False)
    df_lk.to_csv(lk_output_file, sep='\t', index=False)
    df_lk_kk.to_csv(lk_kk_output_file, sep='\t', index=False)
    
    print(f"Exported NK annotations to {nk_output_file} ({len(df_nk)} rows)")
    print(f"Exported LK annotations to {lk_output_file} ({len(df_lk)} rows)")
    print(f"Exported LK/KK annotations to {lk_kk_output_file} ({len(df_lk_kk)} rows)")





def split_aliases(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def split_go_annotations(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def extract_go_id(annotation: str) -> Optional[str]:
    if not annotation:
        return None

    parts = annotation.split("|")
    if not parts:
        return None

    go_id = parts[0].strip()
    if go_id.startswith("GO:"):
        return go_id

    return None


def parse_file(filepath: str) -> Dict[str, int]:
    rep_count = 0
    protein_total = 0

    mf_total = 0
    cc_total = 0
    bp_total = 0

    mf_unique: Set[str] = set()
    cc_unique: Set[str] = set()
    bp_unique: Set[str] = set()

    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        required_columns = {
            "Representative_ACCID",
            "Aliases",
            "MF_Annotations",
            "CC_Annotations",
            "BP_Annotations",
        }

        if not reader.fieldnames:
            raise ValueError("File vuoto o header non trovato")

        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError("Colonne mancanti: {}".format(", ".join(missing)))

        for row in reader:
            representative = (row.get("Representative_ACCID") or "").strip()
            aliases = split_aliases((row.get("Aliases") or "").strip())

            mf_annotations = split_go_annotations((row.get("MF_Annotations") or "").strip())
            cc_annotations = split_go_annotations((row.get("CC_Annotations") or "").strip())
            bp_annotations = split_go_annotations((row.get("BP_Annotations") or "").strip())

            if representative:
                rep_count += 1

            protein_total += (1 if representative else 0) + len(aliases)

            mf_total += len(mf_annotations)
            cc_total += len(cc_annotations)
            bp_total += len(bp_annotations)

            for ann in mf_annotations:
                go_id = extract_go_id(ann)
                if go_id:
                    mf_unique.add(go_id)

            for ann in cc_annotations:
                go_id = extract_go_id(ann)
                if go_id:
                    cc_unique.add(go_id)

            for ann in bp_annotations:
                go_id = extract_go_id(ann)
                if go_id:
                    bp_unique.add(go_id)

    return {
        "Representative_ACCID_totali": rep_count,
        "Proteine_totali": protein_total,
        "MF_GO_totali": mf_total,
        "CC_GO_totali": cc_total,
        "BP_GO_totali": bp_total,
        "MF_GO_unici": len(mf_unique),
        "CC_GO_unici": len(cc_unique),
        "BP_GO_unici": len(bp_unique),
    }


def make_stats(input_file, type, outfile):
    stats = parse_file(input_file)

    with open(outfile, 'a') as f:
        print(f"\nExtracted stats from type {type}:\n")
        f.write(f"\nExtracted stats from type {type}:\n")
        for key, value in stats.items():
            print("{:<30} {}".format(key + ":", value))
            f.write("{:<30} {}".format(key + ":", value)+"\n")



##############################################################################
######################### POST-PROCESSING  ###################################
##############################################################################


def postprocess(gt_path, model_name, cafa, dir_tree, extra_correction=False):

    alias_path = dir_tree['prep_data_dir_path'] + f'/aliases_{cafa}_nx.tsv' 

    
    ensure_directory_exists(dir_tree['post_dir_path'])
    postproc_dir = dir_tree['post_dir_path'] + '/'

    if extra_correction:
        gt_correct_IDs = postproc_dir + "gt_correct_IDs.tsv" 
        # Cover for specific edge-case
        correct_IDs(alias_path, gt_path, gt_correct_IDs)
    else:
        gt_correct_IDs = gt_path

    ensure_directory_exists(dir_tree['post_data_dir_path'])
    post_persist_dir = dir_tree['post_data_dir_path'] + '/'
    # Restructure the data itself
    gt_abridged = post_persist_dir + f"gt_abridged_{model_name}_{cafa}_general.tsv" 
    restructure_go_data(alias_path, gt_correct_IDs, gt_abridged)

    # Split into knowledge types
    gt_abridged_NK = post_persist_dir + f"gt_abridged_{model_name}_{cafa}_NK.tsv" 
    gt_abridged_LK = post_persist_dir + f"gt_abridged_{model_name}_{cafa}_LK.tsv" 
    gt_abridged_LK_KK = post_persist_dir + f"gt_abridged_{model_name}_{cafa}_LK_KK.tsv" 
    split_by_knowledge_type(gt_abridged, gt_abridged_NK, gt_abridged_LK, gt_abridged_LK_KK)

    stat_outfile = post_persist_dir + f"stat_report_{model_name}.tsv" 
    with open(stat_outfile, "w") as file:
        file.write(f"-----------   Statistical report of the ground truth   -----------\n")

    # Print some stats about everything
    make_stats(gt_abridged, 'general', stat_outfile)
    make_stats(gt_abridged_NK, 'NK', stat_outfile)
    make_stats(gt_abridged_LK, 'LK', stat_outfile)
    make_stats(gt_abridged_LK_KK, 'LK and KK', stat_outfile)


    print("[SUCCESS] --- Postprocessed data saved successfully! ---")






    
