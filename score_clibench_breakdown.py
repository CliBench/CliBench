import os, json, argparse, random, math, re
import pickle
from preprocess import save_sparse, save_data
from preprocess.parse_csv import Mimic3Parser, Mimic4Parser, EICUParser, Mimic4NoteParser
from preprocess.encode import encode_code
from preprocess.build_dataset import split_patients, build_code_xy, build_heart_failure_y
from preprocess.auxiliary import generate_code_code_adjacent, generate_neighbors, normalize_adj, divide_middle, generate_code_levels
import pandas as pd
from tqdm import tqdm
import simple_icd_10_cm as cm
from statistics import mean, median
from sentence_transformers import SentenceTransformer, util

target_task = 'target_diagnoses'
llm_name = 'gpt-4-0125-preview'
# llm_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
save_name = llm_name.split('/')[-1] if '/' in llm_name else llm_name

output_parsed_save_path = os.path.join('data', 'mimic4', f'{target_task}_output', f'{save_name}_parsed.json')
result_save_dir = os.path.join('data', 'mimic4', f'{target_task}_result')

with open(output_parsed_save_path, 'r') as f:
    data_gold = json.load(f)

if target_task == 'target_diagnoses':
    # ['E08.3293', 'E08.329', 'E08.32', 'E08.3', 'E08', 'E08-E13', '4']
    granularity_index = [-1, -2, -3, -4, 0]
    granularity_name = ['l1_chapter', 'l2_category-groups', 'l3_category', 'l4_sub-category', 'l5_leaf']
else:
    raise NotImplementedError

# Load ontology for this task
def universal_get_ancestors(code, level_idx):
    if target_task == 'target_diagnoses':
        ancs = [code] + cm.get_ancestors(code)
    else:
        raise NotImplementedError
    try:
        return ancs[level_idx]
    except:
        return code
    
def group_data(adm_data, group_key):
    print(f'------- filtering for {group_key}')
    if type(adm_data[0][group_key]) == list:
        all_labels = []
        for dp in adm_data:
            all_labels.extend(dp[group_key])
        all_labels = sorted(list(set(all_labels)))
    else:
        all_labels = sorted(list(set([dp[group_key] for dp in adm_data])))
    data_index_map = {}
    count_map = {}
    for label in all_labels:
        data_index_map[label] = []
    for i, dp in enumerate(adm_data):
        if type(adm_data[i][group_key]) == list:
            for label_this in dp[group_key]:
                data_index_map[label_this].append(i)
        else:
            data_index_map[dp[group_key]].append(i)
    count_per_label = [len(data_index_map[label]) for label in all_labels]
    count_small_bound = min(count_per_label)
    for label, count in zip(all_labels, count_per_label):
        count_map[label] = count
        print(f'{label}: {count}')
    print(f'total number of labels: {len(all_labels)}')
    print('lowest category count:', count_small_bound)
    return all_labels, data_index_map, count_small_bound, count_map

# Create processed category flags
data_gold_new = []
for i, dp in enumerate(data_gold):
    # Create diagnosis chapters of this data instance
    diag_codes = [item[0] for item in dp['target_diagnoses']]
    uniq_chapter = list(set([cm.get_ancestors(cm.add_dot(c))[-1] for c in diag_codes]))
    dp['_target_diagnoses_chapters'] = uniq_chapter

    # Simplify race
    race_new = ''
    race_start_keywords = ['ASIA', 'BLACK', 'HISPANIC/LATINO', 'WHITE']
    for race_start_this in race_start_keywords:
        if dp['patient_race'].startswith(race_start_this):
            race_new = race_start_this
    if race_new == '':
        race_new = dp['patient_race']
    if race_new == 'PORTUGUESE' or race_new == 'SOUTH AMERICAN':
        race_new = 'OTHER'
    if race_new == 'PATIENT DECLINED TO ANSWER' or race_new == 'UNKNOWN':
        race_new = 'UNKNOWN/NOT ANSWER'
    dp['_patient_race_simplified'] = race_new

    data_gold_new.append(dp)
data_gold = data_gold_new

check_dist_fields = [
    '_target_diagnoses_chapters',
    '_service_processed',
    'admission_type',
    'hospital_expire_flag',
    'admission_location',
    'discharge_location',
    'patient_insurance',
    'patient_lang',
    'patient_marital',
    'patient_race',
    '_patient_race_simplified',
    'patient_gender'
]

for field in check_dist_fields:
    aggregated_result_all_groups = {}
    all_labels, data_index_map, count_small_bound, count_map = group_data(data_gold, field)
    print(f'All possible label for {field}: {all_labels}')
    for group_name in all_labels:
        print(f'=============== {field} : {group_name}')
        selected_index = data_index_map[group_name]
        aggregated_result_all_levels = {}
        f1_list_all_levels = {}
        latex_text = ''
        for level_name, level_idx in zip(granularity_name, granularity_index):
            aggregated_result_all_levels[level_name] = {
                'count_true': 0,
                'count_gold': 0,
                'count_pred': 0,
            }
            f1_list_all_levels[level_name] = []

        for dp_i, dp in enumerate(tqdm(data_gold)):
            if 'extracted_details' not in dp:
                continue

            # filter the data point according to grouping criteria
            if dp_i not in selected_index:
                continue

            # Get ground-truth
            if target_task == 'target_diagnoses':
                codes_gold = list(set([item[0] for item in dp['target_diagnoses']]))
                codes_gold = [cm.add_dot(c) for c in codes_gold]
            else:
                raise NotImplementedError

            codes_pred = []
            for point_item in dp['extracted_details']:
                codes_pred_this = point_item['codes_pred']
                codes_pred.extend(codes_pred_this)
            codes_pred = list(set(codes_pred))

            if len(codes_pred) == 0:
                print(f"Did not extract any codes for admission {dp['hadm_id']}")
                continue

            for level_name, level_idx in zip(granularity_name, granularity_index):
                # Convert code to the correct granularity
                codes_gold_this_level = list(set([universal_get_ancestors(c, level_idx) for c in codes_gold]))
                codes_pred_this_level = list(set([universal_get_ancestors(c, level_idx) for c in codes_pred]))

                # Add count and F1 score for this one
                count_true_this = len(set(codes_gold_this_level).intersection(set(codes_pred_this_level)))
                count_gold_this = len(codes_gold_this_level)
                count_pred_this = len(codes_pred_this_level)
                f1_this = 2 * count_true_this / (count_gold_this + count_pred_this) if count_gold_this + count_pred_this > 0 else 0
                
                aggregated_result_all_levels[level_name]["count_true"] += count_true_this
                aggregated_result_all_levels[level_name]["count_gold"] += count_gold_this
                aggregated_result_all_levels[level_name]["count_pred"] += count_pred_this
                f1_list_all_levels[level_name].append(f1_this)

        # Calculate aggregated scores
        prec_of_levels = []
        reca_of_levels = []
        f1_of_levels = []
        latex_texts = []

        for level_name, level_idx in zip(granularity_name, granularity_index):
            aggregated_result = aggregated_result_all_levels[level_name]
            prec = aggregated_result['count_true']/aggregated_result['count_pred'] if aggregated_result['count_pred'] > 0 else 0
            reca = aggregated_result['count_true']/aggregated_result['count_gold'] if aggregated_result['count_gold'] > 0 else 0
            f1 = 2 * prec * reca / (prec + reca) if prec + reca > 0 else 0
            aggregated_result_all_levels[level_name]['precision'] = prec
            aggregated_result_all_levels[level_name]['recall'] = reca
            aggregated_result_all_levels[level_name]['f1'] = f1
            aggregated_result_all_levels[level_name]['f1_macro'] = mean(f1_list_all_levels[level_name])
            prec_of_levels.append(prec)
            reca_of_levels.append(reca)
            f1_of_levels.append(f1)

            aggregated_result_all_levels[level_name] = aggregated_result
            latex_texts.append(f" & {aggregated_result['precision'] * 100:.2f} & {aggregated_result['recall'] * 100:.2f} & {aggregated_result['f1'] * 100:.2f}")

        aggregated_result_all_levels['avg_precision'] = mean(prec_of_levels)
        aggregated_result_all_levels['avg_recall'] = mean(reca_of_levels)
        aggregated_result_all_levels['avg_f1'] = mean(f1_of_levels)
        latex_texts.append(f" & {aggregated_result_all_levels['avg_precision'] * 100:.2f} & {aggregated_result_all_levels['avg_recall'] * 100:.2f} & {aggregated_result_all_levels['avg_f1'] * 100:.2f}")
        aggregated_result_all_levels['latex'] = latex_texts
        print(json.dumps(aggregated_result_all_levels, indent=4))
        aggregated_result_all_groups[group_name] = aggregated_result_all_levels
    with open(os.path.join(result_save_dir, f'{save_name}_by_{field}.json'), 'w') as f:
        json.dump(aggregated_result_all_groups, f, indent=4)