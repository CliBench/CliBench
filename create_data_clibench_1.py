import os, json, argparse, random, math, re
import pickle

from preprocess import save_sparse, save_data
from preprocess.parse_csv import Mimic3Parser, Mimic4Parser, EICUParser, Mimic4NoteParser
from preprocess.encode import encode_code
from preprocess.build_dataset import split_patients, build_code_xy, build_heart_failure_y
from preprocess.auxiliary import generate_code_code_adjacent, generate_neighbors, normalize_adj, divide_middle, generate_code_levels
import pandas as pd
from tqdm import tqdm

conf = {
    'mimic3': {
        'parser': Mimic3Parser,
        'train_num': 6000,
        'test_num': 1000,
        'threshold': 0.01
    },
    'mimic4': {
        'parser': Mimic4Parser,
        'train_num': 8000,
        'test_num': 1000,
        'threshold': 0.01,
    },
    'mimic4note': {
        'parser': Mimic4NoteParser,
        'train_num': 8000,
        'test_num': 1000,
        'threshold': 0.01,
    },
    'eicu': {
        'parser': EICUParser,
        'train_num': 8000,
        'test_num': 1000,
        'threshold': 0.01
    }
}
END_SEGMENT_TOKEN = '<end_of_visit>'

def verbalize_codes(codes, shuffle_code_order=False, random_seed=0, mode='code_only', code_descriptions=None):
    # verbalize code list of a visit
    code_tokens = [f'ICD9_{c}' for c in codes]
    if shuffle_code_order:
        random.seed(random_seed)
        random.shuffle(code_tokens)
    if mode == 'code_only':
        result = ' '.join(code_tokens)
        result += END_SEGMENT_TOKEN
    elif mode == 'code+desc':
        code_segment = []
        for code in code_tokens:
            code_clean = code.replace('ICD9_', '')
            if code_clean in code_descriptions:
                code_segment.append(f'{code} ({code_descriptions[code_clean]})')
            else:
                code_segment.append(code)

        result = '; '.join(code_segment)
    return result

def clean_note(note_text):
    sections = re.split(r'\n\s*\n', note_text)

    service = ''
    sections_clean = []
    sections_removed = []
    for sec_i, sec in enumerate(sections):
        pattern_profile_1 = r'name:\s+___'
        pattern_profile_2 = r'unit no:\s+___'
        pattern_profile_3 = r'admission date:\s+___'
        pattern_profile_4 = r'discharge date:\s+___'
        pattern_profile_5 = r'date of birth:\s+___'
        pattern_profile_6 = r'sex:\s+___'
        pattern_service = r'service:\s+'
        pattern_procedure = r'major surgical or invasive procedure'
        pattern_medications = r'discharge medications'
        pattern_disposition = r'discharge disposition'
        pattern_diagnosis = r'discharge diagnosis'
        pattern_instructions = r'discharge instructions'
        pattern_followup = r'followup instructions'
        pattern_physical_exam = r'discharge physical examination'
        pattern_discharge_labs_1 = r'discharge labs'
        pattern_discharge_labs_2 = r'labs on discharge'
        pattern_discharge_condition = r'discharge condition'
        pattern_discharge_any = r'discharge'

        # extract service info
        if re.search(pattern_service, sec.lower()):
            service = re.split(pattern_service, sec.lower())[1].strip()

        # remove anything after Discharge Instructions
        if re.search(pattern_instructions, sec.lower()):
            break

        if not re.search(pattern_profile_1, sec.lower()) and \
            not re.search(pattern_profile_2, sec.lower()) and \
            not re.search(pattern_profile_3, sec.lower()) and \
            not re.search(pattern_profile_4, sec.lower()) and \
            not re.search(pattern_profile_5, sec.lower()) and \
            not re.search(pattern_profile_6, sec.lower()) and \
            not re.search(pattern_procedure, sec.lower()) and \
            not re.search(pattern_medications, sec.lower()) and \
            not re.search(pattern_disposition, sec.lower()) and \
            not re.search(pattern_diagnosis, sec.lower()) and \
            not re.search(pattern_instructions, sec.lower()) and \
            not re.search(pattern_followup, sec.lower()) and \
            not re.search(pattern_physical_exam, sec.lower()) and \
            not re.search(pattern_discharge_labs_1, sec.lower()) and \
            not re.search(pattern_discharge_labs_2, sec.lower()) and \
            not re.search(pattern_discharge_condition, sec.lower()):
            sections_clean.append(sec)
        else:
            sections_removed.append(sec)
    
    extracted_info = {'service': service}
    note_text_clean = '\n\n'.join(sections_clean)

    """
    Name: ___ Unit No: ___ ...

    Service: MEDICINE

    Major Surgical or Invasive Procedure: 
    none while on medical service

    Discharge Medications:
    1. Lovenox 40 mg ...
    2. Dilaudid 2md Tablet ...

    Discharge Disposition:
    Home With Service

    Discharge Diagnosis:
    Primary Diagnosis:
    Septic Arthritis
    Acute Kidney Injury

    Discharge Instructions:
    [until the very end...]

    """

    return note_text_clean, extracted_info

def clean_radiology_note(note_text):
    sections = re.split(r'\n\s*\n', note_text)
    technique = ''
    for sec_i, sec in enumerate(sections):
        pattern_service = r'technique:\s+'
        # extract technique info
        if re.search(pattern_service, sec.lower()):
            technique = re.split(pattern_service, sec.lower())[1].strip()
    extracted_info = {'technique': technique}
    return note_text, extracted_info


def build_seqs(pids, patient_admission, admission_codes, 
               shuffle_code_order=False, every_visit_as_target=False,
               shuffle_input_count=5, shuffle_target_count=5,
               code_descriptions=None, mode='code_only'):
    # Mode code_only: just code, use Diagnosis codes for the next patient visit: to separate different visit
    # Mode code+desc: add code description following that code
    dps = []
    for i, pid in enumerate(pids):
        admissions = patient_admission[pid]
        # admissions is like
        # [{'adm_id': 152223, 'adm_time': Timestamp('2153-09-03 07:15:00')},
        #  {'adm_id': 124321, 'adm_time': Timestamp('2157-10-18 19:34:00')}]
        admission_list_full = [dp['adm_id'] for dp in admissions]
        admission_list_candidates = []

        if every_visit_as_target and len(admission_list_full) > 2:
            for visit_i in range(2, len(admission_list_full) + 1):
                admission_list_this = admission_list_full[:visit_i]
                admission_list_candidates.append(admission_list_this)
        else:
            admission_list_candidates = [admission_list_full]

        for admission_list_index, admission_list in enumerate(admission_list_candidates):
            if shuffle_code_order:
                random_seeds = list(range(101, 101 + shuffle_input_count))
            else:
                random_seeds = [101]
            for variant_input_index, random_seed in enumerate(random_seeds):
                # get seqs for seq input
                codes_seqs = []
                for j, aid in enumerate(admission_list[:-1]):
                    codes = admission_codes[aid]
                    # codes is like
                    # ['414.01', '411.1', '424.1', 'V45.82', '272.4', '401.9', '600.00', '389.9']
                    codes_seq = verbalize_codes(codes, shuffle_code_order=shuffle_code_order, random_seed=random_seed, mode=mode, code_descriptions=code_descriptions)
                    codes_seq = 'Diagnosis codes for the visit: ' + codes_seq
                    codes_seqs.append(codes_seq)
                adm_seq_i = ' '.join(codes_seqs)
                # adm_seq_i = adm_seq_i + ' Diagnosis codes for the next patient visit: '

                if shuffle_code_order:
                    random_seeds_inner = list(range(101, 101 + shuffle_target_count))
                else:
                    random_seeds_inner = [101]
                for variant_target_index, random_seed_inner in enumerate(random_seeds_inner):
                    # get seqs for seq target
                    codes_seqs = []
                    target_code_list = []
                    for j, aid in enumerate([admission_list[-1]]):
                        codes = admission_codes[aid]
                        codes_seq = verbalize_codes(codes, shuffle_code_order=shuffle_code_order, random_seed=random_seed_inner, mode=mode, code_descriptions=code_descriptions)
                        codes_seqs.append(codes_seq)
                        target_code_list.extend(codes)
                    adm_seq_o = ' '.join(codes_seqs)
                    
                    dp = {
                        'id': f"{i}_admission-segment-{admission_list_index}_input-{variant_input_index}_target-{variant_target_index}",
                        'patient_id': int(pid), 
                        'text': adm_seq_i + adm_seq_o,
                        'input': adm_seq_i,
                        'target_gold': adm_seq_o,
                        'target_gold_list': target_code_list,
                    }
                    dps.append(dp)
    
    return dps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation configurations')
    parser.add_argument('--dataset_hosp', type=str, nargs='?', default='mimic4',
                        choices=['mimic3', 'mimic4', 'eicu'],
                        help='dataset name to be used')
    parser.add_argument('--dataset_note', type=str, nargs='?', default='mimic4note',
                        choices=['mimic4note'],
                        help='dataset name to be used')
    parser.add_argument('--data_path_hosp', type=str, nargs='?', default='data/physionet.org/files/mimiciv/2.2/hosp',
                        help='path to the data directory for hosp data')
    parser.add_argument('--data_path_note', type=str, nargs='?', default='data/physionet.org/files/mimic-iv-note/2.2/note',
                        help='path to the data directory for note data')
    parser.add_argument('--save_path_parsed', type=str, nargs='?', default='data/mimic4/parsed',
                        help='path to save the intermediate parsed data')
    parser.add_argument('--mode', type=str, nargs='?', default='code_only',
                        choices=['code_only', 'code+desc'],
                        help='dataset name to be used')
    parser.add_argument('--top_n', type=int, nargs='?', default=-1,
                        help='select only top n data points for each split, if top_n > 0')
    parser.add_argument('--from_saved', action='store_true', help='whether used saved cache')
    parser.add_argument('--shuffle_code_order', action='store_false', help='whether shuffle code and create variant for training input sequence')
    parser.add_argument('--shuffle_input_count', type=int, nargs='?', default=0,
                        help='number of varient for code list of each visit in the input sequence')
    parser.add_argument('--shuffle_target_count', type=int, nargs='?', default=0,
                        help='number of varient for code list of each visit in the output sequence')
    args = parser.parse_args()

    assert args.shuffle_code_order or args.shuffle_input_count <= 0, 'if shuffle_code_order is False, shuffle_input_count should be not positive'
    assert args.shuffle_code_order or args.shuffle_target_count <= 0, 'if shuffle_code_order is False, shuffle_target_count should be not positive'

    if os.path.exists(os.path.join(args.save_path_parsed, 'admission_codes.pkl')) and args.from_saved:
        print(f'using intermediate files saved in {args.save_path_parsed} for hosp')
        admission_codes = pickle.load(open(os.path.join(args.save_path_parsed, 'admission_codes.pkl'), 'rb'))
        admission_metadata = pickle.load(open(os.path.join(args.save_path_parsed, 'admission_metadata.pkl'), 'rb'))
        patient_metadata = pickle.load(open(os.path.join(args.save_path_parsed, 'patient_metadata.pkl'), 'rb'))
        admission_labevents = pickle.load(open(os.path.join(args.save_path_parsed, 'admission_labevents.pkl'), 'rb'))
        admission_prescriptions = pickle.load(open(os.path.join(args.save_path_parsed, 'admission_prescriptions.pkl'), 'rb'))
        admission_procedures = pickle.load(open(os.path.join(args.save_path_parsed, 'admission_procedures.pkl'), 'rb'))
        admission_microbiologyevents = pickle.load(open(os.path.join(args.save_path_parsed, 'admission_microbiologyevents.pkl'), 'rb'))
        admission_services = pickle.load(open(os.path.join(args.save_path_parsed, 'admission_services.pkl'), 'rb'))
        admission_transfers = pickle.load(open(os.path.join(args.save_path_parsed, 'admission_transfers.pkl'), 'rb'))
    else:
        if not os.path.exists(args.data_path_hosp):
            os.makedirs(args.data_path_hosp)
            print(f'please put the hosp CSV files in `{args.data_path_hosp}`')
            exit()
        if not os.path.exists(args.data_path_note):
            os.makedirs(args.data_path_note)
            print(f'please put the note CSV files in `{args.data_path_note}`')
            exit()
        
        print('hosp data parsing ...')
        parser_hosp = conf[args.dataset_hosp]['parser'](args.data_path_hosp)
        parsed_dict = parser_hosp.parse_diag()
        admission_codes = parsed_dict['admission_codes']
        admission_metadata = parsed_dict['admission_metadata']
        patient_metadata = parsed_dict['patient_metadata']
        admission_labevents = parsed_dict['admission_labevents']
        admission_prescriptions = parsed_dict['admission_prescriptions']
        admission_procedures = parsed_dict['admission_procedures']
        admission_microbiologyevents = parsed_dict['admission_microbiologyevents']
        admission_services = parsed_dict['admission_services']
        admission_transfers = parsed_dict['admission_transfers']
        if not os.path.exists(args.save_path_parsed):
            os.makedirs(args.save_path_parsed)
        pickle.dump(admission_codes, open(os.path.join(args.save_path_parsed, 'admission_codes.pkl'), 'wb'))
        pickle.dump(admission_metadata, open(os.path.join(args.save_path_parsed, 'admission_metadata.pkl'), 'wb'))
        pickle.dump(patient_metadata, open(os.path.join(args.save_path_parsed, 'patient_metadata.pkl'), 'wb'))
        pickle.dump(admission_labevents, open(os.path.join(args.save_path_parsed, 'admission_labevents.pkl'), 'wb'))
        pickle.dump(admission_prescriptions, open(os.path.join(args.save_path_parsed, 'admission_prescriptions.pkl'), 'wb'))
        pickle.dump(admission_procedures, open(os.path.join(args.save_path_parsed, 'admission_procedures.pkl'), 'wb'))
        pickle.dump(admission_microbiologyevents, open(os.path.join(args.save_path_parsed, 'admission_microbiologyevents.pkl'), 'wb'))
        pickle.dump(admission_services, open(os.path.join(args.save_path_parsed, 'admission_services.pkl'), 'wb'))
        pickle.dump(admission_transfers, open(os.path.join(args.save_path_parsed, 'admission_transfers.pkl'), 'wb'))

    if os.path.exists(os.path.join(args.save_path_parsed, 'admission_notes.pkl')) and args.from_saved:
        print(f'using intermediate files saved in {args.save_path_parsed} for note')
        admission_notes = pickle.load(open(os.path.join(args.save_path_parsed, 'admission_notes.pkl'), 'rb'))
        admission_radiology_notes = pickle.load(open(os.path.join(args.save_path_parsed, 'admission_radiology_notes.pkl'), 'rb'))
    else:
        print('note data parsing ...')
        parser_note = conf[args.dataset_note]['parser'](args.data_path_note)
        parsed_dict = parser_note.parse()
        admission_notes = parsed_dict['admission_notes']
        admission_radiology_notes = parsed_dict['admission_radiology_notes']
        if not os.path.exists(args.save_path_parsed):
            os.makedirs(args.save_path_parsed)
        pickle.dump(admission_notes, open(os.path.join(args.save_path_parsed, 'admission_notes.pkl'), 'wb'))
        pickle.dump(admission_radiology_notes, open(os.path.join(args.save_path_parsed, 'admission_radiology_notes.pkl'), 'wb'))

    if os.path.exists(os.path.join(args.save_path_parsed, 'diagcode_longtitle.json')) and args.from_saved:
        print(f'using intermediate files saved in {args.save_path_parsed} for dict')
        with open(os.path.join(args.save_path_parsed, 'diagcode_longtitle.json')) as f:
            diagcode_longtitle = json.load(f)
        with open(os.path.join(args.save_path_parsed, 'procedurecode_longtitle.json')) as f:
            procedurecode_longtitle = json.load(f)
        with open(os.path.join(args.save_path_parsed, 'labitem_labels.json')) as f:
            labitem_labels = json.load(f)
    else:
        print('dict data parsing ...')
        # load description for diagnosis icd codes
        icd_df = pd.read_csv(os.path.join(args.data_path_hosp, 'd_icd_diagnoses.csv'))
        diagcode_longtitle_raw = {'9': {}, '10': {}}
        diagcode_longtitle = {}
        for index, row in icd_df.iterrows():
            diagcode_longtitle_raw[str(row['icd_version'])][(row['icd_code'])] = row['long_title']
            diagcode_longtitle[f"ICD-{row['icd_version']}_{row['icd_code']}"] = row['long_title']
        print(f'there are total {len(diagcode_longtitle)} codes in d_icd_diagnoses.csv')
        print(f'ICD-9 codes count: {len(diagcode_longtitle_raw["9"])}')
        print(f'ICD-10 codes count: {len(diagcode_longtitle_raw["10"])}')
        with open(os.path.join(args.save_path_parsed, 'diagcode_longtitle.json'), 'w') as f:
            json.dump(diagcode_longtitle, f, indent=4, sort_keys=True)

        # load description for procedure icd codes
        icd_df = pd.read_csv(os.path.join(args.data_path_hosp, 'd_icd_procedures.csv'))
        procedurecode_longtitle_raw = {'9': {}, '10': {}}
        procedurecode_longtitle = {}
        for index, row in icd_df.iterrows():
            procedurecode_longtitle_raw[str(row['icd_version'])][(row['icd_code'])] = row['long_title']
            procedurecode_longtitle[f"ICD-{row['icd_version']}_{row['icd_code']}"] = row['long_title']
        print(f'there are total {len(procedurecode_longtitle)} codes in d_icd_procedures.csv')
        print(f'ICD-9 codes count: {len(procedurecode_longtitle_raw["9"])}')
        print(f'ICD-10 codes count: {len(procedurecode_longtitle_raw["10"])}')
        with open(os.path.join(args.save_path_parsed, 'procedurecode_longtitle.json'), 'w') as f:
            json.dump(procedurecode_longtitle, f, indent=4, sort_keys=True)

        # load description for lab item
        lab_df = pd.read_csv(os.path.join(args.data_path_hosp, 'd_labitems.csv'))
        labitem_labels = {}
        for index, row in lab_df.iterrows():
            labitem_labels[row['itemid']] = f"{row['label']} for {row['fluid']} ({row['category']})"
        print(f'there are total {len(labitem_labels)} lab items in d_labitems.csv')
        with open(os.path.join(args.save_path_parsed, 'labitem_labels.json'), 'w') as f:
            json.dump(labitem_labels, f, indent=4, sort_keys=True)

    # Clean medical discharge note and radiology note, to remove unnecessary/empty fields, remove conclusive info
    for adm_id, note_list in admission_notes.items():
        for i, note_item in enumerate(note_list):
            admission_notes[adm_id][i][0], extracted_info = clean_note(note_item[0])
            for info_name, info_value in extracted_info.items():
                admission_notes[adm_id][i].append(info_value)
    for adm_id, note_list in admission_radiology_notes.items():
        for i, note_item in enumerate(note_list):
            admission_radiology_notes[adm_id][i][0], extracted_info = clean_radiology_note(note_item[0])
            for info_name, info_value in extracted_info.items():
                admission_radiology_notes[adm_id][i].append(info_value)

    # Clean diagnosis and procedure list to remove all records in ICD-9-CM and ICD-9-PCS
    admission_codes_new = {}
    for adm_id, code_list in admission_codes.items():
        remove_flag = False
        for code_pair in code_list:
            if code_pair[1] == 9:
                remove_flag = True
                break
        if not remove_flag:
            admission_codes_new[adm_id] = code_list
    print(f'removed admission_codes records in ICD-9-CM, size from {len(admission_codes)} to {len(admission_codes_new)}')
    admission_codes = admission_codes_new
    admission_procedures_new = {}
    for adm_id, code_list in admission_procedures.items():
        remove_flag = False
        for code_pair in code_list:
            if code_pair[1] == 9:
                remove_flag = True
                break
        if not remove_flag:
            admission_procedures_new[adm_id] = code_list
    print(f'removed admission_procedures records in ICD-9-PCS, size from {len(admission_procedures)} to {len(admission_procedures_new)}')
    admission_procedures = admission_procedures_new

    # Gather all round information for an admission
    # Find a set of admission ids that has information in all the extracted information tables
    adm_list_all = list(admission_metadata.keys())
    print(f'there are total {len(adm_list_all)} admissions')
    adm_list_1 = list(admission_codes.keys())
    joint_1 = list(set(adm_list_all) & set(adm_list_1))
    print(f'1 admission_codes: \t{len(joint_1)}/{len(adm_list_1)}')
    adm_list_2 = list(admission_labevents.keys())
    joint_2 = list(set(adm_list_all) & set(adm_list_2))
    print(f'2 admission_labevents: \t{len(joint_2)}/{len(adm_list_2)}')
    adm_list_3 = list(admission_prescriptions.keys())
    joint_3 = list(set(adm_list_all) & set(adm_list_3))
    print(f'3 admission_prescriptions: \t{len(joint_3)}/{len(adm_list_3)}')
    adm_list_4 = list(admission_procedures.keys())
    joint_4 = list(set(adm_list_all) & set(adm_list_4))
    print(f'4 admission_procedures: \t{len(joint_4)}/{len(adm_list_4)}')
    adm_list_5 = list(admission_notes.keys())
    joint_5 = list(set(adm_list_all) & set(adm_list_5))
    print(f'5 admission_notes: \t{len(joint_5)}/{len(adm_list_5)}')
    adm_list_6 = list(admission_radiology_notes.keys())
    joint_6 = list(set(adm_list_all) & set(adm_list_6))
    print(f'6 admission_radiology_notes: \t{len(joint_6)}/{len(adm_list_6)}')
    # Option 1: only keep the admissions that has info for all aspects
    # admission_list = list(set(joint_1) & set(joint_2) & set(joint_3) & set(joint_4) & set(joint_5) & set(joint_6))
    # Option 2: keep the admissions that has discharge notes, other aspects can be empty
    admission_list = joint_5
    print(f'we keep \t{len(admission_list)}/{len(adm_list_all)} admissions only')
    
    verbose_example = False
    adm_data = []
    for i, adm_id in enumerate(tqdm(admission_list)):
        # Get metadata for the patient
        this_metadata = admission_metadata[adm_id]
        this_patient = this_metadata[0]
        if verbose_example:
            print(f'adm id: {adm_id}, patient id: {this_patient}')
        this_patient_metadata = patient_metadata[this_patient]
        admittime = this_metadata[2]
        admission_type = this_metadata[3]
        admission_location = this_metadata[4]
        discharge_location = this_metadata[5]
        patient_insurance = this_metadata[6]
        patient_lang = this_metadata[7]
        patient_marital = this_metadata[8]
        patient_race = this_metadata[9]
        hospital_expire_flag = this_metadata[10]
        patient_gender = this_patient_metadata[1]
        patient_age = this_patient_metadata[2]
        if verbose_example:
            print(f'patient language: {patient_lang}, marital: {patient_marital}, race: {patient_race}, gender: {patient_gender}, age: {patient_age}')

        # Get metadata for service department
        this_services = admission_services[adm_id] if adm_id in admission_services else []
        this_transfers = admission_transfers[adm_id] if adm_id in admission_transfers else []
        if verbose_example:
            print(f'services: {this_services}')
            print(f'transfers: {this_transfers}')

        # Get input information
        # input 1: medical notes
        this_notes = admission_notes[adm_id] if adm_id in admission_notes else []
        # input 2: radiology notes
        this_radiology_notes = admission_radiology_notes[adm_id] if adm_id in admission_radiology_notes else []
        # input 3: lab events
        this_labevents = admission_labevents[adm_id] if adm_id in admission_labevents else []
        this_microbiologyevents = admission_microbiologyevents[adm_id] if adm_id in admission_microbiologyevents else []
        this_labevents_clean = []
        for labevent in this_labevents:
            itemid = str(labevent[2])
            value = labevent[6]
            value_unit = labevent[7]
            ref_range_lower = labevent[8]
            ref_range_upper = labevent[9]
            flag = labevent[10]
            if itemid in labitem_labels:
                itemdesc = labitem_labels[itemid]
                if not math.isnan(value):
                    range_msg = ''
                    if (not isinstance(ref_range_lower, str)) and not (isinstance(ref_range_upper, str)):
                        if not math.isnan(ref_range_lower) and not math.isnan(ref_range_upper):
                            range_msg = f'normal range is {ref_range_lower}-{ref_range_upper}'
                    unit_msg = ''
                    if type(value_unit) == str and value_unit.strip() != '' and value_unit.strip() != 'nan':
                        unit_msg = f" {value_unit}"
                    if isinstance(flag, str) and flag != '':
                        if range_msg != '':
                            msg = f'{itemdesc}: {value}{unit_msg}, {range_msg}, flagged as {flag}'
                        else:
                            msg = f'{itemdesc}: {value}{unit_msg}, flagged as {flag}'
                    else:
                        if range_msg != '':
                            msg = f'{itemdesc}: {value}{unit_msg}, {range_msg}'
                        else:
                            msg = f'{itemdesc}: {value}{unit_msg}'
                    this_labevents_clean.append(msg)
        if verbose_example:
            print(this_notes)
            print(this_radiology_notes)
            print(this_labevents_clean)
            print(this_microbiologyevents)

        # Get target ground-truth
        # target 1: diagnosis codes of this admission
        this_diagnoses = admission_codes[adm_id] if adm_id in admission_codes else []
        # target 2: prescriptions of this admission
        this_prescriptions = admission_prescriptions[adm_id] if adm_id in admission_prescriptions else []
        # target 3: procedure codes of this admission
        this_procedures = admission_procedures[adm_id] if adm_id in admission_procedures else []
        if verbose_example:
            print(this_diagnoses)
            print(this_prescriptions)
            print(this_procedures)

        adm_data_this = {
            'hadm_id': adm_id,
            'admittime': admittime,
            'admission_type': admission_type,
            'hospital_expire_flag': hospital_expire_flag,
            'admission_location': admission_location,
            'discharge_location': discharge_location,
            'patient_insurance': patient_insurance,
            'patient_lang': patient_lang,
            'patient_marital': patient_marital,
            'patient_race': patient_race,
            'patient_gender': patient_gender,
            'patient_age': patient_age,
            'services': this_services,
            'transfers': this_transfers,
            'notes_discharge': this_notes,
            'notes_radiology': this_radiology_notes,
            'labevents': this_labevents,
            '_labevents_verbalized': this_labevents_clean,
            'microbiologyevents': this_microbiologyevents,
            'target_diagnoses': this_diagnoses,
            'target_prescriptions': this_prescriptions,
            'target_procedures': this_procedures,
            '_service': [note[-1] for note in this_notes]
        }
        adm_data.append(adm_data_this)

    # Timestamp object is not json serializable
    # with open(os.path.join(args.save_path_parsed, 'adm_data_100.json'), 'w') as f:
    #     json.dump(adm_data[:100], f, indent=4)

    with open(os.path.join(args.save_path_parsed, 'adm_data.pkl'), 'wb') as f:
        pickle.dump(adm_data, f)
