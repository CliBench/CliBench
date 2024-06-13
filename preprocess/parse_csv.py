import os
from datetime import datetime
from collections import OrderedDict

import pandas
import pandas as pd
import numpy as np

GLOBAL_NROWS = None

class EHRParser:
    pid_col = 'pid'
    adm_id_col = 'adm_id'
    adm_time_col = 'adm_time'
    cid_col = 'cid'

    def __init__(self, path, icd10to9=False):
        self.path = path
        self.icd10to9 = icd10to9
        self.icd9to10 = False # Not implemented

        self.skip_pid_check = False

        self.patient_admission = None
        self.admission_codes = None
        self.admission_procedures = None
        self.admission_medications = None
        pd.set_option('display.max_colwidth', None)

        self.parse_fn = {
            'admission': self.set_admission,
            'diagnosis': self.set_diagnosis,
            'patient': self.set_patient,
            'labevent': self.set_labevent,
            'prescription': self.set_prescription,
            'procedure': self.set_procedure,
            'microbiologyevents': self.set_microbiologyevents,
            'services': self.set_services,
            'transfers': self.set_transfers,
            'discharge': self.set_discharge,
            'radiology': self.set_radiology,
        }
        self.after_read_fn = {
            'admission': self._after_read_admission,
            'diagnosis': self._after_read_concepts,
            'procedure': self._after_read_concepts,
        }
        self.post_process_fn = {
            'admission': self.post_process_admission,
            'diagnosis': self.post_process_diagnosis,
            'patient':   self.post_process_patient,
            'labevent': self.post_process_labevent,
            'prescription': self.post_process_prescription,
            'procedure': self.post_process_procedure,
            'microbiologyevents': self.post_process_microbiologyevents,
            'services': self.post_process_services,
            'transfers': self.post_process_transfers,
            'discharge': self.post_process_discharge,
            'radiology': self.post_process_radiology,
        }

    def set_admission(self):
        raise NotImplementedError

    def set_diagnosis(self):
        raise NotImplementedError

    def set_patient(self):
        raise NotImplementedError
    
    def set_labevent(self):
        raise NotImplementedError
    
    def set_prescription(self):
        raise NotImplementedError
    
    def set_procedure(self):
        raise NotImplementedError
    
    def set_microbiologyevents(self):
        raise NotImplementedError
    
    def set_services(self):
        raise NotImplementedError
    
    def set_transfers(self):
        raise NotImplementedError
    
    def set_discharge(self):
        raise NotImplementedError
    
    def set_radiology(self):
        raise NotImplementedError
    
    def post_process_admission(self, concepts, cols, tasks=[]):
        patient_admission = OrderedDict()
        if 'patient_admission' in tasks:
            print('task 1: create mapping from patient id to admission list sorted by time')
            all_patients = OrderedDict()
            for i, row in concepts.iterrows():
                if i % 100 == 0:
                    print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
                pid, adm_id, adm_time = row[cols[self.pid_col]], row[cols[self.adm_id_col]], row[cols[self.adm_time_col]]
                if pid not in all_patients:
                    all_patients[pid] = []
                admission = all_patients[pid]
                admission.append({self.adm_id_col: adm_id, self.adm_time_col: adm_time})
            print('\r\t%d in %d rows' % (len(concepts), len(concepts)))

            for pid, admissions in all_patients.items():
                if len(admissions) >= 2:
                    patient_admission[pid] = sorted(admissions, key=lambda admission: admission[self.adm_time_col])
        self.patient_admission = patient_admission

        all_admissions = OrderedDict()
        if 'admission_metadata' in tasks:
            print('task 2: create mapping from admission id to metadata of this admission')
            for i, row in concepts.iterrows():
                if i % 100 == 0:
                    print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
                all_admissions[row[cols[self.adm_id_col]]] = [row[col] for col in list(cols.values())]
        self.admission_metadata = all_admissions

    def post_process_diagnosis(self, concepts, cols, tasks=[]):
        result = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:
                adm_id, code, code_version, seq_num = row[cols[self.adm_id_col]], row[cols[self.cid_col]], row[cols[self.icd_ver_col]], row[cols[self.seq_num]]
                if code == '':
                    continue
                if adm_id not in result:
                    result[adm_id] = []
                result[adm_id].append([code, code_version, seq_num])
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        self.admission_codes = result

    def post_process_procedure(self, concepts, cols, tasks=[]):
        result = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:
                adm_id, code, code_version, seq_num, chartdate = row[cols[self.adm_id_col]], row[cols[self.cid_col]], row[cols[self.icd_ver_col]], row[cols[self.seq_num]], row[cols['chartdate']]
                if code == '':
                    continue
                if adm_id not in result:
                    result[adm_id] = []
                result[adm_id].append([code, code_version, seq_num, chartdate])
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        self.admission_procedures = result

    def post_process_patient(self, concepts, cols, tasks=[]):
        all_patients = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            all_patients[row[cols[self.pid_col]]] = [row[col] for col in list(cols.values())]
        self.patient_metadata = all_patients

    def post_process_labevent(self, concepts, cols, tasks=[]):
        all_admissions = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            adm_id = int(row[cols[self.adm_id_col]])
            if adm_id not in all_admissions:
                all_admissions[adm_id] = []
            all_admissions[adm_id].append([row[col] for col in list(cols.values())])
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        self.admission_labevents = all_admissions

    def post_process_microbiologyevents(self, concepts, cols, tasks=[]):
        all_admissions = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            adm_id = int(row[cols[self.adm_id_col]])
            if adm_id not in all_admissions:
                all_admissions[adm_id] = []
            all_admissions[adm_id].append([row[col] for col in list(cols.values())])
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        self.admission_microbiologyevents = all_admissions

    def post_process_prescription(self, concepts, cols, tasks=[]):
        result = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            adm_id = row[cols[self.adm_id_col]]
            if adm_id not in result:
                result[adm_id] = []
            result[adm_id].append([row[col] for col in list(cols.values())])
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        self.admission_prescriptions = result

    def post_process_services(self, concepts, cols, tasks=[]):
        all_admissions = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            adm_id = int(row[cols[self.adm_id_col]])
            if adm_id not in all_admissions:
                all_admissions[adm_id] = []
            all_admissions[adm_id].append([row[col] for col in list(cols.values())])
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        self.admission_services = all_admissions

    def post_process_transfers(self, concepts, cols, tasks=[]):
        all_admissions = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            adm_id = int(row[cols[self.adm_id_col]])
            if adm_id not in all_admissions:
                all_admissions[adm_id] = []
            all_admissions[adm_id].append([row[col] for col in list(cols.values())])
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        self.admission_transfers = all_admissions
    
    def post_process_discharge(self, concepts, cols, tasks=[]):
        result = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:
                adm_id = row[cols[self.adm_id_col]]
                note_id = row[cols['note_id']]
                note_type = row[cols['note_type']]
                note_seq = row[cols['note_seq']]
                text = row[cols['text']]
                if text == '':
                    continue
                if adm_id not in result:
                    result[adm_id] = []
                result[adm_id].append([text, note_id, note_seq])
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))

        for adm_id, note_list in result.items():
            # sort by the last item, which is note_seq
            result[adm_id] = sorted(note_list, key=lambda notes: notes[-1])
        self.admission_notes = result

    def post_process_radiology(self, concepts, cols, tasks=[]):
        result = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:
                adm_id = int(row[cols[self.adm_id_col]])
                note_id = row[cols['note_id']]
                note_type = row[cols['note_type']]
                note_seq = row[cols['note_seq']]
                text = row[cols['text']]
                if text == '':
                    continue
                if adm_id not in result:
                    result[adm_id] = []
                result[adm_id].append([text, note_id, note_seq])
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))

        for adm_id, note_list in result.items():
            # sort by the last item, which is note_seq
            result[adm_id] = sorted(note_list, key=lambda notes: notes[-1])
        self.admission_radiology_notes = result

    @staticmethod
    def to_standard_icd9(code: str):
        raise NotImplementedError

    def _parse_concept(self, concept_type, tasks=[], after_read=False):
        assert concept_type in self.parse_fn.keys()
        print(f'reading concept {concept_type}')
        filename, cols, converters = self.parse_fn[concept_type]()
        concepts = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters, nrows=GLOBAL_NROWS)
        print(f'parsing concept {concept_type}')
        if concept_type in ['labevent', 'radiology', 'labitems', 'labevent', 'prescription', 'procedure', 'microbiologyevents', 'services', 'transfers']:
            print(f'remove rows that have no {self.adm_id_col}')
            print(f'original size: {concepts.size}')
            mask = concepts[cols[self.adm_id_col]] == ''
            concepts = concepts[~mask]
            print(f'left size: {concepts.size}')
        # print(concepts.head(n=1))
        if after_read:
            concepts = self.after_read_fn[concept_type](concepts, concept_type, cols)
        self.post_process_fn[concept_type](concepts, cols, tasks=tasks)

    def _after_read_concepts(self, concepts, concept_type, cols):
        return concepts

    def _after_read_admission(self, admissions, concept_type, cols):
        return admissions

    def calibrate_patient_by_admission(self):
        print('calibrating patients by admission ...')
        del_pids = []
        for pid, admissions in self.patient_admission.items():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id not in self.admission_codes:
                    break
            else:
                continue
            del_pids.append(pid)
        for pid in del_pids:
            admissions = self.patient_admission[pid]
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                for concepts in [self.admission_codes]:
                    if adm_id in concepts:
                        del concepts[adm_id]
            del self.patient_admission[pid]

    def calibrate_admission_by_patient(self):
        print('calibrating admission by patients ...')
        adm_id_set = set()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id_set.add(admission[self.adm_id_col])
        del_adm_ids = [adm_id for adm_id in self.admission_codes if adm_id not in adm_id_set]
        for adm_id in del_adm_ids:
            del self.admission_codes[adm_id]

    def sample_patients(self, sample_num, seed):
        np.random.seed(seed)
        keys = list(self.patient_admission.keys())
        selected_pids = np.random.choice(keys, sample_num, False)
        self.patient_admission = {pid: self.patient_admission[pid] for pid in selected_pids}
        admission_codes = dict()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                admission_codes[adm_id] = self.admission_codes[adm_id]
        self.admission_codes = admission_codes

    def parse(self, sample_num=None, seed=6669):
        # Prepare data for clinical diagnosis prediction from longitudinal history diagnosis data
        # Create self.patient_admission dict mapping from patient to admission list sorted by time
        self._parse_concept('admission', tasks=['patient_admission'], after_read=True)
        # Create self.admission_codes dict mapping from admission id to diagnosis code list
        self._parse_concept('diagnosis', after_read=True)
        self.calibrate_patient_by_admission()
        self.calibrate_admission_by_patient()
        if sample_num is not None:
            self.sample_patients(sample_num, seed)
        return_dict = {
            'patient_admission': self.patient_admission,
            'admission_codes': self.admission_codes,
        }
        return return_dict
    
    def parse_diag(self, seed=6669):
        # Prepare data for clinical diagnosis decision from current admission multi-modal data
        # Create self.patient_admission dict mapping from patient to admission list sorted by time
        self._parse_concept('admission', tasks=['admission_metadata'])
        # Create self.admission_codes dict mapping from admission id to diagnosis code list
        self._parse_concept('diagnosis', after_read=False)
        # Create self.patient_metadata dict mapping from patient id to their metadata
        self._parse_concept('patient')
        # Create self.admission_labevents dict mapping from admission id to a list of lab events
        self._parse_concept('labevent')
        # Create self.admission_prescriptions dict mapping from admission id to a list of prescriptions
        self._parse_concept('prescription')
        # Create self.admission_procedures dict mapping from admission id to a list of procedure code
        self._parse_concept('procedure', after_read=False)
        # Create self.admission_microbiologyevents dict mapping from admission id to a list of microbiology events
        self._parse_concept('microbiologyevents')
        # Create self.admission_services dict mapping from admission id to a list of services
        self._parse_concept('services')
        # Create self.admission_transfers dict mapping from admission id to a list of transfer events
        self._parse_concept('transfers')
        return_dict = {
            'admission_metadata': self.admission_metadata,
            'admission_codes': self.admission_codes,
            'patient_metadata': self.patient_metadata,
            'admission_labevents': self.admission_labevents,
            'admission_prescriptions': self.admission_prescriptions,
            'admission_procedures': self.admission_procedures,
            'admission_microbiologyevents': self.admission_microbiologyevents,
            'admission_services': self.admission_services,
            'admission_transfers': self.admission_transfers,
        }
        return return_dict


class Mimic3Parser(EHRParser):
    def set_admission(self):
        filename = 'ADMISSIONS.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.adm_time_col: 'ADMITTIME'}
        converter = {
            'SUBJECT_ID': int,
            'HADM_ID': int,
            'ADMITTIME': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'DIAGNOSES_ICD.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.cid_col: 'ICD9_CODE'}
        converter = {'SUBJECT_ID': int, 'HADM_ID': int, 'ICD9_CODE': Mimic3Parser.to_standard_icd9}
        return filename, cols, converter

    @staticmethod
    def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code
    
    @staticmethod
    def to_standard_icd10(code: str):
        icd10_code = str(code)
        return icd10_code


class Mimic4Parser(EHRParser):
    def __init__(self, path, icd10to9=False):
        super().__init__(path, icd10to9)
        self.icd_ver_col = 'icd_version'
        self.seq_num = 'seq_num'
        self.skip_pid_check = True
        if self.icd10to9 or self.icd9to10:
            self.icd_map, self.icd_map_9to10 = self._load_icd_map()
        self.patient_year_map = self._load_patient()

    def _load_icd_map(self):
        print('loading ICD-10 to ICD-9 map ...')
        filename = 'icd10-icd9.csv'
        cols = ['ICD10', 'ICD9']
        converters = {'ICD10': str, 'ICD9': str}
        icd_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        icd_map_10to9 = {row['ICD10']: row['ICD9'] for _, row in icd_csv.iterrows()}
        icd_map_9to10 = {row['ICD9']: row['ICD10'] for _, row in icd_csv.iterrows()}
        return icd_map_10to9, icd_map_9to10

    def _load_patient(self):
        print('loading patients anchor year ...')
        filename = 'patients.csv'
        cols = ['subject_id', 'anchor_year', 'anchor_year_group']
        converters = {'subject_id': int, 'anchor_year': int, 'anchor_year_group': lambda cell: int(str(cell)[:4])}
        patient_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        patient_year_map = {row['subject_id']: row['anchor_year'] - row['anchor_year_group']
                            for i, row in patient_csv.iterrows()}
        return patient_year_map

    def set_admission(self):
        filename = 'admissions.csv'
        cols = {self.pid_col: 'subject_id', self.adm_id_col: 'hadm_id', self.adm_time_col: 'admittime'}
        converter = {
            'subject_id': int,
            'hadm_id': int,
            'admittime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            # 'dischtime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            # 'deathtime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'admission_type': str,
            # 'admit_provider_id': str,
            'admission_location': str,
            'discharge_location': str,
            'insurance': str,
            'language': str,
            'marital_status': str,
            'race': str,
            # 'edregtime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            # 'edouttime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'hospital_expire_flag': int,
        }
        for converter_key in converter.keys():
            if converter_key not in list(cols.values()):
                cols[converter_key] = converter_key
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnoses_icd.csv'
        cols = {
            self.pid_col: 'subject_id',
            self.adm_id_col: 'hadm_id',
            self.seq_num: 'seq_num',
            self.cid_col: 'icd_code',
            self.icd_ver_col: 'icd_version'
        }
        converter = {'subject_id': int, 'hadm_id': int, 'seq_num': int, 'icd_code': str, 'icd_version': int}
        return filename, cols, converter
    
    def set_procedure(self):
        filename = 'procedures_icd.csv'
        cols = {
            self.pid_col: 'subject_id',
            self.adm_id_col: 'hadm_id',
            'chartdate': 'chartdate',
            self.seq_num: 'seq_num',
            self.cid_col: 'icd_code',
            self.icd_ver_col: 'icd_version'
        }
        converter = {
            'subject_id': int, 
            'hadm_id': int, 
            'chartdate': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d') if str(cell) != "" else None,
            'seq_num': int, 
            'icd_code': str, 
            'icd_version': int
        }
        return filename, cols, converter

    def set_patient(self):
        filename = 'patients.csv'
        cols = {
            self.pid_col: 'subject_id',
            'gender': 'gender',
            'anchor_age': 'anchor_age',
            # 'anchor_year': 'anchor_year',
            # 'anchor_year_group': 'anchor_year_group',
            # 'dod': 'dod',
        }
        converter = {
            'subject_id': int, 
            'gender': str,
            'anchor_age': int,
            # 'anchor_year': int,
            # 'anchor_year_group': str,
            # 'dod': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
        }
        return filename, cols, converter

    def set_labevent(self):
        filename = 'labevents.csv'
        cols = {
            'labevent_id': 'labevent_id',
            # self.pid_col: 'subject_id',
            self.adm_id_col: 'hadm_id',
            # 'specimen_id': 'specimen_id',
            'itemid': 'itemid',
            # 'order_provider_id': 'order_provider_id',
            'charttime': 'charttime', # specimen was acquired
            'storetime': 'storetime', # measurement was made available
            'value': 'value',
            'valuenum': 'valuenum',
            'valueuom': 'valueuom',
            'ref_range_lower': 'ref_range_lower',
            'ref_range_upper': 'ref_range_upper',
            'flag': 'flag',
            'priority': 'priority', # The priority of the laboratory measurement: either routine or stat (urgent).
            'comments': 'comments',
        }
        converter = {
            'labevent_id': int,
            # 'subject_id': int,
            'hadm_id': str,
            # 'specimen_id': int,
            'itemid': int,
            # 'order_provider_id': str,
            'charttime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'storetime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'value': str,
            'valuenum': float,
            'valueuom': str,
            'ref_range_lower': float,
            'ref_range_upper': float,
            'flag': str,
            'priority': str,
            'comments': str,
        }
        converter = {
            'labevent_id': int,
            'hadm_id': str,
            'itemid': int,
            'charttime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'storetime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
        }
        return filename, cols, converter

    def set_microbiologyevents(self):
        filename = 'microbiologyevents.csv'
        cols = {
            self.adm_id_col: 'hadm_id',
            'charttime': 'charttime', # specimen was acquired
            'spec_itemid': 'spec_itemid',
            'spec_type_desc': 'spec_type_desc', # The specimen is a sample derived from a patient; e.g. blood, urine, sputum, etc.
            'storetime': 'storetime', # when the microbiology result was available. the times here are the time of the last known update
            'test_itemid': 'test_itemid',
            'test_name': 'test_name',
            'dilution_text': 'dilution_text',
            'interpretation': 'interpretation', # interpretation of the antibiotic sensitivity, and indicates the results of the test. “S” is sensitive, “R” is resistant, “I” is intermediate, and “P” is pending
            'comments': 'comments',
        }
        converter = {
            'hadm_id': str,
            'charttime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'spec_itemid': str,
            'spec_type_desc': str,
            'storetime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'test_itemid': str,
            'test_name': str,
            'dilution_text': str,
            'interpretation': str,
            'comments': str,
        }
        return filename, cols, converter

    def set_prescription(self):
        filename = 'prescriptions.csv'
        cols = {
            # self.pid_col: 'subject_id',
            self.adm_id_col: 'hadm_id',
            'starttime': 'starttime',
            'stoptime': 'stoptime',
            'drug_type': 'drug_type',
            'drug': 'drug', # A free-text description of the medication administered
            'gsn': 'gsn', # The Generic Sequence Number (GSN), a coded identifier used for medications
            'ndc': 'ndc', # The National Drug Code (NDC), a coded identifier which uniquely identifiers medications.
            'prod_strength': 'prod_strength',
        }
        converter = {
            'subject_id': int,
            'hadm_id': int,
            'starttime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'stoptime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'drug_type': str,
            'drug': str,
            'gsn': str,
            'ndc': str,
            'prod_strength': str,
        }
        return filename, cols, converter

    def set_services(self):
        filename = 'services.csv'
        cols = {
            self.adm_id_col: 'hadm_id',
            'transfertime': 'transfertime', # patient move from prev_service to curr_service
            'prev_service': 'prev_service', # previous service that the patient resides under
            'curr_service': 'curr_service', # current service that the patient resides under
        }
        converter = {
            'hadm_id': int, 
            'transfertime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'prev_service': str, 
            'curr_service': str, 
        }
        return filename, cols, converter
    
    def set_transfers(self):
        filename = 'transfers.csv'
        cols = {
            self.adm_id_col: 'hadm_id',
            'transfer_id': 'transfer_id',
            'eventtype': 'eventtype', # transfer event, ‘ed’ for an emergency department stay, ‘admit’ for an admission to the hospital, ‘transfer’ for an intra-hospital transfer and ‘discharge’ for a discharge from the hospital
            'careunit': 'careunit', # type of unit or ward in which the patient is physically located. Examples of care units include medical ICUs, surgical ICUs, medical wards, new baby nurseries, and so on
            'intime': 'intime', # the time the patient was transferred into the current careunit
            'outtime': 'outtime', # the time the patient was transferred out of the current careunit
        }
        converter = {
            'hadm_id': str, 
            'transfer_id': int,
            'eventtype': str,
            'careunit': str,
            'intime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'outtime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
        }
        return filename, cols, converter

    def _after_read_admission(self, admissions, concept_type, cols):
        print('\tselecting valid admission ...')
        valid_admissions = []
        n = len(admissions)
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t\t%d in %d rows' % (i + 1, n), end='')
            pid = row[cols[self.pid_col]]
            year = row[cols[self.adm_time_col]].year - self.patient_year_map[pid]
            if year > 2012:
                valid_admissions.append(i)
        print('\r\t\t%d in %d rows' % (n, n))
        print('\t\tremaining %d rows' % len(valid_admissions))
        return admissions.iloc[valid_admissions]

    def _after_read_concepts(self, concepts, concept_type, cols):
        if self.icd10to9:
            print('\tmapping ICD-10 to ICD-9 ...')
        else:
            print('\tno ICD version conversion')
        n = len(concepts)
        if concept_type == 'diagnosis':
            def _10to9(i, row):
                if i % 100 == 0:
                    print('\r\t\t%d in %d rows' % (i + 1, n), end='')
                cid = row[cid_col]
                if row[icd_ver_col] == 10:
                    if cid not in self.icd_map:
                        code = self.icd_map[cid + '1'] if cid + '1' in self.icd_map else ''
                    else:
                        code = self.icd_map[cid]
                    if code == 'NoDx':
                        code = ''
                else:
                    code = cid
                return Mimic4Parser.to_standard_icd9(code)
            
            def _9to10(i, row):
                # TODO: there is still some mapping issues
                if i % 100 == 0:
                    print('\r\t\t%d in %d rows' % (i + 1, n), end='')
                cid = row[cid_col]
                ver_this = row[icd_ver_col]
                if row[icd_ver_col] == 9:
                    ver_this = 10
                    if cid in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid]
                    elif '0' + cid in self.icd_map_9to10:
                        code = self.icd_map_9to10['0' + cid]
                    elif cid[0] == '0' and cid[1:] in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[1:]]
                    elif cid[:-1] in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[:-1]]
                    elif cid[:-1] + '0' in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[:-1] + '0']
                    elif cid[:-1] + '1' in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[:-1] + '1']
                    elif cid[:-1] + '2' in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[:-1] + '2']
                    elif cid[:-1] + '3' in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[:-1] + '3']
                    elif cid[:-1] + '4' in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[:-1] + '4']
                    elif cid[:-1] + '5' in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[:-1] + '5']
                    elif cid[:-1] + '6' in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[:-1] + '6']
                    elif cid[:-1] + '7' in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[:-1] + '7']
                    elif cid[:-1] + '8' in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[:-1] + '8']
                    elif cid[:-1] + '9' in self.icd_map_9to10:
                        code = self.icd_map_9to10[cid[:-1] + '9']
                    else:
                        # print(cid, row[icd_ver_col])
                        code = cid
                        ver_this = 9
                else:
                    code = cid
                if ver_this == 10:
                    code = Mimic4Parser.to_standard_icd10(code)
                return [code, ver_this]

            cid_col, icd_ver_col = cols[self.cid_col], self.icd_ver_col
            if self.icd10to9:
                col = np.array([_10to9(i, row) for i, row in concepts.iterrows()])
                print('\r\t\t%d in %d rows' % (n, n))
                concepts[cid_col] = col
            elif self.icd9to10:
                converted_code_and_ver = [_9to10(i, row) for i, row in concepts.iterrows()]
                print('\r\t\t%d in %d rows' % (n, n))
                col_code = [pair[0] for pair in converted_code_and_ver]
                col_ver = [pair[1] for pair in converted_code_and_ver]
                concepts[cid_col] = col_code
                concepts[icd_ver_col] = col_ver
        return concepts

    @staticmethod
    def to_standard_icd9(code: str):
        return Mimic3Parser.to_standard_icd9(code)
    
    @staticmethod
    def to_standard_icd10(code: str):
        return Mimic3Parser.to_standard_icd10(code)
    

class Mimic4NoteParser(Mimic4Parser):
    def set_discharge(self):
        filename = 'discharge.csv'
        cols = {
            'note_id': 'note_id', 
            self.pid_col: 'subject_id', 
            self.adm_id_col: 'hadm_id', 
            'note_type': 'note_type',
            'note_seq': 'note_seq',
            'charttime': 'charttime', # time the note was charted, amy not be the written time
            'storetime': 'storetime', # time the note was stored in DB, usually when the note was completed and signed
            'text': 'text'
        }
        converter = {
            'note_id': str,
            'subject_id': int,
            'hadm_id': int,
            'note_type': str,
            'note_seq': int,
            'charttime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'storetime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'text': str
        }
        return filename, cols, converter

    def set_radiology(self):
        filename = 'radiology.csv'
        cols = {
            'note_id': 'note_id', 
            self.pid_col: 'subject_id', 
            self.adm_id_col: 'hadm_id', 
            'note_type': 'note_type',
            'note_seq': 'note_seq',
            'charttime': 'charttime', # time the note was charted, amy not be the written time
            'storetime': 'storetime', # time the note was stored in DB, usually when the note was completed and signed
            'text': 'text'
        }
        converter = {
            'note_id': str,
            'subject_id': int,
            'hadm_id': str,
            'note_type': str,
            'note_seq': int,
            'charttime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'storetime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S') if str(cell) != "" else None,
            'text': str
        }
        return filename, cols, converter

    def parse(self, sample_num=None, seed=6669):
        # Prepare data for clinical diagnosis decision from current admission multi-modal data
        # Create self.admission_notes dict mapping from patient id to their discharge medical notes
        self._parse_concept('discharge')
        # Create self.admission_radiology_notes dict mapping from patient id to their radiology medical notes
        self._parse_concept('radiology')
        return_dict = {
            'admission_notes': self.admission_notes,
            'admission_radiology_notes': self.admission_radiology_notes,
        }
        return return_dict


class EICUParser(EHRParser):
    def __init__(self, path, icd10to9):
        super().__init__(path, icd10to9)
        self.skip_pid_check = True

    def set_admission(self):
        filename = 'patient.csv'
        cols = {
            self.pid_col: 'patienthealthsystemstayid',
            self.adm_id_col: 'patientunitstayid',
            self.adm_time_col: 'hospitaladmitoffset'
        }
        converter = {
            'patienthealthsystemstayid': int,
            'patientunitstayid': int,
            'hospitaladmitoffset': lambda cell: -int(cell)
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnosis.csv'
        cols = {self.pid_col: 'diagnosisid', self.adm_id_col: 'patientunitstayid', self.cid_col: 'icd9code'}
        converter = {'diagnosisid': int, 'patientunitstayid': int, 'icd9code': EICUParser.to_standard_icd9}
        return filename, cols, converter

    @staticmethod
    def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        code = code.split(',')[0]
        c = code[0].lower()
        dot = code.find('.')
        if dot == -1:
            dot = None
        if not c.isalpha():
            prefix = code[:dot]
            if len(prefix) < 3:
                code = ('%03d' % int(prefix)) + code[dot:]
            return code
        if c == 'e':
            prefix = code[1:dot]
            if len(prefix) != 3:
                return ''
        if c != 'e' or code[0] != 'v':
            return ''
        return code

    def parse_diagnoses(self):
        super().parse_diagnoses()
        t = OrderedDict.fromkeys(self.admission_codes.keys())
        for adm_id, codes in self.admission_codes.items():
            t[adm_id] = list(set(codes))
        self.admission_codes = t
