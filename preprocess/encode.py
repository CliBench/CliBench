from collections import OrderedDict

from preprocess.parse_csv import EHRParser

def code_to_our_token(code_item):
    """
    code_item: a list of three items such as ['64511', 9, 1], code, version, seq_num
    """
    return f"ICD{code_item[1]}_{code_item[0]}"

def code_token_to_code(code_token):
    return "_".join(code_token.split('_')[1:])

def encode_code(patient_admission, admission_codes):
    code_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            codes = admission_codes[admission[EHRParser.adm_id_col]]
            code_tokens = [code_to_our_token(c) for c in codes]
            for ct in code_tokens:
                if ct not in code_map:
                    code_map[ct] = len(code_map)

    admission_codes_encoded = {
        admission_id: list(set(code_map[code_to_our_token(code)] for code in codes))
        for admission_id, codes in admission_codes.items()
    }
    return admission_codes_encoded, code_map
