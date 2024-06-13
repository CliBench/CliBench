import os, json, random, copy, re, time, pickle, time
import openai
from openai import OpenAI, AzureOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from peft import PeftModel
import torch
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import copy
import gc
from statistics import mean, median

target_task = 'target_diagnoses'
# target_task = 'target_procedures'
# target_task = 'target_laborders'
# target_task = 'target_prescriptions'

split_to_use = 'test'
# split_to_use = 'train'

# mode = 'seq_gen'
mode = 'inference'

# inference_engine = 'hf'
inference_engine = 'vllm'

# ----- GPT family
# llm_name = 'gpt-3.5-turbo'
# llm_name = 'gpt-4-turbo'
# llm_name = 'gpt-4o'
# ----- LLaMA3 family
llm_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# llm_name = 'meta-llama/Meta-Llama-3-8B'
# llm_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
# ----- LLaMA2 family
# llm_name = 'NousResearch/Llama-2-7b-chat-hf'
# llm_name = 'meta-llama/Llama-2-7b-hf'
# llm_name = 'NousResearch/Llama-2-13b-chat-hf' # (secondary)
# llm_name = 'NousResearch/Llama-2-70b-chat-hf' # (secondary)
# ----- Mistral family
# llm_name = 'mistralai/Mistral-7B-Instruct-v0.1'
# llm_name = 'mistralai/Mistral-7B-Instruct-v0.2'
# llm_name = 'mistralai/Mistral-7B-Instruct-v0.3'
# llm_name = 'mistralai/Mistral-7B-v0.3'
# llm_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
# llm_name = 'mistralai/Mixtral-8x22B-Instruct-v0.1'
# ----- Biomed LLM
# llm_name = 'aaditya/Llama3-OpenBioLLM-8B'
# llm_name = 'aaditya/Llama3-OpenBioLLM-70B'
# llm_name = 'BioMistral/BioMistral-7B-DARE'
# llm_name = 'epfl-llm/meditron-7b'
# llm_name = 'epfl-llm/meditron-70b'
# llm_name = 'starmpcc/Asclepius-Llama2-7B'
# ----- Biomed LLM
# llm_name = 'medalpaca/medalpaca-13b' # (secondary)
# llm_name = 'axiong/PMC_LLaMA_13B' # (secondary)

# If loading from a saved model
llm_path = llm_name

lora_path = ''
# lora_path = '/home/ubuntu/derek-240318/clinical-event-pred/alignment-handbook/data/llama3-8b-instruct-dpo-qlora-codes-diagnoses-full/checkpoint-4800'

# Set up GPU number
tensor_parallel_size = 4 if '70b' in llm_name.lower() else 1

save_path_parsed = 'data/mimic4/parsed'
openai_engine = 'openai' # openai, azure
from_saved = True

# Set up batch size
if 'gpt' in llm_name:
    batch_size = 1
    batch_size_big = 1
elif inference_engine == 'hf':
    if '8x7b' in llm_name.lower():
        batch_size = 4
        batch_size_big = 40
    elif 'Mistral-7B' in llm_name:
        batch_size = 1
        batch_size_big = 1
    elif '7b' in llm_name.lower():
        batch_size = 4
        batch_size_big = 40
    else:
        batch_size = 8
        batch_size_big = 80
    batch_size = 1
    batch_size_big = 1
elif inference_engine == 'vllm':
    batch_size_big = 20 if '70b' in llm_name.lower() else 100

if mode == 'seq_gen' and 'Llama' not in llm_name:
    raise ValueError('Only Llama supports seq_gen mode')

save_name = llm_name.split('/')[-1] if '/' in llm_name else llm_name
lora_name = '-'.join(lora_path.split('/')[-2:]) if '/' in lora_path else lora_path
if lora_name != '':
    lora_name = '_' + lora_name
evaldata_save_path = f'data/mimic4/{target_task}/{split_to_use}_{save_name}_evaldata.pkl'
batch_proc = False
char_per_token_est = 2 # -1, 4, 3, 2
token_count_lab = 500
estimated_output_len = 650 # this number is from output length distribution by LLaMA3-8B-Instruct model

# Prepare instructioin and system role for the selected task
prompt_system_role = 'You are a professional clinician in a hospital with expert knowledge in medical and clinical domains.'
if target_task == 'target_diagnoses':
    prompt_task_instruction = 'The task is to make a list of diagnoses for this patient based on the provided information of the patient. '
    prompt_task_instruction += 'The diagnosis can be in ICD-10-CM code format (such as S12.000G), or natural language description of the disease. '
    prompt_task_instruction += 'Separate each diagnosis with a new line. '
    prompt_task_instruction += 'Please provide as many diagnoses as you can until you are not confident about your diagnosis decision. '

    prompt_task_instruction_end = 'What are the diagnoses for this patient?'
elif target_task == 'target_procedures':
    prompt_task_instruction = 'The task is to decide a list of procedures for this patient based on the provided information of the patient. '
    prompt_task_instruction += 'A clinical procedure can be defined as any practice of a health practitioner that involves a combination of special skills or abilities and may require drugs, devices, or both. Clinical procedure is an activity directed at or performed on an individual with the object of improving health, treating disease or injury, or making a diagnosis. '
    prompt_task_instruction += 'The procedure can be in ICD-10-PCS code format (such as 4A023N6), or natural language description of the procedure. '
    prompt_task_instruction += 'Separate each procedure with a new line. '
    prompt_task_instruction += 'Please provide as many procedures as you can until you are not confident about your procedure decision. '

    prompt_task_instruction_end = 'What are the procedures for this patient?'
elif target_task == 'target_laborders':
    prompt_task_instruction = 'The task is to decide a list of lab tests to be done for this patient based on the provided information of the patient to facilitate downstream diagnosis. '
    prompt_task_instruction += 'Lab test is a medical procedure that involves testing a sample of blood, urine, or other substance from the body. Laboratory tests can help determine a diagnosis, plan treatment, check to see if treatment is working, or monitor the disease over time. '
    prompt_task_instruction += 'Please produce natural language name or definition of the lab tests to be ordered. '
    prompt_task_instruction += 'Separate each lab test with a new line. '
    prompt_task_instruction += 'Please provide as many lab tests as you can until you are not confident about your lab test order decision. '

    prompt_task_instruction_end = 'What lab tests need to be ordered for this patient?'
elif target_task == 'target_prescriptions':
    prompt_task_instruction = 'The task is to decide a list of medications to be prescribed for this patient based on the provided information of the patient. '
    prompt_task_instruction += 'Please produce natural language brand names or generic names of the medications. '
    prompt_task_instruction += 'Separate each medication with a new line. '
    prompt_task_instruction += 'Please provide as many prescriptions as you can until you are not confident about your prescription decision. '

    prompt_task_instruction_end = 'What medications need to be prescribed for this patient?'
else:
    assert False, 'Not implemented for other tasks'

with open(os.path.join(save_path_parsed, 'labitem_labels.json')) as f:
    labitem_labels = json.load(f)

def main():
    if 'gpt' in llm_name:
        if openai_engine == 'azure':
            print(f'Using Azure AI {llm_name}')
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01",
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
                timeout=30
            )
            if 'gpt-4' in llm_name:
                azure_deployment_name='gpt4-turbo'
            else:
                azure_deployment_name='medical'
        else:
            print(f'Using Open AI {llm_name}')
            # Prepare Open AI API
            client = OpenAI(timeout=30)
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        # Load model and gen config parameters
        if '70b' not in llm_name.lower():
            model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16, device_map="auto")
            default_generation_config = model.generation_config
            default_model_config = model.config
            default_max_length = default_model_config.max_position_embeddings
            default_max_length_output = max(default_generation_config.max_length, estimated_output_len)
            defaul_temperature = default_generation_config.temperature
            default_top_p = default_generation_config.top_p
        else:
            with open(f'infer_settings/model_config_manual.json', 'r') as f:
                model_config_manual = json.load(f)
                print('Using manually saved model config for 70B model')
            default_generation_config = model_config_manual[llm_path]['generation_config']
            default_model_config = model_config_manual[llm_path]['config']
            default_max_length = default_model_config["max_position_embeddings"]
            default_max_length_output = max(default_generation_config["max_length"], estimated_output_len) if 'max_length' in default_generation_config else estimated_output_len
            defaul_temperature = default_generation_config["temperature"] if 'temperature' in default_generation_config else 1.0
            default_top_p = default_generation_config["top_p"] if 'top_p' in default_generation_config else 1.0
        print(default_max_length, default_max_length_output, defaul_temperature, default_top_p)
        
        if inference_engine == 'hf':
            print(f'Using Huggingface {llm_path}')
            if batch_proc:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                if lora_path == '':
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    pipeline = transformers.pipeline(
                        "text-generation",
                        model=llm_path,
                        return_full_text=False,
                        torch_dtype=torch.float16,
                        device_map="auto",
                    )
                else:
                    model = PeftModel.from_pretrained(
                        model,
                        lora_path,
                        torch_dtype=torch.float16,
                    )
                    pipeline = transformers.pipeline(
                        "text-generation",
                        model=model,
                        return_full_text=False,
                        tokenizer=tokenizer,
                        framework="pt",
                    )
                pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
        elif inference_engine == 'vllm':
            print(f'Using vLLM {llm_path}, default_max_length_output is {default_max_length_output}')
            if '70b' not in llm_name.lower():
                del model
                gc.collect()
                torch.cuda.empty_cache()
            from vllm import LLM, SamplingParams
            sampling_params = SamplingParams(temperature=defaul_temperature, 
                                            top_p=default_top_p,
                                            max_tokens=default_max_length_output)
            if lora_path != '':
                from vllm.lora.request import LoRARequest
                llm_model = LLM(model=llm_path, enable_lora=True)
            else:
                if '70b' in llm_name.lower():
                    print('70 model start loading')
                    llm_model = LLM(
                        llm_path,
                        dtype="float16",
                        enforce_eager=True,
                        gpu_memory_utilization=0.98,
                        tensor_parallel_size=tensor_parallel_size,
                    )
                    print('70 model loaded')
                    # llm_model = LLM(model=llm_path, dtype='float16') 
                    # llm_model = LLM(model=llm_path, quantization="AWQ", dtype='float16')
                else:
                    llm_model = LLM(model=llm_path)

    # Set max token length for each model
    if llm_name in ['gpt-4-turbo', 'gpt-4o']:
        max_token_length = 128000
    elif llm_name == 'gpt-3.5-turbo':
        max_token_length = 16385
    else:
        assert default_max_length <= 65536, f'the max_token_length {default_max_length} is probably wrong, please double check'
        max_token_length = default_max_length - estimated_output_len
        # if 'llama-3' in llm_name.lower():
        #     max_token_length = 8192
        # elif 'llama-2' in llm_name.lower():
        #     max_token_length = 4096
        # elif 'Mistral-7B-Instruct-v0.2' in llm_name:
        #     max_token_length = 32000
        # elif 'Mixtral-8x7B' in llm_name or 'BioMistral' in llm_name:
        #     max_token_length = 32768
        # elif 'Mixtral-8x22B' in llm_name:
        #     max_token_length = 65536
        # elif 'mistral' in llm_name.lower() or 'medalpaca' in llm_name.lower():
        #     max_token_length = 2048
        # elif 'meditron' in llm_name.lower():
        #     max_token_length = 4096
        # elif 'openbiollm' in llm_name.lower():
        #     max_token_length = 8192
        # else:
        #     assert False, 'max_token_length not defined for this model'

    prompt_basic_length_char = len(prompt_system_role) + len(prompt_task_instruction) + len(prompt_task_instruction_end)
    left_char_count = max_token_length * char_per_token_est - prompt_basic_length_char

    def model_specific_prompt(system_prompt, task_instruction, user_message, task_instruction_end):
        if 'gpt' in llm_name.lower():
            return f"{task_instruction}\n{user_message} {task_instruction_end}"
        if 'llama' in llm_name.lower() or 'alpaca' in llm_name.lower():
            # not having <s> at the beginning, since it will be added by tokenizer automatically
            return f"""[INST] <<SYS>>
    { system_prompt }

    { task_instruction }
    <</SYS>>

    { user_message } 
    { task_instruction_end } [/INST]"""
        if 'mistral' in llm_name.lower() or 'mixtral' in llm_name.lower():
            return f"""[INST] { system_prompt }
    { task_instruction }

    { user_message }
    { task_instruction_end } [/INST]"""

    def inference(dps, verbose=False):
        responses = []
        outputs = []
        dps_is_list = True
        if not isinstance(dps, list):
            dps = [dps]
            dps_is_list = False
        if 'gpt' in llm_name:
            for dp in dps:
                response = None
                output = []
                patient_info = dp['input_raw'][0]
                discharge_note = patient_info + dp['input_raw'][1]
                radiology_note = dp['input_raw'][2]
                lab_events = dp['input_raw'][3]

                user_message_ideal = f"{discharge_note}{radiology_note}{lab_events}"
                cut_rate = left_char_count / len(user_message_ideal)

                try_count = 0
                success_gen_flag = False
                while success_gen_flag is False and try_count < 10:
                    try_count += 1
                    try:
                        # first time, use full ideal message
                        if try_count == 1:
                            if cut_rate < 1:
                                discharge_note_cut = discharge_note[:int(len(discharge_note) * cut_rate)]
                                radiology_note_cut = radiology_note[:int(len(radiology_note) * cut_rate)]
                                lab_events_cut = lab_events[:int(len(lab_events) * cut_rate)]
                                user_message_cut = f"{discharge_note_cut}{radiology_note_cut}{lab_events_cut}"
                            else:
                                user_message_cut = user_message_ideal
                        else:
                            # user_message_cut = user_message_ideal[:left_char_count - 200 * (try_count - 1)]
                            cut_rate_this_iter = min(cut_rate, 1) - 0.15 * (try_count - 1)
                            discharge_note_cut = discharge_note[:int(len(discharge_note) * cut_rate_this_iter)]
                            radiology_note_cut = radiology_note[:int(len(radiology_note) * cut_rate_this_iter)]
                            lab_events_cut = lab_events[:int(len(lab_events) * cut_rate_this_iter)]
                            user_message_cut = f"{discharge_note_cut}{radiology_note_cut}{lab_events_cut}"
                            print(f'truncate user message and try again {try_count} with cut rate {cut_rate_this_iter}')
                        input_full = model_specific_prompt(prompt_system_role, prompt_task_instruction, user_message_cut, prompt_task_instruction_end)
                        # if 'gpt-4' in llm_name:
                        #     time.sleep(2)
                        messages_this_call = [
                                                {"role": "system", "content": prompt_system_role},
                                                {"role": "user", "content": input_full},
                                            ]
                        if openai_engine == 'azure':
                            response = client.chat.completions.create(
                                model=azure_deployment_name,
                                messages=messages_this_call
                            )
                        else:
                            response = client.chat.completions.create(
                                model=llm_name,
                                messages=messages_this_call
                            )
                        for choice in response.choices:
                            output.append(choice.message.content)
                        success_gen_flag = True
                    except openai.APIError as e:
                        #Handle API error here, e.g. retry or log
                        print(f"OpenAI API returned an API Error: {e}")
                        success_gen_flag = False
                        pass
                    except openai.APIConnectionError as e:
                        #Handle connection error here
                        print(f"Failed to connect to OpenAI API: {e}")
                        success_gen_flag = False
                        pass
                    except openai.RateLimitError as e:
                        #Handle rate limit error (we recommend using exponential backoff)
                        print(f"OpenAI API request exceeded rate limit: {e}")
                        success_gen_flag = False
                        pass
                    except UserWarning as e:
                        print(f"UserWarning: {e}")
                        if 'Input length of input_ids is' in e:
                            success_gen_flag = False
                        pass
                responses.append(response)
                outputs.append(output)
        elif inference_engine == 'hf':
            sentences = [dp['input'] for dp in dps]
            if batch_proc:
                inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
                # print(inputs['input_ids'].shape)
                output_sequences = model.generate(
                    **inputs,
                    do_sample=True,
                    # top_k=10, # might lead to RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=default_max_length_output,
                )
                outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else:
                if verbose: 
                    print(sentences[0])
                try:
                    outputs_batch = pipeline(
                        sentences,
                        do_sample=True,
                        num_return_sequences=1,
                        batch_size=batch_size,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=default_max_length_output,
                    )
                    for outputs_single in outputs_batch:
                        outputs.append([o['generated_text'] for o in outputs_single])
                except Exception as e:
                    print(f"Failed to process {[dp['hadm_id'] for dp in dps]}: {e}")
                    outputs.append([])
        elif inference_engine == 'vllm':
            sentences = [dp['input'] for dp in dps]
            if verbose:
                print(sentences[0])
            if lora_path == '':
                outputs_batch = llm_model.generate(sentences, sampling_params)
            else:
                outputs_batch = llm_model.generate(
                    sentences, 
                    sampling_params,
                    lora_request=LoRARequest("pref_adapter", 1, lora_path)
                )
            for outputs_single in outputs_batch:
                # output_single would be something like
                # # RequestOutput(request_id=0, prompt='Hello, my name is', prompt_token_ids=[2, 31414, 6, 127, 766, 16], prompt_logprobs=None, outputs=[CompletionOutput(index=0, text=' Joel, my dad is my friend and we are in a relationship. I am', token_ids=[8966, 6, 127, 4252, 16, 127, 1441, 8, 52, 32, 11, 10, 1291, 4, 38, 524], cumulative_logprob=-36.403580874204636, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestMetrics(arrival_time=1715543838.7726061, last_token_time=1715543838.7726061, first_scheduled_time=1715543838.782087, first_token_time=1715543838.8140318, time_in_queue=0.009480953216552734, finished_time=1715543838.854454), lora_request=None)
                outputs.append([o.text for o in outputs_single.outputs])

        if not dps_is_list:
            responses = responses[0]
            outputs = outputs[0]
        return responses, outputs


    if from_saved and mode == 'inference' and os.path.exists(evaldata_save_path):
        evaldata = pickle.load(open(evaldata_save_path, 'rb'))
        print(f'-> Using cached evaldata saved at {evaldata_save_path}')
    else:
        if split_to_use == 'train':
            evaldata = pickle.load(open(f'data/mimic4/{target_task}/{split_to_use}.pkl', 'rb'))
            # sample 1% of the training data
            evaldata = random.sample(evaldata, int(len(evaldata) * 0.1))
            print(f'sampled {len(evaldata)} data points')
        else:
            # with open(f'data/mimic4/{target_task}/{split_to_use}.json', 'r') as f:
            #     evaldata = json.load(f)
            with open(f'data/mimic4/{target_task}/{split_to_use}.pkl', 'rb') as f:
                evaldata = pickle.load(f)

        gendata = []

        cut_count_user_msg_end = 0
        cut_count_user_msg_note = 0
        cut_rates = []
        if 'gpt' not in llm_name.lower():
            # For open-source models, we can cut the input directly here
            prompt_basic = prompt_system_role + prompt_task_instruction + prompt_task_instruction_end
            left_token_count = max_token_length - len(tokenizer(prompt_basic)['input_ids']) - 30
            print(f'left_token_count: {left_token_count}')

        for dp_i, dp in enumerate(tqdm(evaldata, desc=f"{target_task}, {llm_name}, gen data")):
            patient_info = f"Patient information: age is {dp['patient_age']}, gender is {dp['patient_gender']}, race is {dp['patient_race']}, marital status is {dp['patient_marital']}, insurance category is {dp['patient_insurance']}, admission_location is {dp['admission_location']}. "
            
            # include cleaned discharge note for all kinds of tasks
            discharge_note = ''
            if len(dp['notes_discharge']) > 0:
                discharge_note += '\n\nCLINICAL NOTE: \n'
                for note_item in dp['notes_discharge']:
                    discharge_note += note_item[0] + '\n'

            # only include radiology note for the diagnosis task
            radiology_note = ''
            if target_task == 'target_diagnoses' and len(dp['notes_radiology']) > 0:
                radiology_note += '\n\nRADIOLOGY NOTE: \n'
                for note_item in dp['notes_radiology']:
                    radiology_note += note_item[0] + '\n'

            # only include lab results for the diagnosis task
            lab_events = ''
            # if the task is to decide which lab to order, then do not provide lab results in the input sequence
            if target_task == 'target_diagnoses' and len(dp['_labevents_verbalized']) > 0:
                lab_events += '\n\nLAB EVENTS: \n'
                for lab_item in dp['_labevents_verbalized']:
                    lab_events += lab_item + '\n'

            # For GPT models, the full input will be generated dynamically when provide the input
            evaldata[dp_i]['input_raw'] = [
                patient_info,
                discharge_note,
                radiology_note,
                lab_events
            ]

            if 'gpt' not in llm_name.lower():
                # Truncate the input message to fit the model input length
                # For open-source models, we can cut the input directly here
                user_message_ideal = f"{discharge_note}{radiology_note}{lab_events}"
                user_message_ideal_tokens = tokenizer(user_message_ideal)['input_ids'][1:]
                patient_info_tokenized = tokenizer(patient_info)['input_ids'][1:]
                cut_rate = (left_token_count - len(patient_info_tokenized)) / len(user_message_ideal_tokens)
                if cut_rate > 1: cut_rate = 1
                cut_rates.append(cut_rate)
                
                if cut_rate < 1:
                    if len(lab_events) > 0:
                        lab_events_tokenized = tokenizer(lab_events)['input_ids'][1:]
                        # make sure there are at least some lab results presented even the cut rate is very small
                        len_lab_events = max(token_count_lab, int(len(lab_events_tokenized) * cut_rate))
                    else:
                        lab_events_tokenized = []
                        len_lab_events = 0
                    discharge_note_tokenized = tokenizer(discharge_note)['input_ids'][1:]
                    radiology_note_tokenized = tokenizer(radiology_note)['input_ids'][1:]
                    cut_rate_note = (left_token_count - len_lab_events) / (len(discharge_note_tokenized) + len(radiology_note_tokenized))
                    len_discharge_note = int(len(discharge_note_tokenized) * cut_rate_note)
                    len_radiology_note = int(len(radiology_note_tokenized) * cut_rate_note)

                    # truncated segments, starting from 1 to skip the <s> starting token
                    discharge_note_cut = tokenizer.decode(discharge_note_tokenized[:len_discharge_note])
                    radiology_note_cut = tokenizer.decode(radiology_note_tokenized[:len_radiology_note])
                    lab_events_cut = tokenizer.decode(lab_events_tokenized[:len_lab_events])

                    user_message_cut = f"{patient_info}{discharge_note_cut}{radiology_note_cut}{lab_events_cut}"

                    if cut_rate_note < 1:
                        cut_count_user_msg_note += 1
                    else:
                        cut_count_user_msg_end += 1
                else:
                    user_message_cut = f"{patient_info}{user_message_ideal}"
                
                # Option 1: use our function to fill in prompt template -> not up to date
                # evaldata[dp_i]['input'] = model_specific_prompt(prompt_system_role, prompt_task_instruction, user_message_cut, prompt_task_instruction_end)
                # Option 2: use huggingface chat template
                if 'mistral' in llm_name.lower():
                    messages_this = [
                        {"role": "user", "content": prompt_system_role + '\n' + prompt_task_instruction + '\n\n' + user_message_cut + '\n' + prompt_task_instruction_end}
                    ]
                else:
                    messages_this = [
                        {"role": "system", "content": prompt_system_role + '\n' + prompt_task_instruction},
                        {"role": "user", "content": user_message_cut + '\n' + prompt_task_instruction_end}
                    ]
                evaldata[dp_i]['input'] = tokenizer.apply_chat_template(messages_this, tokenize=False, add_generation_prompt=True) 

                # Prepare target sequence used for seq2seq model training
                if target_task == 'target_diagnoses':
                    gt_codes = [item[0] for item in dp['target_diagnoses']]
                    evaldata[dp_i]['target_gold'] = "\n".join(gt_codes)
                elif target_task == 'target_procedures':
                    gt_codes = [item[0] for item in dp['target_procedures']]
                    evaldata[dp_i]['target_gold'] = "\n".join(gt_codes)
                elif target_task == 'target_laborders':
                    # if use lab item names, but this is outdated implementation
                    # gt_names = [item.split(":")[0] for item in dp['_labevents_verbalized']]
                    # evaldata[dp_i]['target_gold'] = ". ".join(gt_names)
                    # if use lab item names
                    gt_names = [labitem_labels[str(item[0])] for item in dp['target_laborders']]
                    evaldata[dp_i]['target_gold'] = "\n".join(gt_names)
                    # if use lab LOINC code
                    # gt_codes = [item[0] for item in dp['target_laborders']]
                    # evaldata[dp_i]['target_gold'] = "\n".join(gt_codes)
                elif target_task == 'target_prescriptions':
                    gt_names = [item[4] for item in dp['target_prescriptions']]
                    evaldata[dp_i]['target_gold'] = "\n".join(gt_names)
                else:
                    assert False, 'Not implemented for other tasks'

                gen_dp = {
                    'id': dp['hadm_id'],
                    'text': evaldata[dp_i]['input'] + evaldata[dp_i]['target_gold'],
                    'input': evaldata[dp_i]['input'],
                    'target_gold': evaldata[dp_i]['target_gold'],
                    'eval_gold': evaldata[dp_i]['target_gold'],
                }
                gendata.append(gen_dp)

        if 'gpt' not in llm_name.lower():
            print(f'{cut_count_user_msg_note}/{len(evaldata)} input sequence cut discharge + radiology note')
            print(f'{cut_count_user_msg_end}/{len(evaldata)} input sequence cut the end of patient record')
            print(f'cut rates mean: {mean(cut_rates)}, medium: {median(cut_rates)}, min: {min(cut_rates)}, max: {max(cut_rates)}')
            # Statistics of input length and target_gold of the gen dataset
            lengths_i = [len(tokenizer(dp['input'])['input_ids'][1:]) for dp in gendata]
            lengths_o = [len(tokenizer(dp['target_gold'])['input_ids'][1:]) for dp in gendata]
            print('Statistics of number of tokens in input and target_gold of the gen dataset:')
            for list_this in [lengths_i, lengths_o]:
                print(f"Mean: {mean(list_this)}")
                print(f"Median: {median(list_this)}")
                print(f"Min: {min(list_this)}")
                print(f"Max: {max(list_this)}")
            if mode == 'seq_gen':
                with open(f'data/mimic4/{target_task}/{split_to_use}_gen.json', 'w', encoding='utf-8') as f:
                    json.dump(gendata, f, indent=4)
                print('-> gen io saved.')

        pickle.dump(evaldata, open(evaldata_save_path, 'wb'))

    if mode == 'inference':
        results = copy.deepcopy(evaldata)
        if os.path.exists(f'data/mimic4/{target_task}_output/{save_name}{lora_name}.json'):
            with open(f'data/mimic4/{target_task}_output/{save_name}{lora_name}.json', 'r') as f:
                results_2 = json.load(f)
            # generation for this dp has been done, skip
            # generated content is long enough, otherwise not skip
            results_2 = [dp for dp in results_2 if len(dp['output']) > 0 and len(dp['output'][0]) > 100]
            ids_done = [result['hadm_id'] for result in results_2]
        else:
            results_2 = []
            ids_done = []
        print(f'Produced output: {len(ids_done)}/{len(evaldata)}')

        # skip inference for the data instances that already have generated output
        results = [dp for dp in results if dp['hadm_id'] not in ids_done]

        if not os.path.exists(f'data/mimic4/{target_task}_output'):
            os.makedirs(f'data/mimic4/{target_task}_output')

        global_verbose_flag = True
        runtime_sum_s = time.time()
        runtime_list = []
        metadata_dict = {
            'runtime_each': []
        }
        for batch_i in tqdm(range(len(results) // batch_size_big + 1), desc=f"{target_task}, {llm_name}, inference"):
            input_dps = results[batch_i * batch_size_big: (batch_i + 1) * batch_size_big]
            runtime_batch_s = time.time()
            responses, outputs = inference(input_dps, verbose=global_verbose_flag)
            runtime_batch_e = time.time()
            runtime_this_batch = runtime_batch_e - runtime_batch_s
            for in_batch_i, result in enumerate(input_dps):
                result_new = {
                    'hadm_id': result['hadm_id'],
                    'output': outputs[in_batch_i]
                }
                results_2.append(result_new)
                ids_done.append(result['hadm_id'])
                global_verbose_flag = False
                runtime_this = runtime_this_batch/len(input_dps)
                metadata_dict['runtime_each'].append(runtime_this)
            with open(os.path.join(f'data/mimic4/{target_task}_output', f"{save_name}{lora_name}.json"), 'w', encoding='utf-8') as f:
                json.dump(results_2, f, indent=4)
            with open(os.path.join(f'data/mimic4/{target_task}_output', f"{save_name}{lora_name}_metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=4)
        runtime_sum_e = time.time()
        metadata_dict['runtime_sum'] = runtime_sum_e - runtime_sum_s,
        with open(os.path.join(f'data/mimic4/{target_task}_output', f"{save_name}{lora_name}_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=4)

if __name__ == '__main__':
    main()