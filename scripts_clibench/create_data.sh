python create_data_clibench_1.py \
    --dataset_hosp mimic4 \
    --dataset_note mimic4note \
    --data_path_hosp /home/ubuntu/derek-240306/clinical-event-pred/data/physionet.org/files/mimiciv/2.2/hosp \
    --data_path_note /home/ubuntu/derek-240306/clinical-event-pred/data/physionet.org/files/mimic-iv-note/2.2/note \
    --save_path_parsed data/mimic4/parsed \
    --shuffle_input_count 0 \
    --shuffle_target_count 0