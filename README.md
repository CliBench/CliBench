# CliBench: Multifaceted Evaluation of Large Language Models in Clinical Decisions on Diagnoses, Procedures, Lab Tests Orders and Prescriptions

## Download Data
### Raw MIMIC-IV dataset
Go to [https://mimic.physionet.org/](https://mimic.physionet.org/gettingstarted/access/) for access. Once you have the authority for the dataset, download the dataset at the `data` folder under the same directory as the this repository.

- MIMIC IV: https://physionet.org/content/mimiciv/2.2/

```
cd ../data/physionet.org/files/mimiciv/2.2/hosp
gzip -d patients.csv.gz
gzip -d admissions.csv.gz
gzip -d procedures_icd.csv.gz
gzip -d prescriptions.csv.gz
gzip -d diagnoses_icd.csv.gz
gzip -d labevents.csv.gz
gzip -d microbiologyevents.csv.gz
gzip -d d_icd_diagnoses.csv.gz
gzip -d d_icd_procedures.csv.gz
gzip -d d_labitems.csv.gz
gzip -d services.csv.gz
gzip -d transfers.csv.gz
```
```
cd ../data/physionet.org/files/mimic-iv-note/2.2/note
gzip -d discharge.csv.gz
gzip -d discharge_detail.csv.gz
gzip -d radiology.csv.gz
gzip -d radiology_detail.csv.gz
```

### NDC code metadata

Please download the NDC code metadata from the [this link](https://drive.google.com/drive/folders/160wDdKE4mZUDeHjC0HhQaNs5igIJGoQY?usp=sharing) and put `ndc_metadata.json` under `code_sys/NDC` directory.

## Dependencies

```
conda env create -f env.yml
```

## Clinical Benchmark

Data processing
1. `sh scripts_clibench/create_data.sh`: parse MIMIC data, save intermediate parsing result
2. `create_data_clibench_2.ipynb`: data processing, sample evaluation set, save splits
3. `infer_clibench.py` while `mode = 'inference'`: obtain seq2seq data used for fine-tuning LM

Generate output
* `sh scripts_clibench/infer.sh`

Calculate scores
* `score_clibench.ipynb`: Calculate metrics from generation result
* `score_clibench_breakdown.py`: Calculate metrics breakdown by data instance subgroups

## Citation

```
@article{clibench,
  title={CliBench: Multifaceted Evaluation of Large Language Models in Clinical Decisions on Diagnoses, Procedures, Lab Tests Orders and Prescriptions},
  author={Mingyu Derek Ma and Chenchen Ye and Yu Yan and Xiaoxuan Wang and Peipei Ping and Timothy Chang and Wei Wang},
month=jun,
  year={2024},
  url = {https://clibench.github.io/}
}
```