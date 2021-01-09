# Text Classification Project

## Set up environment
```bash
conda create -n nlp-hw-2 python=3.7
conda activate nlp-hw-2
conda install tqdm matplotlib spacy scikit-learn pandas
conda install prettytable -c conda-forge
conda install pytorch torchtext cudatoolkit=10.2 -c pytorch
python -m spacy download en_core_web_lg
```

```python
$ python
>>> import nltk
>>> nltk.download("punkt")
```

## How to run
Train:
```bash
python -W ignore run.py --model_name CNN --max_len 200 --optimizer AdamW --use_glove
```

Predict:
```bash
python -W ignore run.py --model_name CNN --max_len 200 --optimizer AdamW --use_glove --test_only --output_path ./runs/run_CNN_0/
```

## Results
See [report](nlp_homework_2_report.pdf).

The final prediction file is `prediction.csv`.
