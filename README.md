# explain-preds
Code repo of the final LFD paper

## Setup

### Installing the requirements

```
conda create -n lfdenvg8 python=3.9
conda activate lfdenvg8
pip install -r requirements.txt
```
## Structure

* Discriminatory models are present inside "baselines/" folder

* Generative models (our approach) are present in "t5-scripts/" folder
```
├── data
├── baselines
│   ├── neural
│   │   ├── **bert/**
│   │   ├── **lstm/**
│   │   └── utils.py
│   └── **svm**
└── t5-scripts
    ├── common_utils.py
    ├── **explain-w-t5/**
    └── **t5-wo-explanation/**
    ├── lexicon_words
    │   └── final_offensive_lexicon.txt
├── LICENSE
├── README.md
├── requirements.txt
```

## Running T5 experiments

1. For running the model with explanations, navigate to

`cd t5-scripts/explain-w-t5/`

2. __Selecting a template:__ This folder contains multiple `templatefile_{1,2,3,4,5,6}.py` template files. Decide which template you want to try out. Currently, we use the template __(template 4)__ that gave us good results. If you want to try other templates, update `specific_utils.py` with the filename you want while importing TemplateHandler from `templatefile_{1,2,3,4,5,6}.py`. For example,

    From 

    `from templatefile_4 import TemplateHandler`

    to

    `from templatefile_1 import TemplateHandler`


4. Run the training script

```
python train.py --train_file ../../data/train.tsv --dev_file ../../data/dev.tsv --learning_rate 1e-4 --batch_size 8 --num_epochs 5 --max_seq_len 150 --langmodel_name t5-base --offensive_lexicon ../lexicon_words/final_offensive_lexicon.txt --ckpt_folder ./t5explain-files/ --seed 1234
```

5. The best model will be stored in t5explain-files/best-model.ckpt

6. Evaluating the model on the test file

T5 explanation model performance on different prompts 

| Prompt                                                                                                                                                                                | Macro-F1       |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| prompt-1 <br /><br />the model observed classified offensive since the following words showed up<br /><br /> the model observed classified not offensive                                                      | 78.93 +_ 0.9   |
| prompt-2 <br /><br />we had several words that rendered this offensive they were <br /><br />we had no words that rendered this offensive, they are non-existent!                                             | 79.64 +_ 0.45  |
| prompt-3                                                                                                                                                                              | NA             |
| prompt-4 <br /><br /> which words made us decide this is offensive, you ask? here you go: <br /><br />which words made us decide this is not offensive, you ask?                                               | 78.53 +_ 0.465 |
| prompt-5 <br /><br />the provided sentence may be interpreted as offensive by some users as certain offensive words occur such as<br /><br /> the provided sentence may not be found offensive by most users. | 79.51 +_ 0.6   |