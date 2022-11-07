# explain-preds
Code repo of the final LFD paper

## Setup

### Installing the requirements

```
conda create -n lfdenvg8 python=3.9
conda activate lfdenvg8
pip install -r requirements.txt
```

## Running T5 experiments

1. For running the model with explanations, navigate to

`cd t5-scripts/explain-w-t5/`

2. Select a template you want. `cd t5-scripts/explain-w-t5/` contains multiple `templatefile_{1,2,3,4,5,6}.py` template files. Decide which template you want to try out.

3. Currently, we use the template (template 4) that gave us good results. If you want to try other template, update `specific_utils.py` with the filename you want. You need to change the import statement.

From 

`from templatefile_4 import TemplateHandler`

to

`from templatefile_1 import TemplateHandler`
```

4. Run the training script

python train.py --train_file ../../data/train.tsv --dev_file ../../data/dev.tsv --learning_rate 1e-4 --batch_size 8 --num_epochs 5 --max_seq_len 150 --langmodel_name t5-base --offensive_lexicon ../lexicon_words/final_offensive_lexicon.txt --ckpt_folder ./t5explain-files/ --seed 1234
```
5. The best model will be stored in t5explain-files/best-model.ckpt

6. Evaluating the model on the test file

