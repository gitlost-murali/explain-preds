# T5 w/o explanation

## Running 

1. __Selecting a template:__ This folder contains multiple `templatefile_{1,2,3}.py` template files. Decide which template you want to try out. Currently, we use the template __(template 3)__ that gave us good results (Look at report for more details). If you want to try other templates, update `specific_utils.py` with the filename you want while importing TemplateHandler from `templatefile_{1,2,3}.py`. For example, change the following in `specific_utils.py` for template-1.

    From 

    `from templatefile_3 import TemplateHandler`

    to

    `from templatefile_1 import TemplateHandler`


2. Run the training script

```
python train.py --train_file ../../data/train.tsv --dev_file ../../data/dev.tsv --learning_rate 1e-4 --batch_size 8 --num_epochs 5 --max_seq_len 150 --langmodel_name t5-base --offensive_lexicon ../lexicon_words/final_offensive_lexicon.txt --ckpt_folder ./t5vanilla-wo-explain/ --seed 1234 --device cpu
```

3. The best model will be stored in t5explain-files/best-model.ckpt

4. Evaluating the model on the test file

```
python evaluate.py --test_file ../../data/test.tsv --best_modelname ./t5vanilla-wo-explain/bestmodel.ckpt --batch_size 16 --device cpu
```

5. Getting the predictions into a file

```
python predict.py --test_file ../../data/test.tsv --best_modelname t5explain-files/bestmodel.ckpt  --batch_size 16 --device cpu --output_predfile preds.txt
```
