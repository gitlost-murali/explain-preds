
# Baseline LSTM

Trains a BERT model using train & dev set.

## Running the model

1. To train the model

```
python train.py --train_file ../../../data/train.tsv --dev_file ../../../data/dev.tsv --learning_rate 5e-5 --max_seq_len 200 --langmodel_name bert-base-uncased --output_modelname models/bert-outputs
```

2. To evaluate the model

```
python evaluate.py --test_file ../../../data/test.tsv --best_modelname models/bert-outputs
```

3. To predict the model

```
python predict.py --test_file ../../../data/test.tsv --best_modelname models/bert-outputs --output_predfile preds.txt
```

If you want to try any other bert variant, please type

python train.py --langmodel_name distilbert-base-uncased

Or for other models, try these:

microsoft/deberta-v3-base
xlnet-base-cased
roberta-base
bert-base-uncased
distilbert-base-uncased
albert-base-v2

