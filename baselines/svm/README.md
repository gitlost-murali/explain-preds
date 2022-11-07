# SVM

## Running

1. Training the model

```
python train.py --train_file ../../data/train.tsv --dev_file ../../data/dev.tsv --vec ngram_count --C 0.01 --class_weight balanced --output_modelname best_model.pt
```

2. Evaluating the model

```
python evaluate.py --test_file ../../data/test.tsv --best_modelname best_model.pt
```

3. Predicting on the model

```
python predict.py --test_file ../../data/test.tsv --best_modelname best_model.pt --output_predsfile preds.txt
```