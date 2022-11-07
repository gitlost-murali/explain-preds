# LSTM baseline

## Embeddings

* fasttext embeddings can be downloaded from [here](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz)

* Extract the embeddings (.vec file) and store in "embeddings/" folder under the name "cc.en.300.vec".

## Usage

1. To train the model

```
python train.py --train_file ../../../data/train.tsv --dev_file ../../../data/dev.tsv --embeddings embeddings/cc.en.300.vec --learning_rate 1e-4 --max_seq_len 100 --output_modelname models/best_model.h5
```

2. To evaluate the model

```
python evaluate.py --test_file ../../../data/test.tsv --best_modelname models/best_model.h5
```

3. To predict the model

```
python predict.py --test_file ../../../data/test.tsv --best_modelname models/best_model.h5 --output_predfile preds.txt
```
