import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

from transformers import AutoTokenizer
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
from sklearn.metrics import f1_score, accuracy_score

from utils import read_corpus
from common_utils import calculate_confusion_matrix, plot_confusion_matrix

from sklearn.utils import class_weight

from templatefile_1 import TemplateHandler

class T5DatasetClass(Dataset):
    def __init__(self, texts, labels, tokenizer,
                max_length, labelmapper,
                class_weightscores = {"OFF": 1.5, "NOT": 0.7}, label_maxlen = 25):
        super(T5DatasetClass, self).__init__()
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.texts=texts
        self.orglabels=labels
        self.labelmapper = labelmapper
        self.class_weightscores = class_weightscores
        self.class_weights = [ self.class_weightscores[lb] for lb in self.orglabels ]
        self.labels = [ self.labelmapper[lb] for lb in self.orglabels]
        self.label_maxlen = label_maxlen

        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        text1 = self.texts[index]
        inputs = self.tokenizer(
            text = text1 ,
            text_pair = None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]


        target_encoding = self.tokenizer(
            self.labels[index],
            max_length=self.label_maxlen,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        label = target_encoding.input_ids
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        label[label == self.tokenizer.pad_token_id] = -100

        output =  {
            'ids': ids[0],
            'mask': mask[0],
            'labels': label[0],
            'class_weights': torch.tensor(self.class_weights[index], dtype=torch.float),
            'sent_idx': torch.tensor(index, dtype=torch.long),
            'actual_label': self.labels[index]
            }
        
        return output


class LitOffData(pl.LightningDataModule):
    def __init__(self, templatehandler: TemplateHandler,
                 train_file: str = 'data/train.tsv',
                 dev_file: str = 'data/dev.tsv',
                 batch_size = 4,
                 max_seq_len = 100,
                 modelname = 't5-base',
                ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file, self.dev_file = train_file, dev_file
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.labelmapper = templatehandler.labelmapper
        self.read_data()
        
    def read_data(self):
        # Read in the data
        self.X_train, self.Y_train = read_corpus(self.train_file)
        self.X_dev, self.Y_dev = read_corpus(self.dev_file)

    def setup(self, stage = None):
        self.get_weights()
        self.train_dataset= T5DatasetClass(tokenizer = self.tokenizer, max_length=self.max_seq_len,
                                        texts = self.X_train, labels = self.Y_train, labelmapper = self.labelmapper, class_weightscores = self.class_weightscores)
        self.val_dataset= T5DatasetClass(tokenizer = self.tokenizer, max_length=self.max_seq_len,
                                      texts = self.X_dev, labels = self.Y_dev, labelmapper = self.labelmapper, class_weightscores = self.class_weightscores)

    def train_dataloader(self):
        dataloader=DataLoader(dataset=self.train_dataset,batch_size=self.batch_size)
        return dataloader

    def val_dataloader(self):
        dataloader=DataLoader(dataset=self.val_dataset,batch_size=self.batch_size)
        return dataloader

    def get_weights(self):
        classnames = list(self.labelmapper.keys())
        weightscores = class_weight.compute_class_weight(class_weight = 'balanced'
                                                    ,classes = classnames
                                                    ,y = self.Y_train)
        weightscores = list(weightscores)
        self.class_weightscores = dict((k,v) for (k,v) in zip(classnames, weightscores))
        print(self.class_weightscores)
        return self.class_weightscores

# define the LightningModule

class LitModel(pl.LightningModule):
    def __init__(self, templatehandler: TemplateHandler, modelname = "t5-base", num_labels = 2, dropout = 0.2,
                learning_rate = 1e-5, batch_size = 4, save_cm_plot = True):
        super().__init__()
        self.base = T5ForConditionalGeneration.from_pretrained(modelname)
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.dropout = nn.Dropout(dropout)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.templatehandler = templatehandler
        force_words = ["comment", "is", "offensive"]
        self.force_words_ids = self.tokenizer(force_words, add_special_tokens=False).input_ids
        self.save_cm_plot = save_cm_plot
        self.log("batch_size", self.batch_size)

    def forward(self, ids, mask, labels, **kwargs):
        out = self.base( input_ids = ids,attention_mask=mask, labels=labels, return_dict=True)
        # https://stackoverflow.com/questions/73314467/output-logits-from-t5-model-for-text-generation-purposes
        # For generating stuff greedily. For debugging purpose only. Can be ignored for now.
        loss = out.loss
        return loss
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss = self(**batch)
        loss = loss * batch['class_weights']
        loss = torch.mean(loss)

        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(**batch)
        loss = loss * batch['class_weights']
        loss = torch.mean(loss)

        self.log('val_loss', loss,on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        outputs = self.base.generate(batch["ids"])
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds = [self.templatehandler.decode_preds(prd) for prd in preds]
        gts = [self.templatehandler.decode_preds(snt) for snt in batch["actual_label"]]
        return {"preds": preds, "gts": gts}

    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.base.generate(batch["ids"])
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds = [self.templatehandler.decode_preds(prd) for prd in preds]
        gts = [self.templatehandler.decode_preds(snt) for snt in batch["actual_label"]]
        input_sentences = self.tokenizer.batch_decode(batch['ids'], skip_special_tokens = True)
        return {"input_sentences": input_sentences, "preds": preds, "gts": gts}

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        losses = torch.stack([ item["loss"]  for item in outs])
        loss = losses.mean()
        self.log("val_loss_epoch", loss)
        print("val_loss_epoch", loss)

    def test_epoch_end(self,outs):
        # outs is a list of whatever you returned in `validation_step`

        preds = []
        gts = []        
        for item in outs:
            preds.extend(item["preds"])
            gts.extend(item["gts"])

        acc = accuracy_score(preds, gts)
        f1 = f1_score(preds, gts, average='macro')

        if self.save_cm_plot:
            # get the classnames from encoder
            matrix = calculate_confusion_matrix(gts, preds, list(set(gts+preds)) )
            plot_confusion_matrix(matrix)

        self.log("test_epoch_acc", acc)
        self.log("test_epoch_f1", f1)
        print("test_acc_epoch", acc)
        print("test_F1_epoch", f1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class Inference_LitOffData(pl.LightningDataModule):
    def __init__(self, 
                 test_file: str = 'data/test.tsv',
                 batch_size = 4,
                 max_seq_len = 100,
                 modelname = 't5-base',
                 labelmapper = {"OFF": "offensive", "NOT": "not offensive"},
                ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.test_file = test_file
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.labelmapper = labelmapper
        self.read_data()
        self.setup()

    def read_data(self):
        # Read in the data
        self.X_test, self.Y_test = read_corpus(self.test_file)


    def setup(self, stage = None):
        self.test_dataset= T5DatasetClass(tokenizer = self.tokenizer, max_length=self.max_seq_len,
                                       texts = self.X_test, labels = self.Y_test,
                                       labelmapper = self.labelmapper,
                                       class_weightscores = {'OFF': 1.51, 'NOT': 0.74})

    def test_dataloader(self):
        dataloader=DataLoader(dataset=self.test_dataset,batch_size=self.batch_size)    
        return dataloader
