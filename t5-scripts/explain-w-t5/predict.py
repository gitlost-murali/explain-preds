"""## INFERENCE from checkpoint"""
from specific_utils import LitModel
from transformers import AutoTokenizer
import pytorch_lightning as pl
from specific_utils import Inference_LitOffData, TemplateHandler
from common_utils import write_preds_tofile, debug_w_template
import argparse
import torch


def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_file", type=str, default='../../data/test.tsv', required= True,
                        help="If added, use trained model to predict on test set")

    parser.add_argument("--output_predfile", type=str, default='preds.txt', required= True,
                        help="File to store the predictions. Each prediction in a line")

    parser.add_argument("--debug_file", type=str, default='debug.csv',
                        help="Shows failed instances and all their predictions")

    parser.add_argument("--best_modelname", default="models/bestmodel.ckpt", type=str,
                        help="Name of the trained model that will be saved after training")

    parser.add_argument("--offensive_lexicon",
                        default="../lexicon_words/final_offensive_lexicon.txt",
                        type=str, help="Path of the txt file containing offensive lexicon")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training")

    parser.add_argument("--device", default="gpu", type=str,
                        help="Type of device to use. gpu/cpu strict naming convention")

    parser.add_argument("--show_cm", default=True, type=bool,
                        help="Show confusion matrix")

    args = parser.parse_args()
    return args


def main():
    '''Main function to test neural network given cmd line arguments'''
    args = create_arg_parser()
    print(args)
    templatehandler = TemplateHandler()
    model = LitModel.load_from_checkpoint(args.best_modelname,
                                          templatehandler = templatehandler)
    model.eval()
    testdm = Inference_LitOffData(test_file = args.test_file,
                                  offensive_lexfile= args.offensive_lexicon,
                                  templatehandler=templatehandler)
    device_to_train = args.device if torch.cuda.is_available() else "cpu"
    print("Device to train", device_to_train)
    trainer = pl.Trainer(accelerator=device_to_train, devices=1)
    # trainer.test(model, testdm.test_dataloader())
    outs = trainer.predict(model, testdm.test_dataloader())
    write_preds_tofile(outs, "onlylabel_preds", templatehandler,args.output_predfile)
    debug_w_template(outs, args.debug_file)

if __name__ == '__main__':
    main()
