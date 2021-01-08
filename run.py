from numpy.core.fromnumeric import mean
from numpy.lib.function_base import average
from torch._C import Value
from torch.utils.data.dataloader import DataLoader
from models import RnnModel, TextCNN
import torch
import torch.nn as nn
import spacy
import torch
from torchtext import data
import argparse
import os
import logging
import random
from tqdm import tqdm
import numpy as np
import json
from utils import AverageMeter, TextDataset, count_parameters, layer_wise_parameters, human_format
import torch.optim as optim
import torch.cuda
import nltk
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


spacy_en = spacy.load("en_core_web_lg")


def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


def process_args(in_args):
    if not args.test_only:
        if not os.path.exists(in_args.output_path):
            os.system("mkdir -p %s" % in_args.output_path)
        output_path = os.path.join(
            in_args.output_path, "run_%s" % args.model_name)
        max_num = -1
        for folder in os.listdir(in_args.output_path):
            if ("run_" + args.model_name) in folder:
                curr_num = int(folder.split('_')[-1])
                if curr_num > max_num:
                    max_num = curr_num
        output_path += '_%d' % (max_num + 1)
        in_args.output_path = output_path
        assert not os.path.exists(in_args.output_path)
        os.system("mkdir -p %s" % in_args.output_path)

    if in_args.test_only:
        in_args.log_path = os.path.join(in_args.output_path, "test_log.txt")
    else:
        in_args.log_path = os.path.join(
            in_args.output_path, "training_log.txt")
    if os.path.isfile(os.path.join(in_args.output_path, in_args.model_name)):
        in_args.checkpoint = os.path.join(in_args.output_path,
                                          in_args.model_name)
    else:
        in_args.checkpoint = None
    if torch.cuda.is_available():
        args.gpu = True
    else:
        args.gpu = False


@torch.no_grad()
def do_validation(model: nn.Module, val_loader: DataLoader):
    model.eval()
    total = 0
    correct = 0

    all_predictions = []
    all_labels = []

    for batch in tqdm(val_loader, desc="Validating"):
        labels = batch[0]  # [batch_size]
        texts = batch[1].t()  # [text_len, batch_size]

        output = model(texts)
        predictions = torch.argmax(output, dim=1)
        all_predictions += predictions.tolist()
        all_labels += labels.tolist()
        correct_num = torch.sum(predictions == labels).item()

        total += len(batch[0])
        correct += correct_num

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    precision = precision_score(all_labels, all_predictions, average="macro")
    recall = recall_score(all_labels, all_predictions, average="macro")
    F1 = f1_score(all_labels, all_predictions, average="macro")
    logger.info("Validation: total %d items; %d are correct." %
                (total, correct))
    return correct / total, precision, recall, F1


@torch.no_grad()
def predict_test():
    saved_file = torch.load(os.path.join(
        args.output_path,
        "best_%s_model.pkl" % args.model_name
    ))
    TEXT = saved_file["vocab"]
    test = data.TabularDataset(os.path.join(args.data_path, args.test_file), format='csv', skip_header=True,
                               fields=[('label_id', None), ('text', TEXT)])
    test_lens = list(map(lambda x: len(x), test.text))
    logger.info("Max Length: %d - Avg Length: %d" % (
        max(test_lens), mean(test_lens)
    ))
    test_lengths = []
    for test_item in test:
        test_item.text, test_length = padding(
            test_item.text[:args.max_len], args.max_len
        )
        test_lengths.append(test_length)
    test_loader = DataLoader(
        TextDataset(test, TEXT.vocab, DEVICE, test_lengths, is_test=True),
        batch_size=args.test_batch_size, shuffle=False
    )
    if args.model_name in ["LSTM", "RNN", "GRU"]:
        model = RnnModel(logger=logger, num_words=len(TEXT.vocab),
                         pretrained_embedding=TEXT.vocab.vectors,
                         num_layers=args.num_layers,
                         model_type=args.model_name, input_size=300, hidden_size=args.hidden_size,
                         dropout=args.dropout, num_classes=4,
                         bidirectional=args.bidirectional, RNN_nonlinear_type=args.RNN_nonlinear_type)
    elif args.model_name == "CNN":
        model = TextCNN(logger=logger, num_words=len(TEXT.vocab),
                        pretrained_embedding=TEXT.vocab.vectors,
                        input_size=300, num_classes=4)
    else:
        raise ValueError("Unsupported model.")
    model.load_state_dict(saved_file["model"])
    model = model.to(DEVICE)
    all_predictions = []
    model.eval()
    for batch in tqdm(test_loader, desc="Generating"):
        texts = batch[0].t()  # [text_len, batch_size]

        output = model(texts)
        predictions = torch.argmax(output, dim=1) + 1
        all_predictions += predictions.tolist()
    test_data = pd.read_csv('data/test.csv')
    test_data["Class Index"] = pd.Series(all_predictions)
    test_data.to_csv(os.path.join(args.output_path, 'prediction.csv'),
                     index=False)


def padding(text, max_length):
    length = len(text)
    while len(text) < max_length:
        text.append("<pad>")
    return text, length


def main():
    logger.info("Reading data...")
    train, val = data.TabularDataset.splits(
        path=args.data_path, train=args.train_file, validation=args.dev_file,
        format='csv', skip_header=True,
        fields=[('label_id', LABEL), ('text', TEXT)]
    )
    logger.info("Done.")

    train_lengths = []
    val_lengths = []
    for train_item in train:
        train_item.text, train_length = padding(
            train_item.text[:args.max_len], args.max_len
        )
        train_lengths.append(train_length)
    for val_item in val:
        val_item.text, val_length = padding(
            val_item.text[:args.max_len], args.max_len
        )
        val_lengths.append(val_length)

    train_lens = list(map(lambda x: len(x), train.text))
    val_lens = list(map(lambda x: len(x), val.text))
    logger.info("Max Length:\n\tTrain: %d\n\tVal: %d\n"
                "Avg Length:\n\tTrain: %d\n\tVal: %d" % (
                    max(train_lens), max(val_lens),
                    mean(train_lens), mean(val_lens)
                ))

    all_labels = np.unique(list(map(lambda x: int(x), train.label_id)))
    num_classes = len(all_labels)
    logger.info("Num classes: %d. They are: %s" % (num_classes, all_labels))

    logger.info("Training examples: %d. Dev examples: %d" %
                (len(train), len(val)))
    logger.info("Print an example from training set.")
    example_id = random.randint(0, len(train) - 1)
    logger.info("Example %d -- Label: %s -- Text: %s" %
                (example_id, train[example_id].label_id, train[example_id].text))
    TEXT.build_vocab(train, vectors='glove.840B.300d',
                     max_size=args.vocab_size if args.vocab_size > 0 else None,
                     min_freq=args.min_freq)
    logger.info("Vocab size: %d" % len(TEXT.vocab))
    vocab_size = len(TEXT.vocab)

    # train_iter = data.BucketIterator(train, batch_size=args.train_batch_size,
    #                                  sort_key=lambda x: len(x.text),
    #                                  shuffle=True, device=DEVICE)
    # val_iter = data.BucketIterator(val, batch_size=args.test_batch_size,
    #                                sort_key=lambda x: len(x.text),
    #                                shuffle=True, device=DEVICE)

    train_loader = DataLoader(
        TextDataset(train, TEXT.vocab, DEVICE, train_lengths),
        batch_size=args.train_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TextDataset(val, TEXT.vocab, DEVICE, val_lengths),
        batch_size=args.test_batch_size, shuffle=True
    )

    if args.model_name in ["LSTM", "RNN", "GRU"]:
        model = RnnModel(logger=logger, num_words=vocab_size,
                         pretrained_embedding=TEXT.vocab.vectors if args.use_glove else None,
                         num_layers=args.num_layers,
                         model_type=args.model_name, input_size=300, hidden_size=args.hidden_size,
                         dropout=args.dropout, num_classes=num_classes,
                         bidirectional=args.bidirectional, RNN_nonlinear_type=args.RNN_nonlinear_type)
    elif args.model_name == "CNN":
        model = TextCNN(logger=logger, num_words=vocab_size,
                        kernel_nums=args.kernel_num, kernel_sizes=args.kernel_size,
                        pretrained_embedding=TEXT.vocab.vectors if args.use_glove else None,
                        input_size=300, num_classes=num_classes)
    else:
        raise ValueError("Unsupported model.")
    if args.gpu:
        model = model.cuda()

    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unsupported optimizer type.")
    sheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, args.lr_step, verbose=True)
    criterion = nn.CrossEntropyLoss()

    table = layer_wise_parameters(model)
    logger.info('Breakdown of the model parameters\n%s' % table)
    logger.info('Total parameters: %s' % (
        human_format(count_parameters(model)),
    ))

    best_epoch = -1
    best_val_score = 0
    for epoch in range(args.epoch_num):
        model.train()
        logger.info("Epoch %d starts!" % epoch)
        loss_avg = AverageMeter()
        pbar = tqdm(train_loader)
        pbar.set_description("Epoch %d - Batch %d - Loss %.3ff"
                             % (epoch, -1, 0))
        total = 0
        correct = 0
        for batch_idx, batch in enumerate(pbar):
            labels = batch[0]  # [batch_size]
            texts = batch[1].t()  # [text_len, batch_size]
            optimizer.zero_grad()

            out = model(texts)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(out, dim=1)
            correct_num = torch.sum(predictions == labels).item()

            total += len(batch[0])
            correct += correct_num

            loss_avg.update(loss.item())
            pbar.set_description("Epoch %d - Batch %d - Loss %.3f"
                                 % (epoch, batch_idx, loss.item()))
        sheduler.step()
        logger.info("Training: total %d items; %d are correct. Accuracy: %.3f" %
                    (total, correct, correct / total))
        curr_val_score, precision, recall, F1 = do_validation(
            model, val_loader)
        logger.info("Epoch %d ends! Average loss: %.3f.\nValidation metrics:\n\t"
                    "Accuracy: %.2f%%\n\tPrecision: %.2f\n\tRecall: %.2f\n\tF1: %.2f" %
                    (epoch, loss_avg.avg, curr_val_score * 100, precision, recall, F1))
        if curr_val_score > best_val_score:
            best_epoch = epoch
            best_val_score = curr_val_score
            logger.info("Best accuracy is %.2f%% in epoch %d. Saving model..." % (
                curr_val_score * 100, epoch))
            torch.save({
                "model": model.state_dict(),
                "vocab": TEXT
            }, os.path.join(args.output_path, "best_%s_model.pkl" % args.model_name))
    logger.info("The best model is in epoch %d, the best score is %.3f%%" %
                (best_epoch, best_val_score * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run training.")
    parser.add_argument("--data_path", default="./merged_data")
    parser.add_argument("--train_file", default="train.csv")
    parser.add_argument("--dev_file", default="dev.csv")
    parser.add_argument("--test_file", default="test.csv")
    parser.add_argument("--output_path", default="runs")
    parser.add_argument("--model_name", required=True,
                        help="Model name for saving.")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--RNN_nonlinear_type", default="tanh")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tokenizer", type=str, default="nltk")
    parser.add_argument("--vocab_size", type=int, default=-1)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--lr_step", type=float, default=0.1)
    parser.add_argument("--kernel_size", type=int,
                        nargs='+', default=[2, 3, 4])
    parser.add_argument("--kernel_num", type=int,
                        nargs="+", default=[256, 256, 256])
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--use_glove", action="store_true")
    args = parser.parse_args()
    process_args(args)
    assert args.model_name in ["LSTM", "RNN", "GRU", "CNN"]
    assert len(args.kernel_num) == len(args.kernel_size)
    assert args.optimizer in ["SGD", "Adam", "AdamW"]
    assert args.tokenizer in ["spacy", "nltk"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)
    TEXT = data.Field(
        sequential=True,
        tokenize=tokenizer if args.tokenizer == "spacy" else nltk.word_tokenize,
        lower=True
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.checkpoint:
        log_file = logging.FileHandler(args.log_path, 'a')
    else:
        log_file = logging.FileHandler(args.log_path, 'w')
    log_file.setFormatter(fmt)
    logger.addHandler(log_file)

    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' % json.dumps(
        vars(args), indent=4, sort_keys=True))

    if args.test_only:
        logger.info("Do prediction only.")
        predict_test()
    else:
        main()
