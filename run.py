from numpy.core.fromnumeric import mean
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


spacy_en = spacy.load("en_core_web_lg")


def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


def process_args(in_args):
    os.system("mkdir -p %s" % in_args.output_path)
    in_args.log_path = os.path.join(in_args.output_path, "training_log.txt")
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
def do_validation(model: nn.Module, val_iter):
    model.eval()
    total = 0
    correct = 0
    for batch in tqdm(val_iter, desc="Validating"):
        labels = batch.label_id  # [batch_size]
        texts = batch.text  # [text_len, batch_size]

        output = model(texts)
        predictions = torch.argmax(output, dim=1) + 1
        correct_num = torch.sum(predictions == labels).item()

        total += len(batch)
        correct += correct_num
    logger.info("Validation: total %d items; %d are correct." %
                (total, correct))
    return correct / total


def predict_test(test_iter):
    pass


def padding(text, length):
    while len(text) < length:
        text.append("<pad>")
    return text


def main():
    logger.info("Reading data...")
    train, val = data.TabularDataset.splits(
        path=args.data_path, train=args.train_file, validation=args.dev_file,
        format='csv', skip_header=True,
        fields=[('label_id', LABEL), ('text', TEXT)]
    )
    test = data.TabularDataset(os.path.join(args.data_path, args.test_file), format='csv', skip_header=True,
                               fields=[('label_id', None), ('text', TEXT)])
    logger.info("Done.")

    # for train_item in train:
    #     train_item.text = padding(train_item.text[:args.max_len], args.max_len)
    # for val_item in val:
    #     val_item.text = padding(val_item.text[:args.max_len], args.max_len)
    # for test_item in test:
    #     test_item.text = padding(test_item.text[:args.max_len], args.max_len)

    for train_item in train:
        train_item.text = train_item.text[:args.max_len]
    for val_item in val:
        val_item.text = val_item.text[:args.max_len]
    for test_item in test:
        test_item.text = test_item.text[:args.max_len]

    train_lens = list(map(lambda x: len(x), train.text))
    val_lens = list(map(lambda x: len(x), val.text))
    test_lens = list(map(lambda x: len(x), test.text))
    logger.info("Max Length:\n\tTrain: %d\n\tVal: %d\n\tTest: %d\n"
                "Avg Length:\n\tTrain: %d\n\tVal: %d\n\tTest: %d" % (
                    max(train_lens), max(val_lens), max(test_lens),
                    mean(train_lens), mean(val_lens), mean(test_lens)
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
                     min_freq=10)
    logger.info("Vocab size: %d" % len(TEXT.vocab))
    vocab_size = len(TEXT.vocab)

    train_iter = data.BucketIterator(train, batch_size=args.train_batch_size,
                                     sort_key=lambda x: len(x.text),
                                     shuffle=True, device=DEVICE)
    val_iter = data.BucketIterator(val, batch_size=args.test_batch_size,
                                   sort_key=lambda x: len(x.text),
                                   shuffle=True, device=DEVICE)
    test_iter = data.Iterator(dataset=test, batch_size=args.test_batch_size,
                              train=False, sort=False, device=DEVICE)

    # train_loader = DataLoader(TextDataset(
    #     train, TEXT.vocab, DEVICE), batch_size=args.train_batch_size, shuffle=True
    # )
    # val_loader = DataLoader(TextDataset(
    #     val, TEXT.vocab, DEVICE), batch_size=args.test_batch_size, shuffle=True
    # )
    # test_loader = DataLoader(TextDataset(
    #     test, TEXT.vocab, DEVICE, is_test=True), batch_size=args.test_batch_size, shuffle=True
    # )

    if args.model_name in ["LSTM", "RNN", "GRU"]:
        model = RnnModel(logger=logger, num_words=vocab_size,
                         pretrained_embedding=TEXT.vocab.vectors,
                         num_layers=args.num_layers,
                         model_type=args.model_name, input_size=300, hidden_size=args.hidden_size,
                         dropout=args.dropout, num_classes=num_classes,
                         bidirectional=args.bidirectional, RNN_nonlinear_type=args.RNN_nonlinear_type)
    elif args.model_name == "CNN":
        model = TextCNN(logger=logger, num_words=vocab_size,
                        pretrained_embedding=TEXT.vocab.vectors,
                        input_size=300, num_classes=num_classes)
    else:
        raise ValueError("Unsupported model.")
    if args.gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr)
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
        pbar = tqdm(train_iter)
        pbar.set_description("Epoch %d - Batch %d - Loss %.3ff"
                             % (epoch, -1, 0))
        total = 0
        correct = 0
        for batch_idx, batch in enumerate(pbar):
            labels = batch.label_id - 1  # [batch_size]
            texts = batch.text  # [text_len, batch_size]
            optimizer.zero_grad()

            out = model(texts)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(out, dim=1)
            correct_num = torch.sum(predictions == labels).item()

            total += len(batch)
            correct += correct_num

            loss_avg.update(loss.item())
            pbar.set_description("Epoch %d - Batch %d - Loss %.3f"
                                 % (epoch, batch_idx, loss.item()))
        logger.info("Training: total %d items; %d are correct. Accuracy: %.3f" %
                    (total, correct, correct / total))
        curr_val_score = do_validation(model, val_iter)
        logger.info("Epoch %d ends! Average loss: %.3f. Validation accuracy: %.2f%%" %
                    (epoch, loss_avg.avg, curr_val_score * 100))
        if curr_val_score > best_val_score:
            best_epoch = epoch
            best_val_score = curr_val_score
            logger.info("Best accuracy is %.2f%% in epoch %d. Saving model..." % (
                curr_val_score * 100, epoch))
            torch.save({
                "model": model.state_dict()
            }, os.path.join(args.output_path, "best_%s_model.pkl" % args.model_name))
    logger.info("The best model is in epoch %d, the best score is %.3f" %
                (best_epoch, best_val_score))
    predict_test(test_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run training.")
    parser.add_argument("--data_path", default="./merged_data")
    parser.add_argument("--train_file", default="train.csv")
    parser.add_argument("--dev_file", default="dev.csv")
    parser.add_argument("--test_file", default="test.csv")
    parser.add_argument("--output_path", required=True,
                        help="Output folder.")
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
    parser.add_argument("--vocab_size", type=int, default=-1)
    parser.add_argument("--max_len", type=int, required=True)
    args = parser.parse_args()
    process_args(args)
    assert args.model_name in ["LSTM", "RNN", "GRU", "CNN"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)
    TEXT = data.Field(sequential=True, tokenize=nltk.word_tokenize, lower=True)

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

    main()
