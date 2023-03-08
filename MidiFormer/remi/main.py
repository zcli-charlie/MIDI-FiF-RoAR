import argparse
import numpy as np
import random
import pickle
import os
import json

from torch.utils.data import DataLoader
from transformers import RoFormerConfig
from model import MidiFormer
from trainer import FormerTrainer
from midi_dataset import MidiDataset


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='../../dict/remi.pkl')
    parser.add_argument('--name', type=str, default='MidiFormer')

    ### pre-train dataset ###
    parser.add_argument("--datasets", type=str, nargs='+', default=['pop909', 'composer', 'pop1k7', 'ASAP', 'emopia'])

    ### parameter setting ###
    parser.add_argument('--pos_type', type=str, default='absolute')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--mask_percent', type=float, default=0.15,
                        help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)  # hidden state
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default: False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1, 2, 3], help="CUDA device ids")

    parser.add_argument("--use_mlm", action="store_true", default=False, help="whether to use mlm when training")  # default: False
    parser.add_argument("--use_clm", action="store_true", default=False, help="whether to use clm when training")  # default: False

    args = parser.parse_args()

    return args


def load_data(datasets):
    to_concat = []
    root = '../../data/remi'

    for dataset in datasets:
        if dataset in {'pop909', 'composer', 'emopia'}:
            X_train = np.load(os.path.join(root, f'{dataset}_train.npy'), allow_pickle=True)
            X_valid = np.load(os.path.join(root, f'{dataset}_valid.npy'), allow_pickle=True)
            X_test = np.load(os.path.join(root, f'{dataset}_test.npy'), allow_pickle=True)
            data = np.concatenate((X_train, X_valid, X_test), axis=0)

        elif dataset == 'pop1k7' or dataset == 'ASAP':
            data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)

        print(f'   {dataset}: {data.shape}')
        to_concat.append(data)

    training_data = np.vstack(to_concat)
    print('   > all training data:', training_data.shape)

    # shuffle during training phase
    index = np.arange(len(training_data))
    np.random.shuffle(index)
    training_data = training_data[index]
    split = int(len(training_data) * 0.85)
    X_train, X_val = training_data[:split], training_data[split:]

    return X_train, X_val


def main():
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset", args.datasets)
    X_train, X_val = load_data(args.datasets)

    trainset = MidiDataset(X=X_train)
    validset = MidiDataset(X=X_val)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))

    print("\nBuilding Former model")
    configuration = RoFormerConfig(max_position_embeddings=args.max_seq_len, vocab_size=2, d_model=args.hs,
                                   position_embedding_type=args.pos_type)
    # 0: MLM
    # 1: CLM
    midi_former = MidiFormer(formerConfig=configuration, e2w=e2w, w2e=w2e)

    print("\n Model:")
    print(midi_former)

    print("\nCreating Former Trainer")
    trainer = FormerTrainer(midi_former, train_loader, valid_loader, args.lr, args.batch_size, args.max_seq_len,
                              args.mask_percent, args.cpu, args.use_mlm, args.use_clm, args.cuda_devices)

    print("\nTraining Start")
    save_dir = 'result/pretrain/' + args.name
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 30:
            print('valid acc not improving for 30 epochs')
            break
        if args.use_mlm:
            if args.use_clm:
                train_loss, train_mlm_acc, train_clm_acc = trainer.train()
                valid_loss, valid_mlm_acc, valid_clm_acc = trainer.valid()
            else:
                train_loss, train_mlm_acc = trainer.train()
                valid_loss, valid_mlm_acc = trainer.valid()
        else:
            if args.use_clm:
                train_loss, train_clm_acc = trainer.train()
                valid_loss, valid_clm_acc = trainer.valid()
            else:
                train_loss = trainer.train()
                valid_loss = trainer.valid()

        if args.use_mlm:
            is_best = valid_mlm_acc > best_acc
            best_acc = max(valid_mlm_acc, best_acc)
        else:
            is_best = valid_clm_acc > best_acc
            best_acc = max(valid_clm_acc, best_acc)
        
        if is_best:
            bad_cnt, best_epoch = 0, epoch
        else:
            bad_cnt += 1

        if args.use_mlm:
            if args.use_clm:
                print('epoch: {}/{} | Train Loss: {} | Train acc: {}, {} | Valid Loss: {} | Valid acc: {}, {}'.format(
                    epoch + 1, args.epochs, train_loss, train_mlm_acc, train_clm_acc, valid_loss, valid_mlm_acc, valid_clm_acc))
            else:
                print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}'.format(
                    epoch + 1, args.epochs, train_loss, train_mlm_acc, valid_loss, valid_mlm_acc))
        else:
            if args.use_clm:
                print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}'.format(
                    epoch + 1, args.epochs, train_loss, train_clm_acc, valid_loss, valid_clm_acc))
            else:
                print('epoch: {}/{} | Train Loss: {} | Valid Loss: {}'.format(
                    epoch + 1, args.epochs, train_loss, valid_loss))

        if args.use_mlm:
            trainer.save_checkpoint(epoch, best_acc, valid_mlm_acc,
                                    valid_loss, train_loss, is_best, filename)
        else:
            trainer.save_checkpoint(epoch, best_acc, valid_clm_acc,
                                    valid_loss, train_loss, is_best, filename)

        with open(os.path.join(save_dir, 'log'), 'a') as outfile:
            if args.use_mlm:
                if args.use_clm:
                    outfile.write('Epoch {}: train_loss={}, train_acc={}, {}, valid_loss={}, valid_acc={},{}\n'.format(
                        epoch + 1, train_loss, train_mlm_acc, train_clm_acc, valid_loss, valid_mlm_acc, valid_clm_acc))
                else:
                    outfile.write('Epoch {}: train_loss={}, train_acc={}, valid_loss={}, valid_acc={}\n'.format(
                        epoch + 1, train_loss, train_mlm_acc, valid_loss, valid_mlm_acc))
            else:
                if args.use_clm:
                    outfile.write('Epoch {}: train_loss={}, train_acc={}, valid_loss={}, valid_acc={}\n'.format(
                        epoch + 1, train_loss, train_clm_acc, valid_loss, valid_clm_acc))
                else:
                    outfile.write('Epoch {}: train_loss={}, valid_loss={}\n'.format(
                        epoch + 1, train_loss, valid_loss))


if __name__ == '__main__':
    main()
