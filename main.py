import os
import time
import argparse
import torch
import sys
import copy
import random
import numpy as np
import pickle
from collections import defaultdict
from sampler import WarpSampler
from model import CASM
from tqdm import tqdm
from utils import *

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='train', required=False)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--maxlen', default=150, type=int)
parser.add_argument('--hidden_units', default=85, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=801, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.25, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)


tstInt = None
with open('Tianchi_tst_int', 'rb') as fs:
    tstInt = np.array(pickle.load(fs))

tstStat = (tstInt!=None)
tstUsrs = np.reshape(np.argwhere(tstStat!=False), [-1])
tstUsrs = tstUsrs + 1
print(len(tstUsrs))


if __name__ == "__main__":

    args = parser.parse_args()
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    dataset = data_partition_tmall(args.dataset)
    [user_train, user_valid, user_test, Beh, Beh_w, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    sampler = WarpSampler(user_train, Beh, Beh_w, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CASM(usernum, itemnum, args).to(device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass 

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()



    for epoch in range(1, args.num_epochs + 1):
        model.train() 
        
        total_loss = 0
        for step in range(num_batch):
            u, seq, pos, neg, seq_cxt, pos_cxt, pos_weight, neg_weight, recency = sampler.next_batch()
            
            u = torch.from_numpy(np.array(u)).long().to(device)
            seq = torch.from_numpy(np.array(seq)).long().to(device)
            pos = torch.from_numpy(np.array(pos)).long().to(device)
            neg = torch.from_numpy(np.array(neg)).long().to(device)
            seq_cxt = torch.from_numpy(np.array(seq_cxt)).float().to(device)
            pos_cxt = torch.from_numpy(np.array(pos_cxt)).float().to(device)
            pos_weight = torch.from_numpy(np.array(pos_weight)).float().to(device)
            neg_weight = torch.from_numpy(np.array(neg_weight)).float().to(device)
            recency = torch.from_numpy(np.array(recency)).float().to(device)

            
            model.zero_grad()
            loss, auc = model(seq, pos, neg, seq_cxt, pos_cxt, pos_weight, neg_weight, recency)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'loss in epoch {epoch}: {total_loss / num_batch:.4f}')
        
        model.eval()
        
        if epoch % 20 == 0:
            t1 = time.time() - t0
            T += t1
            print('Evaluating')
            t_valid = torch_evaluate_valid(model, dataset, tstUsrs, args)
            
            print(f'epoch {epoch}, time: {T:.1f}s, valid (NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f})')

    sampler.close()
    f.close()

    print("Done")
