import torch
import sys
import copy
import random
import numpy as np
import pickle
from collections import defaultdict



def data_partition_tmall(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_last_indx = {}
    user_valid = {}
    user_test = {}
    Beh = {}
    Beh_w = {}
    if not fname.startswith('data/'):
        fname = 'data/' + fname
    f = open('%s.txt' % fname, 'r')
    for line in f:
        u, i, b = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        if b == 'buy':
            last_pos_idx = len(User[u])
            user_last_indx[u] = last_pos_idx
            Beh[(u,i)] = [1,0,0,0]
            Beh_w[(u,i)] = 0.3

        elif b == 'cart':
            Beh[(u,i)] = [0,0,1,0]
            Beh_w[(u,i)] = 0.3

        elif b == 'fav':
            Beh[(u,i)] = [0,0,0,1]
            Beh_w[(u,i)] = 0.2
        
        elif b == 'pv':
            Beh[(u,i)] = [0,1,0,0]
            Beh_w[(u,i)] = 0.2
            
        User[u].append(i)

    for user in User:
        Beh[(user,0)] = [0,0,0,0]
        Beh_w[(user,0)] = 0
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            last_item_indx = user_last_indx[user]
            last_item = User[user][last_item_indx]
            items_list = User[user]
            del items_list[last_item_indx]

            user_train[user] = items_list
            #user_train[user] = [value for value in items_list if value != last_item]
            user_valid[user] = []
            user_valid[user].append(last_item)
            user_test[user] = []
            user_test[user].append(last_item)
    return [user_train, user_valid, user_test, Beh, Beh_w, usernum, itemnum]

def torch_evaluate_valid(model, dataset, tstUsrs, args):
    [train, valid, test, Beh, Beh_w, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    
    if usernum > 10000:
        users = tstUsrs  
        print(len(users))
    else:
        users = range(1, usernum + 1)
    
    model.eval()  
    with torch.no_grad(): 
        for u in users:
            seq_cxt = list()
            if len(train[u]) < 1 or len(valid[u]) < 1:
                continue
            
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break
            
            for i in seq:
                seq_cxt.append(Beh[(u, i)])
            seq_cxt = np.asarray(seq_cxt)
            
            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]
            testitemscxt = list()
            testitemscxt.append(Beh[(u, valid[u][0])])
            
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
                testitemscxt.append(Beh[(u, valid[u][0])])
            
            predictions = -model.predict(torch.tensor([u]), torch.tensor([seq]), torch.tensor(item_idx), 
                                         torch.tensor([seq_cxt]), torch.tensor(testitemscxt))
            predictions = predictions[0]
            rank = predictions.argsort().argsort()[0].cpu().numpy()  # Move tensor to CPU before converting to NumPy
            
            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            
            if valid_user % 100 == 0:
                sys.stdout.flush()
    
    return NDCG / valid_user, HT / valid_user