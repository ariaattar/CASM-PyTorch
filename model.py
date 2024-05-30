import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CASM(nn.Module):
    def __init__(self, usernum, itemnum, args):
        super(CASM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_training = True
        self.u = None
        self.input_seq = None
        self.pos = None
        self.neg = None
        self.seq_cxt = None
        self.pos_cxt = None
        self.pos_weight = None
        self.neg_weight = None
        self.recency = None
        
        self.item_emb_table = nn.Embedding(itemnum + 1, args.hidden_units, padding_idx=0)
        self.seq_cxt_emb_layer = nn.Linear(4, args.hidden_units)
        self.feat_emb_layer = nn.Linear(args.hidden_units * 2, args.hidden_units)
        self.pos_emb_table = nn.Embedding(args.maxlen, args.hidden_units)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.transformer = nn.Transformer(d_model=args.hidden_units, nhead=args.num_heads, num_encoder_layers=args.num_blocks, num_decoder_layers=args.num_blocks, dim_feedforward=args.hidden_units, dropout=args.dropout_rate)
        
        self.test_item = None
        self.test_item_cxt = None
        self.test_logits = None
        
        self.pos_logits = None
        self.neg_logits = None
        self.loss = None
        self.auc = None

    def forward(self, input_seq, pos, neg, seq_cxt, pos_cxt, pos_weight, neg_weight, recency, test_item=None, test_item_cxt=None):
        self.input_seq = input_seq
        self.pos = pos
        self.neg = neg
        self.seq_cxt = seq_cxt
        self.pos_cxt = pos_cxt
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.recency = recency
        
        mask = (self.input_seq != 0).float().unsqueeze(-1)
        
        # sequence embedding, item embedding table
        self.seq = self.item_emb_table(self.input_seq)
        self.seq_cxt_emb = self.seq_cxt_emb_layer(self.seq_cxt)
        self.seq = torch.cat([self.seq, self.seq_cxt_emb], -1)
        self.seq = self.feat_emb_layer(self.seq)
        
        # Positional Encoding
        pos_indices = torch.arange(self.input_seq.size(1), device=self.input_seq.device).unsqueeze(0).expand(self.input_seq.size(0), -1)
        t = self.pos_emb_table(pos_indices)
        self.seq += t
        
        # Dropout
        self.seq = self.dropout(self.seq)
        self.seq *= mask
        
        # Transformer
        self.seq = self.transformer(self.seq.transpose(0, 1), self.seq.transpose(0, 1)).transpose(0, 1)
        
        # Layer normalization
        self.seq = F.layer_norm(self.seq, self.seq.size()[1:])
        
        pos = pos.view(-1)
        pos_weight = pos_weight.view(-1)
        neg_weight = neg_weight.view(-1)
        recency = recency.view(-1)
        neg = neg.view(-1)
        
        trgt_cxt = pos_cxt.view(-1, 4)
        trgt_cxt_emb = self.seq_cxt_emb_layer(trgt_cxt)
        
        pos_emb = self.item_emb_table(pos)
        neg_emb = self.item_emb_table(neg)


        pos_emb = torch.cat([pos_emb, trgt_cxt_emb], -1)
        neg_emb = torch.cat([neg_emb, trgt_cxt_emb], -1)
        

        pos_emb = self.feat_emb_layer(pos_emb)
        neg_emb = self.feat_emb_layer(neg_emb)
        
        seq_emb = self.seq.view(-1, self.seq.size(-1))
        
        if test_item is not None and test_item_cxt is not None:
            self.test_item = test_item
            self.test_item_cxt = test_item_cxt
            test_item_cxt_emb = self.seq_cxt_emb_layer(self.test_item_cxt)
            test_item_emb = self.item_emb_table(self.test_item)
            test_item_emb = torch.cat([test_item_emb, test_item_cxt_emb], -1)
            test_item_emb = self.feat_emb_layer(test_item_emb)
            
            self.test_logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))
            self.test_logits = self.test_logits.view(self.input_seq.size(0), self.input_seq.size(1), -1)
            self.test_logits = self.test_logits[:, -1, :]
        
        # prediction layer
        self.pos_logits = (pos_emb * seq_emb).sum(-1)
        self.neg_logits = (neg_emb * seq_emb).sum(-1)
        
        # ignore padding items (0)
        istarget = (pos != 0).float()
        self.loss = torch.sum(
            - torch.log(torch.sigmoid(self.pos_logits) + 1e-24) * pos_weight * istarget -
            torch.log(1 - torch.sigmoid(self.neg_logits) + 1e-24) * neg_weight * istarget
        ) / torch.sum(istarget)
        
        self.auc = torch.sum(
            ((torch.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)
        
        return self.loss, self.auc


    def predict(self, u, seq, item_idx, seq_cxt, test_item_cxt):
        with torch.no_grad():
            u = torch.tensor(u).long().to(self.device)
            seq = torch.tensor(seq).long().to(self.device)
            item_idx = torch.tensor(item_idx).long().to(self.device)
            seq_cxt = torch.tensor(seq_cxt).float().to(self.device)
            test_item_cxt = torch.tensor(test_item_cxt).float().to(self.device)

            mask = (seq != 0).float().unsqueeze(-1)

            self.seq = self.item_emb_table(seq)
            self.seq_cxt_emb = self.seq_cxt_emb_layer(seq_cxt)
            self.seq = torch.cat([self.seq, self.seq_cxt_emb], -1)
            self.seq = self.feat_emb_layer(self.seq)

            pos_indices = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand(seq.size(0), -1)
            t = self.pos_emb_table(pos_indices)
            self.seq += t

            self.seq = self.dropout(self.seq)
            self.seq *= mask

            self.seq = self.transformer(self.seq.transpose(0, 1), self.seq.transpose(0, 1)).transpose(0, 1)
            self.seq = F.layer_norm(self.seq, self.seq.size()[1:])

            seq_emb = self.seq.view(-1, self.seq.size(-1))

            test_item_cxt_emb = self.seq_cxt_emb_layer(test_item_cxt)
            test_item_emb = self.item_emb_table(item_idx)
            test_item_emb = torch.cat([test_item_emb, test_item_cxt_emb], -1)
            test_item_emb = self.feat_emb_layer(test_item_emb)

            test_logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))
            test_logits = test_logits.view(seq.size(0), seq.size(1), -1)
            test_logits = test_logits[:, -1, :]

            return test_logits