import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .word_embedding import load_word_embeddings
from .common import MLP

from itertools import product
import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_cosine_similarity(names, weights, return_dict=True):
    pairing_names = list(product(names, names))
    normed_weights = F.normalize(weights,dim=1)
    similarity = torch.mm(normed_weights, normed_weights.t())
    if return_dict:
        dict_sim = {}
        for i,n in enumerate(names):
            for j,m in enumerate(names):
                dict_sim[(n,m)]=similarity[i,j].item()
        return dict_sim
    return pairing_names, similarity.to('cpu')

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        return -1.0*b.mean()


class CompCos(nn.Module):

    def __init__(self, dset, args):
        super(CompCos, self).__init__()
        self.args = args
        self.dset = dset

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            pairs = torch.LongTensor(pairs).to(device)
            return attrs, objs, pairs

        # Validation
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)

        # for indivual projections
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.factor = 2

        self.scale = self.args.cosine_scale

        if dset.open_world:
            self.train_forward = self.train_forward_open
            self.known_pairs = dset.train_pairs
            seen_pair_set = set(self.known_pairs)
            mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
            self.seen_mask = torch.BoolTensor(mask).to(device) * 1.

            self.activated = False

            # Init feasibility-related variables
            self.attrs = dset.attrs
            self.objs = dset.objs
            self.possible_pairs = dset.pairs

            self.validation_pairs = dset.val_pairs

            self.feasibility_margin = (1-self.seen_mask).float()
            self.epoch_max_margin = self.args.epoch_max_margin
            self.cosine_margin_factor = -args.margin

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.known_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.known_pairs:
                self.attrs_by_obj_train[o].append(a)

        else:
            self.train_forward = self.train_forward_closed
            self.attrs = dset.attrs
            self.objs = dset.objs

        # Precompute training compositions
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs

        try:
            self.args.fc_emb = self.args.fc_emb.split(',')
        except:
            self.args.fc_emb = [self.args.fc_emb]
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)


        self.image_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=args.relu, num_layers=args.nlayers,
                                  dropout=self.args.dropout,
                                  norm=self.args.norm, layers=layers)

        # Fixed
        self.composition = args.composition

        input_dim = args.emb_dim
        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)

        # init with word embeddings
        if args.emb_init:
            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        # static inputs
        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        # Composition MLP
        self.projection = nn.Linear(input_dim * 2, args.emb_dim)


    def freeze_representations(self):
        print('Freezing representations')
        for param in self.image_embedder.parameters():
            param.requires_grad = False
        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False


    def compose(self, attrs, objs):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        inputs = torch.cat([attrs, objs], 1)
        output = self.projection(inputs)
        output = F.normalize(output, dim=1)
        return output


    def compute_feasibility(self):
        obj_embeddings = self.obj_embedder(torch.arange(len(self.objs)).long().to('cuda'))
        obj_embedding_sim = compute_cosine_similarity(self.objs, obj_embeddings,
                                                           return_dict=True)
        attr_embeddings = self.attr_embedder(torch.arange(len(self.attrs)).long().to('cuda'))
        attr_embedding_sim = compute_cosine_similarity(self.attrs, attr_embeddings,
                                                            return_dict=True)

        feasibility_scores = self.seen_mask.clone().float()
        for a in self.attrs:
            for o in self.objs:
                if (a, o) not in self.known_pairs:
                    idx = self.dset.all_pair2idx[(a, o)]
                    score_obj = self.get_pair_scores_objs(a, o, obj_embedding_sim)
                    score_attr = self.get_pair_scores_attrs(a, o, attr_embedding_sim)
                    score = (score_obj + score_attr) / 2
                    feasibility_scores[idx] = score

        self.feasibility_scores = feasibility_scores
        return feasibility_scores * (1 - self.seen_mask.float())


    def get_pair_scores_objs(self, attr, obj, obj_embedding_sim):
        score = -1.
        for o in self.objs:
            if o!=obj and attr in self.attrs_by_obj_train[o]:
                temp_score = obj_embedding_sim[(obj,o)]
                if temp_score>score:
                    score=temp_score
        return score

    def get_pair_scores_attrs(self, attr, obj, attr_embedding_sim):
        score = -1.
        for a in self.attrs:
            if a != attr and obj in self.obj_by_attrs_train[a]:
                temp_score = attr_embedding_sim[(attr, a)]
                if temp_score > score:
                    score = temp_score
        return score

    def update_feasibility(self,epoch):
        self.activated = True
        feasibility_scores = self.compute_feasibility()
        self.feasibility_margin = min(1.,epoch/self.epoch_max_margin) * \
                                  (self.cosine_margin_factor*feasibility_scores.float().to(device))


    def val_forward(self, x):
        img = x[0]
        img_feats = self.image_embedder(img)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)  # Evaluate all pairs
        obj_embed = self.obj_embedder(self.uniq_objs).permute(1,0)
        attr_embed = self.attr_embedder(self.uniq_attrs).permute(1,0)
        obj_pred = F.softmax(torch.matmul(img_feats,obj_embed),dim=1)
        attr_pred = F.softmax(torch.matmul(img_feats,attr_embed),dim=1)
        #print(obj_pred.size(),attr_pred.size())
        score = torch.matmul(img_feats_normed, pair_embeds)
        #score = torch.bmm(attr_pred.unsqueeze(2), obj_pred.unsqueeze(1)).view(attr_pred.shape[0],-1)

        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            idx = obj_id + attr_id * len(self.dset.objs)
            scores[(attr, obj)] = score[:, idx]
        return score, scores

    def pseudo_forward(self,x):
        img = x[0]
        img_feats = self.image_embedder(img)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)  # Evaluate all pairs
        pair_pred = F.softmax(self.scale*pair_pred,dim=-1)
        reshaped_pred = pair_pred.view(-1,len(self.dset.attrs),len(self.dset.objs))
        obj_pred = reshaped_pred.sum(-2)
        attr_pred = reshaped_pred.sum(-1)
        return obj_pred.detach(),attr_pred.detach()

    def val_forward_with_threshold(self, x, th=0.):
        img = x[0]
        img_feats = self.image_embedder(img)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)  # Evaluate all pairs
        score = torch.matmul(img_feats_normed, pair_embeds)

        # Note: Pairs are already aligned here
        mask = torch.Tensor((np.array(self.feasibility_scores)>=th)*1.0).cuda()
        score = score*mask + (1.-mask)*(-1.)

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]

        return score, scores


    def train_forward_open(self, x):
        img, attrs, objs, pairs,mask  = x[0], x[1], x[2], x[3], x[4]
        img_feats = self.image_embedder(img)

        pair_embed = self.compose(self.train_attrs, self.train_objs).permute(1, 0)
        img_feats_normed = F.normalize(img_feats, dim=1)
        
        pair_pred = torch.matmul(img_feats_normed, pair_embed)

        if self.args.partial == True:
            pair_pred = F.softmax(self.scale*pair_pred,dim=-1)
            reshaped_pred = pair_pred.view(-1,len(self.dset.attrs),len(self.dset.objs))
            obj_pred = reshaped_pred.sum(-2)
            attr_pred = reshaped_pred.sum(-1)
            nl = nn.NLLLoss()
            attr_loss = nl(torch.log(attr_pred[mask==0,:]), attrs[mask==0])
            obj_loss = nl(torch.log(obj_pred[mask==1,:]), objs[mask==1])
            loss = obj_loss + attr_loss

        if self.args.entropy == True and self.args.partial==True:
            hl = HLoss()
            obj_ent = hl(obj_pred)
            attr_ent = hl(attr_pred)
            ent_loss = obj_ent + attr_ent

        if self.args.pseudo == True and self.args.partial==True:
            ssl_thresh = 0.95
            pseudo_obj,psuedo_attr=self.pseudo_forward(x)
            obj_mask = psuedo_obj.max(dim=1)[0]>ssl_thresh
            attr_mask = psuedo_attr.max(dim=1)[0]>ssl_thresh
            u_obj = F.cross_entropy(obj_pred[obj_mask*mask],psuedo_obj.argmax(dim=1)[obj_mask*mask])
            u_attr = F.cross_entropy(attr_pred[attr_mask*(1-mask)],psuedo_attr.argmax(dim=1)[attr_mask*(1-mask)])
            u_loss = u_obj+u_attr

        if self.args.partial == False:
            if self.activated:
               pair_pred += (1 - self.seen_mask) * self.feasibility_margin
               loss_cos = F.cross_entropy(self.scale * pair_pred, pairs)
            else:
               pair_pred = pair_pred * self.seen_mask + (1 - self.seen_mask) * (-10)
               loss_cos = F.cross_entropy(self.scale * pair_pred, pairs)

            loss = F.cross_entropy(self.scale*pair_pred,pairs)


        if self.args.entropy == True and self.args.partial == True:
            loss = loss + ent_loss
        if self.args.pseudo == True and self.args.partial == True:
            loss = loss+u_loss
        return loss, None


    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred



