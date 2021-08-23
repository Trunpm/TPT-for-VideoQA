import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F



class OutOpenEnded(nn.Module):
    def __init__(self, embed_dim=512, num_answers=1000, drorate=0.1, activation='gelu'):
        super(OutOpenEnded, self).__init__()
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()

        self.classifier = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim * 2, embed_dim),
                                        self.activ,
                                        nn.BatchNorm1d(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, num_answers))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, question_embedding_v, visual_embedding_qu, question_len):
        question_embedding_v = torch.stack([question_embedding_v[0:question_len[j],j,:].mean(dim=0) for j in range(question_len.shape[0])],dim=0)
        visual_embedding_qu = visual_embedding_qu.mean(dim=0)
        out = torch.cat([question_embedding_v, visual_embedding_qu], 1)
        out = self.classifier(out)
        return out



class OutCount(nn.Module):
    def __init__(self, embed_dim=512, drorate=0.1, activation='gelu'):
        super(OutCount, self).__init__()
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()

        self.regression = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim * 2, embed_dim),
                                        self.activ,
                                        nn.BatchNorm1d(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, question_embedding_v, visual_embedding_qu, question_len):
        question_embedding_v = torch.stack([question_embedding_v[0:question_len[j],j,:].mean(dim=0) for j in range(question_len.shape[0])],dim=0)
        visual_embedding_qu = visual_embedding_qu.mean(dim=0)
        out = torch.cat([question_embedding_v, visual_embedding_qu], 1)
        out = self.regression(out)
        return out



class OutMultiChoices(nn.Module):
    def __init__(self, embed_dim=512, drorate=0.1, activation='gelu'):
        super(OutMultiChoices, self).__init__()
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()

        self.classifier = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim * 4, embed_dim),
                                        self.activ,
                                        nn.BatchNorm1d(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, question_embedding_v, visual_embedding_qu, question_len, answer_embedding_v_expand, visual_embedding_an_expand, answers_len):
        question_embedding_v = torch.stack([question_embedding_v[0:question_len[j],j,:].mean(dim=0) for j in range(question_len.shape[0])],dim=0)
        visual_embedding_qu = visual_embedding_qu.mean(dim=0)

        answer_embedding_v_expand = torch.stack([answer_embedding_v_expand[0:answers_len[j],j,:].mean(dim=0) for j in range(answers_len.shape[0])],dim=0)
        visual_embedding_an_expand = visual_embedding_an_expand.mean(dim=0)
        
        out = torch.cat([question_embedding_v, visual_embedding_qu,  answer_embedding_v_expand, visual_embedding_an_expand], 1)
        out = self.classifier(out)
        return out