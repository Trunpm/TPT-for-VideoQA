import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from .modules.TransformerEncoders import *



class visualembedding(nn.Module):
    def __init__(self, embed_dim=512, activation='gelu', v_inDim=2048,proj_v_drop=0.1):
        super(visualembedding, self).__init__()
        self.embed_dim = embed_dim
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()
        self.proj_m = nn.Sequential(
                        nn.Linear(v_inDim,embed_dim),
                        self.activ,
                        nn.Dropout(p=proj_v_drop),
                        )
        self.proj_s = nn.Sequential(
                        nn.Linear(v_inDim,embed_dim),
                        self.activ,
                        nn.Dropout(p=proj_v_drop),
                        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_m, visual_s):
        """
        Args:
        	visual_m: [Tensor] (batch_size, levels, 2048)
            visual_s: [Tensor] (batch_size, levels, 16, 2048)
        return:
            visual_embedding: [Tensor] (levels, 1+16, batch_size, embed_dim)
        """
        visual_embedding_m = self.proj_m(visual_m).permute(1,0,2)
        visual_embedding_s = self.proj_s(visual_s).permute(1,2,0,3)
        visual_embedding = torch.cat([visual_embedding_m.unsqueeze(1),visual_embedding_s],dim=1)
        return visual_embedding


class textembedding(nn.Module):
    def __init__(self, embed_dim=512, activation='gelu', vocab_size=8000,wordvec_dim=300,proj_l_drop=0.1, pos_flag='learned',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,num_layers=6):
        super(textembedding, self).__init__()
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()

        self.embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, embed_dim),
                        self.activ,
                        nn.Dropout(p=proj_l_drop),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim, pos_flag,pos_dropout,num_heads,attn_dropout,res_dropout,activ_dropout,activation,num_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, text, text_len):
        """
        Args:
            text: [Tensor] (batch_size, max_text_length)
            text_len: [Tensor] (batch_size,)
        return:
            text_embedding: [Tensor] (max_text_length, batch_size, embed_dim)
        """
        text_embedding = self.embed(text)
        text_embedding = self.proj_l(text_embedding).permute(1,0,2)
        text_embedding = self.TransformerEncoder_text(text_embedding, None, text_len)
        return text_embedding