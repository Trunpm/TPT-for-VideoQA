import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncodingLearned1D(nn.Module):
    """
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncodingLearned1D(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncodingLearned1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        idx = torch.arange(x.shape[0],device=x.device)
        x = x + self.pos_embed(idx).unsqueeze(1)
        return self.dropout(x)

class TransformerEncoderLayer_QKV(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, res_dropout=0.1, activ_dropout=0.1):
        super(TransformerEncoderLayer_QKV,self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,dropout=attn_dropout)
        self.res_dropout = res_dropout
        self.activ_dropout = activ_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(embed_dim, 4*embed_dim)
        self.fc2 = nn.Linear(4*embed_dim, embed_dim)
        self.activ=nn.GELU()
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_key_padding_mask(self, max_length, key_len):
        #return shape (batch, max_length)
        return torch.arange(0,max_length,device=key_len.device).unsqueeze(0).expand(key_len.shape[0],max_length).ge(key_len.unsqueeze(1)).bool()

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x
  
    def forward(self, x_q, x_k=None, key_len=None):
        """
        Args:
            x_q (Tensor): input to the layer of shape (seq_len, batch, embed_dim)
            x_k (None or Tensor): if tensor, input to the layer of shape (seq_len', batch, embed_dim)
            key_len (None or Tensor): if Tensor, input to the layer of shape (batch,)
        Returns:
            out shape (seq_len, batch, embed_dim)
        """
        residual = x_q
        x_q = self.maybe_layer_norm(0, x_q, before=True)
        if x_k is None:
            key_padding_mask = self._get_key_padding_mask(x_q.shape[0],key_len) if key_len is not None else None
            x_q, _ = self.self_attn(query=x_q, key=x_q, value=x_q, key_padding_mask=key_padding_mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True) 
            key_padding_mask = self._get_key_padding_mask(x_k.shape[0],key_len) if key_len is not None else None
            x_q, _ = self.self_attn(query=x_q, key=x_k, value=x_k, key_padding_mask=key_padding_mask)
        x_q = F.dropout(x_q, p=self.res_dropout, training=self.training)
        x_q = residual + x_q
        x_q = self.maybe_layer_norm(0, x_q, after=True)

        residual = x_q
        x_q = self.maybe_layer_norm(1, x_q, before=True)
        x_q = self.activ(self.fc1(x_q))
        x_q = F.dropout(x_q, p=self.activ_dropout, training=self.training)
        x_q = self.fc2(x_q)
        x_q = F.dropout(x_q, p=self.res_dropout, training=self.training)
        x_q = residual + x_q
        x_q = self.maybe_layer_norm(1, x_q, after=True)
        return x_q

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_scale = 1.0
        self.pos_encoder = PositionalEncodingLearned1D(embed_dim)
        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            new_layer = TransformerEncoderLayer_QKV(embed_dim)
            self.layers.append(new_layer)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_q, x_k=None, key_len=None):
        x_q = self.pos_encoder(self.embed_scale*x_q)
        if x_k is not None:
            x_k = self.pos_encoder(self.embed_scale*x_k)
         # encoder layers
        intermediates = [x_q]
        for layer in self.layers:
            x_q = layer(x_q, x_k, key_len)
            intermediates.append(x_q)

        x_q = self.layer_norm(x_q)
        return x_q




class visualembedding(nn.Module):
    def __init__(self, embed_dim=512, v_inDim=2048,proj_v_drop=0.1):
        super(visualembedding, self).__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
                        nn.Linear(v_inDim,embed_dim),
                        nn.GELU(),
                        nn.Dropout(p=proj_v_drop),
                        )

    def forward(self, visual_sm):
        """
        Args:
            visual_sm: [Tensor] (batch_size, levels, 16+1, 2048)
        return:
            visual_embedding: [Tensor] (levels, 16+1, batch_size, embed_dim)
        """
        visual_embedding = self.proj(visual_sm).permute(1,2,0,3)
        return visual_embedding


class textembedding(nn.Module):
    def __init__(self, embed_dim=512, vocab_size=8000,wordvec_dim=300,proj_l_drop=0.1):
        super(textembedding, self).__init__()

        self.embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, embed_dim),
                        nn.GELU(),
                        nn.Dropout(p=proj_l_drop),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim)


    def forward(self, text, text_len):
        text_embedding = self.embed(text)
        text_embedding = self.proj_l(text_embedding).permute(1,0,2)
        text_embedding = self.TransformerEncoder_text(text_embedding, None, text_len)
        return text_embedding






class OutOpenEnded(nn.Module):
    def __init__(self, embed_dim=512, num_answers=1000, drorate=0.1):
        super(OutOpenEnded, self).__init__()

        self.classifier = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim * 2, embed_dim),
                                        nn.GELU(),
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
    def __init__(self, embed_dim=512, drorate=0.1):
        super(OutCount, self).__init__()

        self.regression = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim * 2, embed_dim),
                                        nn.GELU(),
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

        self.classifier = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim * 4, embed_dim),
                                        nn.GELU(),
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

