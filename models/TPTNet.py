import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from Utils import *




class TransformerEncoder_v2(nn.Module):
    def __init__(self, embed_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            new_layer = TransformerEncoderLayer_QKV(embed_dim)
            self.layers.append(new_layer)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_q, x_k=None, key_len=None):
        intermediates = [x_q]
        for layer in self.layers:
            x_q = layer(x_q, x_k, key_len)
            intermediates.append(x_q)
        x_q = self.layer_norm(x_q)
        return x_q



class pyramidhierarchical_TD_BU(nn.Module):
    def __init__(self, level, embed_dim):
        super(pyramidhierarchical_TD_BU, self).__init__()
        self.level = level
        self.embed_dim = embed_dim
        self.embed_scale = 1.0
        self.pos_encoder = PositionalEncodingLearned1D(embed_dim)

        self.TransformerEncoderLayer_TD = nn.ModuleList([])
        for i in range(self.level):
            self.TransformerEncoderLayer_TD.append(
                TransformerEncoder_v2(embed_dim),
            )

        self.TransformerEncoderLayer_BU = nn.ModuleList([])
        self.TransformerEncoderLayer_Inter = nn.ModuleList([])
        for i in range(self.level):
            self.TransformerEncoderLayer_BU.append(
                TransformerEncoder_v2(embed_dim),
            )
            if i>0:
                self.TransformerEncoderLayer_Inter.append(
                    TransformerEncoder_v2(embed_dim),
                )

    def forward(self, visual_embedding, text_embedding, text_len):
        """
        Args:
            visual_embedding: [Tensor] (levels, 1+16, batch_size, embed_dim)
            text_embedding: [Tensor] (max_question_length, batch_size, embed_dim)
            text_len: [None or Tensor], if a tensor shape is (batch_size,)
        return:
            text_embedding_v: [Tensor] (max_question_length, batch_size, embed_dim)
            visual_embedding_te[Tensor] (T, batch_size, embed_dim)
        """
        for i in range(self.level):
            v_level = visual_embedding[np.power(2,i)-1:np.power(2,i+1)-1].reshape(-1,visual_embedding.shape[2],visual_embedding.shape[3])
            if self.pos_flag is not None:
                v_level = self.pos_encoder(self.embed_scale*v_level)
            if i==0:
                text_embedding_v = self.TransformerEncoderLayer_TD[i](text_embedding,v_level) + text_embedding
            elif i>0:
                text_embedding_v = self.TransformerEncoderLayer_TD[i](text_embedding_v,v_level) + text_embedding_v

        for i in range(self.level-1,-1,-1):
            v_level = visual_embedding[np.power(2,i)-1:np.power(2,i+1)-1].reshape(-1,visual_embedding.shape[2],visual_embedding.shape[3])
            if self.pos_flag is not None:
                v_level = self.pos_encoder(self.embed_scale*v_level)
            if i==self.level-1:
                visual_embedding_te = self.TransformerEncoderLayer_BU[i](v_level,text_embedding_v,text_len) + v_level
            elif i<self.level-1:
                v_level = self.TransformerEncoderLayer_Inter[i](v_level,visual_embedding_te)
                visual_embedding_te = self.TransformerEncoderLayer_BU[i](v_level,text_embedding_v,text_len) + v_level

        return text_embedding_v, visual_embedding_te













class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.question_type = args.question_type
        
        self.visualembedding = visualembedding(args.embed_dim, args.v_inDim,args.proj_v_drop)
        self.textembedding = textembedding(args.embed_dim,  args.vocab_size,args.wordvec_dim,args.proj_l_drop)

        self.Multipyramidhierarchical = pyramidhierarchical_TD_BU(args.level, args.embed_dim)
        
        if self.question_type in ['none', 'frameqa']:
            self.outlayer = OutOpenEnded(args.embed_dim, args.num_answers)
        elif self.question_type in ['count']:
            self.outlayer = OutCount(args.embed_dim)
        else:
            self.outlayer = OutMultiChoices(args.embed_dim)
       

    def forward(self, visual_sm, question, question_len, answers, answers_len):
        """
        Args:
            visual_sm: [Tensor] (batch_size, levels, 16+1, 2048)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [None or Tensor], if a tensor shape is (batch_size,)
            answers: [Tensor] (batch_size, 5, max_answers_length)
            answers_len: [Tensor] (batch_size, 5)
        return: 
            question_embedding_v: [Tensor] (max_question_length, batch_size, embed_dim)
            visual_embedding_qu: [Tensor] (16, batch_size, embed_dim)
            question_embedding: [Tensor] (max_question_length, batch_size, embed_dim)
            question_len: [None or Tensor], if a tensor shape is (batch_size,)

        """
        visual_embedding = self.visualembedding(visual_m, visual_s)
        question_embedding = self.textembedding(question, question_len)

        question_embedding_v, visual_embedding_qu = self.Multipyramidhierarchical(visual_embedding, question_embedding, question_len)

        if self.question_type in ['none', 'frameqa', 'count']:
            out = self.outlayer(question_embedding_v, visual_embedding_qu, question_len)
        else:
            answer_embedding_v_list = []
            visual_embedding_an_list = []
            for i in range(5):
                answer_embedding = self.textembedding(answers[:,i,:], answers_len[:,i])
                answer_embedding_v, visual_embedding_an = self.Multipyramidhierarchical(visual_embedding, answer_embedding, answers_len[:,i])
                answer_embedding_v_list.append(answer_embedding_v)
                visual_embedding_an_list.append(visual_embedding_an)
            answers_len = answers_len.view(-1)
            answer_embedding_v_expand = torch.stack(answer_embedding_v_list,dim=2).reshape(answer_embedding_v.shape[0],-1,answer_embedding_v.shape[-1])
            visual_embedding_an_expand = torch.stack(visual_embedding_an_list,dim=2).reshape(visual_embedding_an.shape[0],-1,visual_embedding_an.shape[-1])

            expan_idx = np.reshape(np.tile(np.expand_dims(np.arange(question_embedding.shape[1]), axis=1), [1, 5]), [-1])
            
            out = self.outlayer(question_embedding_v[:,expan_idx,:], visual_embedding_qu[:,expan_idx,:], question_len[expan_idx],  answer_embedding_v_expand, visual_embedding_an_expand, answers_len)
        return out