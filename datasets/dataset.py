import os
import numpy as np
import json
import pickle
import math
import h5py
import torch
from torch.utils.data import Dataset



def invert_dict(d):
    return {v: k for k, v in d.items()}

def load_vocab_glove_matrix(vocab_path, glovept_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    with open(glovept_path, 'rb') as f:
        obj = pickle.load(f)
        glove_matrix = torch.from_numpy(obj['glove']).type(torch.FloatTensor)
    return vocab, glove_matrix



class VideoQADataset(Dataset):
    def __init__(self, question_type, glovept_path, level, visual_m_path, visual_s_path, transform=None):
        self.question_type = question_type
        self.glovept_path = glovept_path
        self.levels = np.power(2,level)-1
        self.visual_m_path = visual_m_path
        self.visual_s_path = visual_s_path
        #load glovefile
        with open(glovept_path, 'rb') as f:
            obj = pickle.load(f)
            self.questions = torch.from_numpy(obj['questions']).type(torch.LongTensor)
            self.questions_len = torch.from_numpy(obj['questions_len']).type(torch.LongTensor)
            self.question_id = obj['question_id']
            self.video_ids = obj['video_ids']
            self.video_names = obj['video_names']
            if self.question_type in ['count']:
                self.answers = torch.from_numpy(np.array(obj['answers'])).type(torch.FloatTensor).unsqueeze(-1)
            else:
                self.answers = torch.from_numpy(np.array(obj['answers'])).type(torch.LongTensor)
            if self.question_type not in ['none', 'frameqa', 'count']:
                self.ans_candidates = torch.from_numpy(np.array(obj['ans_candidates'])).type(torch.LongTensor)
                self.ans_candidates_len = torch.from_numpy(np.array(obj['ans_candidates_len'])).type(torch.LongTensor)
            
    def __getitem__(self, idx):
        with open(os.path.join(self.visual_m_path,self.video_names[idx],'Features.pkl'), 'rb') as fp:
            temp = pickle.load(fp)
        visual_m = torch.from_numpy(temp['Features']).type(torch.FloatTensor)[0:self.levels]#shape: self.levels,2048
        with open(os.path.join(self.visual_s_path,self.video_names[idx],'Features.pkl'), 'rb') as fp:
            temp = pickle.load(fp)
        visual_s = torch.from_numpy(temp['Features']).type(torch.FloatTensor)[0:self.levels]#shape: self.levels,16,2048
        question = self.questions[idx]
        question_len = self.questions_len[idx]
        ans_candidates = torch.zeros(5).long()
        ans_candidates_len = torch.zeros(5).long()
        answer = self.answers[idx]
        if self.question_type not in ['none', 'frameqa', 'count']:
            ans_candidates = self.ans_candidates[idx]
            ans_candidates_len = self.ans_candidates_len[idx]
        return visual_m, visual_s, question, question_len, ans_candidates, ans_candidates_len, answer

    def __len__(self):
        return self.questions.shape[0]



