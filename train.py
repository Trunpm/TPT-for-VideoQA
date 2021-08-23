import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn

###Data require
import sys
sys.path.append('../')
import argparse
from datasets.dataset import load_vocab_glove_matrix, VideoQADataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

###Model require
from models.MainModel import mainmodel

parser = argparse.ArgumentParser('net')
parser.add_argument('--seed', type=int, default=1)
# ========================= Data Configs ==========================
parser.add_argument('--vocab_path', type=str, default='./datasets/MSRVTT/word/MSRVTT_vocab.json')
parser.add_argument('--question_type', type=str, default='none', help='none | frameqa | count | action | transition')
parser.add_argument('--glovept_path_train', type=str, default='./datasets/MSRVTT/word/MSRVTT_train_questions.pt')
parser.add_argument('--glovept_path_val', type=str, default='./datasets/MSRVTT/word/MSRVTT_val_questions.pt')
parser.add_argument('--glovept_path_test', type=str, default='./datasets/MSRVTT/word/MSRVTT_test_questions.pt')
parser.add_argument('--visual_m_path', type=str, default='./datasets/MSRVTT/pyramid/TemporalFeatures', help='')
parser.add_argument('--visual_s_path', type=str, default='./datasets/MSRVTT/pyramid/SpatialFeatures', help='')
parser.add_argument('--transform', type=bool, default=None)
parser.add_argument('--batch_size', type=int, default=64)
#========================= Model Configs ==========================
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--activation', type=str, default='gelu', help='relu | prelu | elu | gelu')
parser.add_argument('--v_inDim', type=int, default=2048)
parser.add_argument('--proj_v_drop', type=float, default=0.1)
parser.add_argument('--wordvec_dim', type=int, default=300)
parser.add_argument('--proj_l_drop', type=float, default=0.1)
parser.add_argument('--pos_flag', type=str, default='sincos', help='learned | sincos')
parser.add_argument('--pos_dropout', type=float, default=0.0)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--attn_dropout', type=float, default=0.0)
parser.add_argument('--res_dropout', type=float, default=0.3)
parser.add_argument('--activ_dropout', type=float, default=0.1)
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--level', type=int, default=3)
parser.add_argument('--num_layers_py', type=int, default=3)
parser.add_argument('--drorate', type=float, default=0.1)
parser.add_argument('--device_ids', type=int, nargs='+', default=[0])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--factor', type=float, default=0.5)
parser.add_argument('--maxepoch', type=int, default=50)
args = parser.parse_args()
vocab, glove_matrix = load_vocab_glove_matrix(args.vocab_path, args.glovept_path_train)
if args.question_type in ['none', 'frameqa']:
    args.vocab_size = len(vocab['question_token_to_idx'])
    args.num_answers = len(vocab['answer_token_to_idx'])
elif args.question_type in ['count']:
    args.vocab_size = len(vocab['question_token_to_idx'])
else:
    args.vocab_size = len(vocab['question_answer_token_to_idx'])
args.savepath = os.path.join('Result',args.vocab_path.split('/')[2], args.question_type)+'/'




###Data read
train_dataset = VideoQADataset(question_type=args.question_type, glovept_path=args.glovept_path_train, level=args.level, visual_m_path=args.visual_m_path, visual_s_path=args.visual_s_path, transform=args.transform)
train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last=True)
val_dataset = VideoQADataset(question_type=args.question_type, glovept_path=args.glovept_path_val, level=args.level, visual_m_path=args.visual_m_path, visual_s_path=args.visual_s_path, transform=None)
val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=4,drop_last=False)
test_dataset = VideoQADataset(question_type=args.question_type, glovept_path=args.glovept_path_test, level=args.level, visual_m_path=args.visual_m_path, visual_s_path=args.visual_s_path, transform=None)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4,drop_last=False)


###Model
model = mainmodel(args).cuda(args.device_ids[0])
glove_matrix = glove_matrix.cuda(args.device_ids[0])
with torch.no_grad():
    model.textembedding.embed.weight.set_(glove_matrix)
torch.backends.cudnn.benchmark = True
if len(args.device_ids)>1:
    model = nn.DataParallel(model,device_ids=args.device_ids)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
if args.question_type in ['none', 'frameqa']:
    criterion = nn.CrossEntropyLoss().cuda(args.device_ids[0])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=args.patience,factor=args.factor,verbose=True)
elif args.question_type in ['count']:
    criterion = nn.MSELoss().cuda(args.device_ids[0])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=args.patience,factor=args.factor,verbose=True)
else:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=args.patience,factor=args.factor,verbose=True)


###train val test
# Credit https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(args.device_ids[0])
    return torch.index_select(a, dim, order_index)

if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)
argsDict = args.__dict__
with open(args.savepath + 'params.txt', 'a') as f:
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
max_acc = 0.0
min_mae = float('Inf')
patience = 0
for epoch in range(args.maxepoch):
    model.train()
    total_loss = 0.0
    for i, batch_data in enumerate(train_loader):
        model.zero_grad()
        *batch_input, batch_answer = [Variable(x.cuda(args.device_ids[0])) for x in batch_data]
        output = model(*batch_input)
        if args.question_type in ['none', 'frameqa','count']:
            loss = criterion(output, batch_answer)
            loss.backward()
            total_loss += loss.data.cpu()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
            optimizer.step()
        else:
            expan_idx = np.concatenate(np.tile(np.arange(args.batch_size).reshape([args.batch_size, 1]), [1, 5])) * 5
            answers_agg = tile(batch_answer, 0, 5)
            loss = torch.max(torch.tensor(0.0).cuda(args.device_ids[0]),
                            1.0 + output - output[answers_agg + torch.from_numpy(expan_idx).cuda(args.device_ids[0])])
            loss = loss.sum()
            loss.backward()
            total_loss += loss.data.cpu()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
            optimizer.step()
        # print("Running Training loss: {}".format(loss.data.cpu()))
    with open(args.savepath + 'log.txt', 'a') as out_file:
        out_file.write("Epoch {} complete! Total Training loss: {}".format(epoch, total_loss / len(train_loader))+'\n')

    model.eval()
    correct = 0.0
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            *batch_input, batch_answer = [Variable(x.cuda(args.device_ids[0])) for x in batch_data]
            output = model(*batch_input)
            if args.question_type in ['none', 'frameqa']:
                _,pred = torch.max(output, 1)
                correct += torch.sum(pred == batch_answer)
            elif args.question_type in ['count']:
                output = (output + 0.5).long().clamp(min=1, max=10)
                correct += (output - batch_answer) ** 2
            else:
                preds = torch.argmax(output.view(1, 5), dim=1)
                correct += torch.sum(preds == batch_answer)
        with open(args.savepath + 'log.txt', 'a') as out_file:
            out_file.write("Epoch {} complete!, val correct is: {}".format(epoch, float(correct)/len(val_loader))+'\n')
    scheduler.step(correct)

    model.eval()
    correct = 0.0
    correct6 = 0.0
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            *batch_input, batch_answer = [Variable(x.cuda(args.device_ids[0])) for x in batch_data]
            output = model(*batch_input)
            if args.question_type in ['none', 'frameqa']:
                _,pred = torch.max(output, 1)
                correct += torch.sum(pred == batch_answer)
            elif args.question_type in ['count']:
                output6 = (output + 0.5).long().clamp(min=1, max=10)
                correct6 += (output6 - batch_answer) ** 2
            else:
                preds = torch.argmax(output.view(1, 5), dim=1)
                correct += torch.sum(preds == batch_answer)
        if args.question_type in ['count']:
            with open(args.savepath + 'log.txt', 'a') as out_file:
                out_file.write("Epoch {} complete!, test correct is: {}".format(epoch, float(correct6)/len(test_loader))+'\n')
        else:
            with open(args.savepath + 'log.txt', 'a') as out_file:
                out_file.write("Epoch {} complete!, test correct is: {}".format(epoch, float(correct)/len(test_loader))+'\n')
        
        
    ###savepath
    if args.question_type in ['none', 'frameqa']:
        if float(correct)/len(test_loader) > max_acc:
            max_acc = float(correct)/len(test_loader)
            torch.save(model.state_dict(), args.savepath +'epoch' + str(epoch) + '.pt')
            patience=0
        else:
            patience+=1
    elif args.question_type in ['count']:
        if float(correct6)/len(test_loader) < min_mae:
            min_mae = float(correct6)/len(test_loader)
            torch.save(model.state_dict(), args.savepath +'epoch' + str(epoch) + '.pt')
            patience=0
        else:
            patience+=1
    else:
        if float(correct)/len(test_loader) > max_acc:
            max_acc = float(correct)/len(test_loader)
            torch.save(model.state_dict(), args.savepath +'epoch' + str(epoch) + '.pt')
            patience=0
        else:
            patience+=1
    if patience>=4*args.patience:
        break