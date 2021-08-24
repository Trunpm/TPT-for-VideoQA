# TPT for VideoQA
This is the PyTorch Implementation of our paper "[Temporal Pyramid Transformer with Multimodal Interaction forVideo Question Answering](https:)".

![alt text](docs/fig2.png 'overview of the network')

# Platform and dependencies
Ubuntu 14.04  Python 3.7  CUDA10.1  CuDNN7.5+  
pytorch==1.7.0

# Data Preparation
* Download the dataset  
  MSVD-QA: [link](https://github.com/xudejing/video-question-answering)   
  MSRVTT-QA: [link](https://github.com/xudejing/video-question-answering)   
  TGIF-QA: [link](https://github.com/YunseokJANG/tgif-qa)   
* Preprocessing
  1. To extract question or answers Glove Embedding, please ref [here](https://github.com/thaolmk54/hcrn-videoqa).  
  for MSRVTT dataset, we have features at the path /datasets:  
  `MSRVTT/word/MSRVTT_test_questions.pt`  
  `MSRVTT/word/MSRVTT_train_questions.pt`  
  `MSRVTT/word/MSRVTT_val_questions.pt`  
  `MSRVTT/word/MSRVTT_vocab.json`  
  2. To extract appearance and motion feature, use the pretrained models [here](https://drive.google.com/open?id=1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4).  
  for MSRVTT dataset, we have features at the path /datasets:  
  `MSRVTT/pyramid/SpatialFeatures/video0/Features.pkl` (shape is 2^level-1,16,2048)  
  `MSRVTT/pyramid/SpatialFeatures/video1/Features.pkl`  
  `...`  
  `MSRVTT/pyramid/TemporalFeatures/video0/Features.pkl` (shape is 2^level-1,2048)  
  `MSRVTT/pyramid/TemporalFeatures/video1/Features.pkl`  
  `...`  
  
# train and test
run `python train.py` in your terminal.  some trained models will be upload...
