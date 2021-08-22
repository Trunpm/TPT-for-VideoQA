# I-aRPT for VideoQA
This is the PyTorch Implementation of our paper "[Interaction-aware Residual Pyramid Transformer for Video Question Answering](https:)".

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
  1. To extract question or answers Glove Embedding, please ref [here](https://github.com/thaolmk54/hcrn-videoqa)
  2. To extract appearance and motion feature, use the pretrained models [here](https://drive.google.com/open?id=1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4)
  3. after step i and ii, we will get some files in fold /datasets.

# train and test
run `python main.py` in your terminal.
