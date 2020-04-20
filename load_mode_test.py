import pandas as pd
import jieba
import re
from utils import get_text_embedding, seg_sentences, clear_dum
# from tqdm import tqdm
# from tqdm import tqdm_notebook
from mxnet.gluon import loss as gloss, Trainer
from mxnet import autograd
import time
import mxnet as mx
from train_clfs import Dataset, evaluate
from models.Clfs import Textcnn, Textcnn_attention
from mxnet import nd
from gensim.models import Word2Vec
print('导入模块成功')

# (self, vocab, embed_size, head, unit, num_channels, pooling_ints, dropout_rate, dense)
models = Textcnn_attention(7375, 250,2, 100, 200, 90, 0.4, [512, 238])

