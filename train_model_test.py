import pandas as pd
import jieba
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

ctx = mx.cpu()
batch_size = 128

df = pd.read_excel(r'C:\Users\15581\Documents\上海电信\20_04\datas_211_0407_jieba.xlsx')
print('1---读取数据成功')


# 标签表预处理(y值即label的映射，label的数量) eg：'你好':1  1:'你好'
label = list(set(df['concat'].tolist()))
dig_lables = dict(enumerate(label))
lable_dig = dict((lable, dig) for dig, lable in dig_lables.items())
print('2:y值处理成功类别共计***', len(lable_dig))

df['label'] = df['concat'].apply(lambda lable: lable_dig[lable])

index, weight, default, pad, bos, eos = get_text_embedding("output_vector_file.txt", "<pad>", "<bos>", "<eos>")
embed_size = weight.shape[1]
vocab_size = weight.shape[0]
print('3:导入词向量')

data = Dataset(
    df["新文本分词"].to_list(),
    df["label"].to_list(),
    index,
    default,
    pad,
    eos,
    0,
    200,
    test_size=0.1)

print('4:导入文本词向量的token')
# import pdb
# pdb.set_trace()

train_data = mx.io.NDArrayIter(
    data=[
        data.train_x,
        data.train_y],
    batch_size=batch_size,
    shuffle=True,
    last_batch_handle="roll_over")
valid_data = mx.io.NDArrayIter(
    data=[
        data.valid_x,
        data.valid_y],
    batch_size=batch_size,
    shuffle=True,
    last_batch_handle="pad")

# (self, vocab, embed_size, head, unit, num_channels, pooling_ints, dropout_rate, dense)
model = Textcnn_attention(vocab_size, embed_size,2, 100, 200, 100, 0.4, [512, 238])
model.collect_params().initialize(mx.init.Xavier(rnd_type="gaussian"), ctx=ctx)
model.embedding.weight.set_data(weight)
model.embedding.collect_params().setattr('grad_req', 'null')


mask = False
num_epochs = 10
lr = 0.001
start_time = time.time()
# best_acc = 0
# best_f1 = 0
# best_lost = 999
# _max_round = 10
# max_round = _max_round
pad_id = index[pad]
loss = gloss.SoftmaxCrossEntropyLoss(sparse_label=False)
for epoch in range(num_epochs):
    # data reset
    train_data.reset()
    valid_data.reset()
    # Epoch training stats
    epoch_L = 0.0
    epoch_sent_num = 0
    if epoch % 2:
        trainer_name = "adagrad"
        trainer = Trainer(
            model.collect_params(), trainer_name, {
                'learning_rate': lr, "wd": 0.001})
    else:
        trainer_name = "sgd"
        trainer = Trainer(
            model.collect_params(), trainer_name, {
                'learning_rate': lr, "wd": 0.001, "momentum": 0.8})
    for batch in train_data:
        x = batch.data[0].as_in_context(ctx)
        y = batch.data[1].as_in_context(ctx)
        with autograd.record(train_mode=True):
            if mask:
                _mask = nd.not_equal(x, index[pad])
                pred = model(_mask)
            else:
                pred = model(x)
            bl = loss(pred, nd.one_hot(y, len(lable_dig))).as_in_context(ctx)
#             import pdb
#             pdb.set_trace()
            bl.backward()
        trainer.step(batch_size)
        epoch_L += nd.sum(bl).asscalar()
    t_l, t_acc = evaluate(model, valid_data, ctx)
    model.save_parameters("clf_mxnet.params")
    msg = '[Epoch {}] , valid acc {:.6f}, valid avg loss {:.6f} with {}'.format(
        epoch, t_acc, t_l, trainer_name)
    print(msg)
    print(model)