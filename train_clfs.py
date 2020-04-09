from mxnet import nd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from utils import process_one_seq
from mxnet.gluon import loss as gloss


class Dataset(object):
    def __init__(self, x, y, index, default, pad, eos, no_mean, max_seq, test_size=0.4):
        """
        Simple object to process the input Dataset for train
        :param x: list of (list of tokens)
        :param y: list of (list of tag)
        :param index: token index
        :param default: default value in index
        :param pad: pad token in index
        :param eos: eos token in index
        :param no_mean: the value without any mean in Y
        :param max_seq: max size available in X & Y
        """
        self.seq_length = max_seq
        x = [[index.get(i, index[default]) for i in j] for j in x]
        x = [process_one_seq(i, max_seq, index[pad], EOS=index[eos]) for i in x]
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            x, y, test_size=test_size, shuffle=True)
        self.test_x, self.valid_x, self.test_y, self.valid_y = train_test_split(
            self.test_x, self.test_y, test_size=0.5, shuffle=True)
        print('Train/Valid : %d/%d' % (len(self.train_y), len(self.test_y) * 2))
        print('Sentence max words: {shape}'.format(shape=self.seq_length))

    def __str__(self):
        return "Training: {}, Testing: {}, Sequences Length: {}".format(
            len(self.train_x), len(self.test_x), self.seq_length)


def evaluate(model, dataIterator, ctx, pad=None):
    """
    The Evaluation function
    :param model: model object
    :param dataIterator: data iterator in mxnet
    :param ctx: context
    :param weight: NDArray weight matrix of Weighted SCE
    :param pad: Int
    padding id
    :param report: Boolean
    F1 Score report Matrix
    :return:
    """
    loss = gloss.SoftmaxCrossEntropyLoss(sparse_label=False)
    if pad is not None:
        mask = True
    else:
        mask = False
    dataIterator.reset()
    total_loss = 0.0
    total_sample_num = 0
    y_pred, y_true = [], []
    for i, batch in enumerate(dataIterator):
        x = batch.data[0].as_in_context(ctx)
        y = batch.data[1].as_in_context(ctx)
        if mask:
            _mask = nd.not_equal(x, pad)
            pred = model(_mask)
        else:
            pred = model(x)
        bl = loss(pred, nd.one_hot(y,238)).as_in_context(ctx)
        total_sample_num = x.shape[0]
        total_loss += nd.sum(bl).asscalar()
    pred = nd.argmax(pred, axis=1)
    y_pred.extend(pred.asnumpy().tolist())
    y_true.extend(y.asnumpy().tolist())
    acc = metrics.accuracy_score(y_pred, y_true)
#     f1 = metrics.f1_score(y_pred, y_true, average='macro')
    avg_L = total_loss / float(total_sample_num)
#     if report:
#         return avg_L, acc, f1, metrics.classification_report(y_true, y_pred)
#     else:
#         return avg_L, acc, f1
    return avg_L, acc