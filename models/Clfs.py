from mxnet import nd
from mxnet.gluon import nn


class Textcnn(nn.Block):
    def __init__(self, vocab, embed_size, num_channels, pooling_ints, dropout_rate,dense):
        """
        :param vocab: 词汇量
        :param embed_size: 200/300维
        :param num_channels: 多少
        :param pooling_ints: 池化数
        :param dropout_rate: dropout比率
        :param dense: dense层
        """
        super(Textcnn, self).__init__()
        with self.name_scope():
            self.embedding = nn.Embedding(vocab, embed_size)
            self.cnn1 = nn.Conv1D(
                num_channels,
                kernel_size=3,
                padding=2,
                activation='relu')
            self.cnn2 = nn.Conv1D(
                num_channels,
                kernel_size=4,
                padding=3,
                activation='relu')
            self.cnn3 = nn.Conv1D(
                num_channels,
                kernel_size=5,
                padding=4,
                activation='relu')
            self.batchnorm = nn.BatchNorm()
            self.poolings = nn.MaxPool1D(pooling_ints)
            self.dropouts = nn.Dropout(dropout_rate)
            self.flattern = nn.Flatten()
            self.dense = nn.Sequential()
            for each in dense[:-1]:
                self.dense.add(nn.Dense(each, activation="softrelu"))
                self.dense.add(nn.BatchNorm())
                self.dense.add(nn.Dropout(dropout_rate))
            self.dense.add(nn.Dense(dense[-1], activation="sigmoid"))

    def forward(self, X):
        X = self.embedding(X)
        Y_1 = self.poolings(self.batchnorm(self.cnn1(X)))
        Y_2 = self.poolings(self.batchnorm(self.cnn2(X)))
        Y_3 = self.poolings(self.batchnorm(self.cnn3(X)))
        Y_total = nd.concat(Y_1, Y_2, Y_3)
        Y_total = self.flattern(Y_total)
        Y_total = self.dropouts(Y_total)
        outputs = self.dense(Y_total)
        return outputs