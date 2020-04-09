import time
from datetime import timedelta

import mxnet as mx
from mxnet import nd

from utils.tokenization import BasicTokenizer


def get_text_embedding(embedding_path, pad, bos, eos):
    """
    read text embedding vector file, such as Glove/FastText
    :param embedding_path: String
    file path
    :param pad: String
    PAD String
    :param bos: String
    BOS String
    :param eos: String
    EOS String
    :return: Index of embedding matrix, weight matrix, unknown, pad, bos, eos
    """
    embedding = mx.contrib.text.embedding.CustomEmbedding(embedding_path,
                                                          elem_delim=' ',
                                                          encoding='utf8',
                                                          vocabulary=None)
    embedding_index = embedding.token_to_idx
    weight = embedding.idx_to_vec
    dim = weight.shape[-1]
    for each in embedding_index:
        embedding_index[each] = embedding_index[each] + 1
    embedding_index[pad] = 0
    weight = nd.concat(nd.zeros((1, dim)), weight, dim=0)
    embedding_index[bos] = len(embedding_index)
    embedding_index[eos] = len(embedding_index)
    weight = nd.concat(weight, nd.random.normal(shape=(2, dim)), dim=0)
    return embedding_index, weight, embedding.unknown_token, pad, bos, eos


def get_time_dif(start_time):
    """
    Return the time used since start_time.
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def try_gpu(gpu=False):
    """
    If GPU is available, return mx.gpu(0); else return mx.cpu()
    :param gpu: use GPU or not
    :return:
    """
    try:
        if not gpu:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu()
            _ = mx.nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def process_one_seq(seq_tokens, max_seq_len, PAD, BOS=None, EOS=None):
    """
    padding seq to same size
    :param seq_tokens: List
    List of tokens in inputs list
    :param max_seq_len: Int
    max length for padding or cut
    :param PAD: padding item
    :param BOS: Begin item
    :param EOS: End item
    :return: List
    List after padding or cut
    """
    if BOS is not None:
        seq_tokens = [BOS] + seq_tokens[:max_seq_len - 1]
    if EOS is not None:
        add = [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    else:
        add = [PAD] * (max_seq_len - len(seq_tokens))
    seq_tokens += add
    return seq_tokens[0: max_seq_len]
