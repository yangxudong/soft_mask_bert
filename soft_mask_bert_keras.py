# import os
# os.environ['TF_KERAS'] = '1'
# import tensorflow as tf
import keras
from keras import backend as K
from keras_bert import load_vocabulary, Tokenizer, get_checkpoint_paths, load_model_weights_from_checkpoint
from keras_bert.layers import TokenEmbedding, PositionEmbedding
import json
from data_generator import load_data, convert_to_sample, DataGenerator
import numpy as np
import tqdm

pretrained_path = "/Users/weisu.yxd/Code/bert/chinese_L-12_H-768_A-12"
paths = get_checkpoint_paths(pretrained_path)
token_dict = load_vocabulary(paths.vocab)
mask_id = token_dict.get("[MASK]")

tokenizer = Tokenizer(token_dict)
id2token = {j: i for i, j in token_dict.items()}


def get_model_from_embedding(
        inputs,
        embed_layer,
        transformer_num=12,
        head_num=12,
        feed_forward_dim=3072,
        dropout_rate=0.1,
        attention_activation=None,
        feed_forward_activation='gelu',
        trainable=None,
        output_layer_num=1):
    """Get BERT model.

    See: https://arxiv.org/pdf/1810.04805.pdf
    :param inputs: raw inputs
    :param embed_layer: input embeddings.
    :param transformer_num: Number of transformers.
    :param head_num: Number of heads in multi-head attention in each transformer.
    :param feed_forward_dim: Dimension of the feed forward layer in each transformer.
    :param dropout_rate: Dropout rate.
    :param attention_activation: Activation for attention layers.
    :param feed_forward_activation: Activation for feed-forward layers.
    :param trainable: Whether the model is trainable.
    :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `training` is `False`.
    :return: The built model.
    """
    from keras_transformer import get_encoders, gelu
    from keras_layer_normalization import LayerNormalization
    if attention_activation == 'gelu':
        attention_activation = gelu
    if feed_forward_activation == 'gelu':
        feed_forward_activation = gelu
    if trainable is None:
        trainable = True

    def _trainable(_layer):
        if isinstance(trainable, (list, tuple, set)):
            for prefix in trainable:
                if _layer.name.startswith(prefix):
                    return True
            return False
        return trainable

    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='Embedding-Dropout',
        )(embed_layer)
    else:
        dropout_layer = embed_layer
    embed_layer = LayerNormalization(
        trainable=trainable,
        name='Embedding-Norm',
    )(dropout_layer)
    transformed = get_encoders(
        encoder_num=transformer_num,
        input_layer=embed_layer,
        head_num=head_num,
        hidden_dim=feed_forward_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
    )

    model = keras.models.Model(inputs=inputs, outputs=transformed)
    for layer in model.layers:
        layer.trainable = _trainable(layer)
    if isinstance(output_layer_num, int):
        output_layer_num = min(output_layer_num, transformer_num)
        output_layer_num = [-i for i in range(1, output_layer_num + 1)]
    outputs = []
    for layer_index in output_layer_num:
        if layer_index < 0:
            layer_index = transformer_num + layer_index
        layer_index += 1
        layer = model.get_layer(name='Encoder-{}-FeedForward-Norm'.format(layer_index))
        outputs.append(layer.output)
    if len(outputs) > 1:
        transformed = keras.layers.Concatenate(name='Encoder-Output')(list(reversed(outputs)))
    else:
        transformed = outputs[0]
    return transformed, model


def get_inputs(seq_len):
    """Get input layers.

    See: https://arxiv.org/pdf/1810.04805.pdf

    :param seq_len: Length of the sequence or None.
    """
    names = ['Token', 'Segment', 'Masked']
    return [keras.layers.Input(
        shape=(seq_len,),
        dtype='int32',
        name='Input-%s' % name,
    ) for name in names]


def build_csc_model(max_seq_len, alpha=0.8):
    # build detect model
    with open(paths.config, 'r') as reader:
        config = json.load(reader)
    if max_seq_len is not None:
        config['max_position_embeddings'] = min(max_seq_len, config['max_position_embeddings'])
    seq_len = config["max_position_embeddings"]
    inputs = get_inputs(seq_len)  # [input_ids, segment_ids, input_mask]
    token_num = len(token_dict)
    embed_dim = config["hidden_size"]

    token_embedding_lookup = TokenEmbedding(
        input_dim=token_num,
        output_dim=embed_dim,
        mask_zero=True,
        trainable=True,
        name='Embedding-Token',
    )
    segment_embedding_lookup = keras.layers.Embedding(
        input_dim=2,
        output_dim=embed_dim,
        trainable=True,
        name='Embedding-Segment',
    )
    position_embed_layer = PositionEmbedding(
        input_dim=seq_len,
        output_dim=embed_dim,
        mode=PositionEmbedding.MODE_ADD,
        trainable=True,
        name='Embedding-Position',
    )
    token_emb, embed_weights = token_embedding_lookup(inputs[0])
    seg_emb = segment_embedding_lookup(inputs[1])
    add = keras.layers.Add(name='Embedding-Token-Segment')
    embeddings = position_embed_layer(add([token_emb, seg_emb]))
    # embeddings = keras.layers.Embedding(input_dim=token_num, output_dim=embed_dim, mask_zero=True)(inputs[0])

    mask = K.cast(inputs[2], dtype='bool')
    x = keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True))(embeddings, mask=mask)
    err_prob = keras.layers.Dense(1, activation='sigmoid', name="error_prob")(x)  # shape: (None, seq_len, 1)
    # detect_model = keras.Model(inputs, err_prob)
    # detect_model.summary()

    # build correct model
    char_start_index = 671
    char_end_index = 7992
    num_classes = char_end_index - char_start_index + 2  # add extra id representing the oov original char

    mask_ids = K.constant(mask_id, shape=(1, max_seq_len))
    mask_emb, _ = token_embedding_lookup(mask_ids)
    soft_emb = err_prob * mask_emb + (1. - err_prob) * token_emb  # broadcast, shape(None, seq_len, emb_size)
    new_embeddings = position_embed_layer(add([soft_emb, seg_emb]))

    bert_output, bert = get_model_from_embedding(
        inputs, new_embeddings,
        transformer_num=config['num_hidden_layers'],
        head_num=config['num_attention_heads'],
        feed_forward_dim=config['intermediate_size'],
        feed_forward_activation=config['hidden_act'])
    load_model_weights_from_checkpoint(bert, config, paths.checkpoint)

    output = keras.layers.Dense(num_classes, activation='softmax', name="correct_prob")(bert_output + embeddings)
    # logits = keras.layers.Dense(num_classes)(bert_output)
    # output = keras.layers.Activation('softmax', name="correct_prob")(logits)
    error_prob = err_prob[:, :, 0]  # squeeze
    correct_model = keras.models.Model(inputs, [output, error_prob])
    correct_model.summary()

    # mistake_labels = keras.layers.Input(shape=(seq_len,), dtype='int32', name="mistake_labels")
    # char_labels = keras.layers.Input(shape=(seq_len,), dtype='int32', name="char_labels")
    #
    # mask_float = K.cast(inputs[2], K.floatx())
    # correct_loss = K.sparse_categorical_crossentropy(char_labels, logits, from_logits=True)
    # correct_loss = K.sum(correct_loss * mask_float, axis=1) / K.sum(mask_float, axis=1)
    # correct_loss = K.sum(correct_loss)
    #
    # detect_loss = K.binary_crossentropy(K.cast(mistake_labels, K.floatx()), error_prob)
    # detect_loss = K.sum(detect_loss * mask_float, axis=1) / K.sum(mask_float, axis=1)
    # detect_loss = K.sum(detect_loss)
    #
    # loss = alpha * correct_loss + (1. - alpha) * detect_loss

    # 训练模型
    # train_model = keras.models.Model(
    #     inputs=inputs + [mistake_labels, char_labels],
    #     outputs=[output, error_prob]
    # )
    # train_model.add_loss(loss)
    # train_model.summary()

    # def empty_loss_fn(y_true, y_pred):
    #     return 0.
    #
    # def pred_loss_fn(y_true, y_pred):
    #     return y_pred

    correct_model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                          loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
                          loss_weights=[0.8, 0.2])
    return correct_model, correct_model


SEQ_LEN = 256
learning_rate = 5e-4
min_learning_rate = 1e-4

model, predict_model = build_csc_model(SEQ_LEN)


def extract_items(sample, start=671, end=7992):  # process one by one
    inputs, labels = convert_to_sample(sample, tokenizer, SEQ_LEN, start, end)
    output, err_prob = predict_model.predict(inputs)
    raw_ids, _, mask = inputs
    num_chars = len(mask) - 1
    oov = end - start + 1
    ids = np.argmax(output[0, :, :], axis=-1)  # shape (seq_len, num_classes)
    mistakes = []
    correct_ids = []
    for i in range(1, num_chars):
        if ids[i] == oov:
            correct_ids.append(raw_ids[i])
            if start <= raw_ids[i] <= end:
                mistakes.append({"loc": i, "wrong": id2token.get(raw_ids[i]), "correct": "[OOV]"})
        else:
            correct_id = start + ids[i]
            correct_ids.append(correct_id)
            if correct_id != raw_ids[i]:
                mistakes.append({"loc": i, "wrong": id2token.get(raw_ids[i]), "correct": id2token.get(correct_id)})
    sentence = ''.join([id2token.get(idx) for idx in correct_ids])
    return {"text": sentence, "mistakes": mistakes}


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            model.save_weights('best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))

    def evaluate(self):
        TP, FP, TN, FN = 0, 0, 0, 0
        F = open('dev_pred.json', 'w')
        for d in tqdm(iter(dev_data)):
            pred = extract_items(d)
            positive = "mistakes" in d and d["mistakes"]
            if d["text"] == pred["text"]:
                if positive:
                    TP += 1
                else:
                    TN += 1
            else:
                if positive:
                    FN += 1
                else:
                    FP += 1

            s = json.dumps({
                'text': d['text'],
                'new_text': pred['text'],
                'mistakes': d['mistakes'] if 'mistakes' in d else [],
                'predict': pred['mistakes'] if 'mistakes' in pred else []
            }, ensure_ascii=False, indent=4)
            F.write(s + '\n')
        F.close()
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall


BATCH_SIZE = 128
train_data_file = "/Users/weisu.yxd/Code/info_extract/CSC/Automatic-Corpus-Generation/model/data/sighan/train.sgml"
dev_data_file = "/Users/weisu.yxd/Code/info_extract/CSC/Automatic-Corpus-Generation/model/data/sighan/train15.sgml"

train_data = load_data(train_data_file)
dev_data = load_data(dev_data_file)
train_generator = DataGenerator(train_data, tokenizer, SEQ_LEN, BATCH_SIZE)
evaluator = Evaluate()

if __name__ == '__main__':
    model.fit(train_generator, epochs=5, callbacks=[evaluator])
else:
    model.load_weights('best_model.weights')
