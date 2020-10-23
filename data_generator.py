import keras
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def convert_to_sample(data, tokenizer, max_len, start=670, end=7991):
    text = data["text"]
    chars = list(text)
    for mistake in data["mistakes"]:
        index = int(mistake["loc"]) - 1
        chars[index] = mistake["correct"]
    correct = ''.join(chars)
    indices, segments = tokenizer.encode(first=text, max_len=max_len)
    mask = [1 if idx > 0 else 0 for idx in indices]

    new_indices, _ = tokenizer.encode(first=correct, max_len=max_len)
    mistake_labels = [0 if indices[i] == new_indices[i] else 1 for i in range(len(indices))]

    oov = end - start + 1
    char_labels = [idx - start if start <= idx <= end else oov for idx in new_indices]
    return [indices, segments, mask], [mistake_labels, char_labels]


def load_data(input_file):
    contents = []
    passages = []
    reader = open(input_file)
    text = ''
    line = reader.readline()
    while line:
        line = line.strip()
        if line.startswith("</SENTENCE>"):
            passages.append(line)
            sentence = ET.fromstringlist(passages)
            if not text:
                passages.clear()
                continue
            # text = sentence.findtext("TEXT")
            content = {"text": text, "mistakes": []}
            for mistake in sentence.iter("MISTAKE"):
                wrong = mistake.findtext("WRONG")
                correct = mistake.findtext("CORRECTION")
                if wrong == correct:
                    continue
                reform = {"wrong": wrong, "correct": correct, "loc": mistake.findtext("LOCATION")}
                content["mistakes"].append(reform)
            if len(content["mistakes"]) > 0:
                contents.append(content)
            passages = []
            text = ''
        elif line.startswith("<TEXT>"):
            text = line[len('<TEXT>'):-len('</TEXT>')]
        elif line:
            passages.append(line)
        line = reader.readline()
    reader.close()
    num = len(contents)
    print(f'{input_file} has loaded, total {num} records')
    return contents


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, tokenizer, max_len=256, batch_size=32):
        self.data = []
        for sample in data:
            if 'text' not in sample:
                continue
            inputs, labels = convert_to_sample(sample, tokenizer, max_len)
            if None in (inputs, labels):
                continue
            self.data.append((inputs, labels))
        num_samples = len(self.data)
        print(f'load total {num_samples} samples')
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        # if len(self.data) % self.batch_size != 0:
        #     self.steps += 1
        self.dim = max_len
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_input_ids = np.empty((self.batch_size, self.dim), dtype=np.int32)
        batch_segment_ids = np.empty((self.batch_size, self.dim), dtype=np.int32)
        batch_input_masks = np.empty((self.batch_size, self.dim), dtype=np.int32)
        batch_mistake_labels = np.empty((self.batch_size, self.dim), dtype=np.float32)
        batch_char_labels = np.empty((self.batch_size, self.dim), dtype=np.int32)
        # sample_weights = []
        for i, index in enumerate(indexes):
            inputs, labels = self.data[index]
            # weight = 1.0 / sum(inputs[2])
            # sample_weights.append(weight)
            batch_input_ids[i, ] = np.array(inputs[0], dtype=np.int32)
            batch_segment_ids[i, ] = np.array(inputs[1], dtype=np.int32)
            batch_input_masks[i, ] = np.array(inputs[2], dtype=np.int32)
            batch_mistake_labels[i, ] = np.array(labels[0], dtype=np.float32)
            batch_char_labels[i, ] = np.array(labels[1], dtype=np.int32)
        x = {'Input-Token': batch_input_ids, 'Input-Segment': batch_segment_ids, 'Input-Masked': batch_input_masks,
             'mistake_labels': batch_mistake_labels, 'char_labels': batch_char_labels}
        # return x, [batch_char_labels, batch_mistake_labels], np.array(sample_weights, dtype=np.float32)
        return (x, )


class DataGenerator_old(object):
    def __init__(self, data, tokenizer, max_len=256, batch_size=32):
        self.data = []
        for sample in data:
            inputs, labels = convert_to_sample(sample, tokenizer, max_len)
            self.data.append((inputs, labels))
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            np.random.shuffle(idxs)
            input_ids, segment_ids, input_masks, mistake_labels, char_labels = [], [], [], [], []
            for i in idxs:
                inputs, labels = self.data[i]
                input_ids.append(inputs[0])
                segment_ids.append(inputs[1])
                input_masks.append(inputs[2])
                mistake_labels.append(labels[0])
                char_labels.append(labels[1])
                if len(input_ids) == self.batch_size or i == idxs[-1]:
                    yield [np.array(input_ids, dtype=np.int32), np.array(segment_ids, dtype=np.int32),
                           np.array(input_masks, dtype=np.int32), np.array(mistake_labels, dtype=np.int32),
                           np.array(char_labels, dtype=np.int32)], None
                    input_ids, segment_ids, input_masks, mistake_labels, char_labels = [], [], [], [], []
