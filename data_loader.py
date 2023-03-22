import tensorflow as tf

class data_load:
    def __init__(
        self,
        input_files,
        bands,
        features_dict,
        nclass=2,
        response=None,
        shuffle_size=6000,
        split=.7
    ):
        self.input_files = input_files
        self.bands = bands
        self.response = response
        self.shuffle_size = shuffle_size
        self.features_dict = features_dict
        self.nclass = nclass
        self.split = split

    def parse_tfrecord(self, example_proto):
        return tf.io.parse_single_example(example_proto, self.features_dict)

    def to_tuple(self, inputs):
        image_channels = [inputs.get(key) for key in self.bands]
        image = tf.stack(image_channels, axis=0)
        response = [inputs.get(key) for key in self.response]
        label = tf.stack(response, axis=0)
        # label = tf.squeeze(tf.one_hot(indices=int(stacked[-1,:, :]), depth=self.nclass))
        return image, label

    def to_tuple_prediction(self, inputs):
        inputsList = [inputs.get(key) for key in self.bands]
        stacked = tf.stack(inputsList, axis=0)
        return stacked
    
    def get_dataset(self):
        glob = tf.io.gfile.glob(self.input_files)
        dataset = tf.data.TFRecordDataset(glob, compression_type="GZIP")
        dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=10)
        return dataset
    
    def get_training_dataset(self):
        dataset = self.get_dataset()
        dataset = dataset.map(self.to_tuple).shuffle(self.shuffle_size)
        return dataset

    def get_pridiction_dataset(self):
        dataset = self.get_dataset()
        dataset = dataset.map(self.to_tuple_prediction)
        return dataset
