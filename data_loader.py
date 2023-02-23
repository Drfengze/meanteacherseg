import tensorflow as tf

class data_load:
    def __init__(
        self,
        input_files,
        bands,
        features_dict,
        nclass=2,
        response=None,
        buffer_size=6000,
        batch_size=64,
        split=.7
    ):
        self.input_files = input_files
        self.bands = bands
        self.response = response
        self.buffer_size = buffer_size
        self.batch_size = batch_size
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

    def get_training_dataset(self):
        glob = tf.io.gfile.glob(self.input_files)
        dataset = tf.data.TFRecordDataset(glob, compression_type="GZIP")
        dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=10)
        dataset = dataset.map(self.to_tuple).shuffle(self.buffer_size)
        return dataset

    # def get_training_dataset(self):
    #     files = self.input_files
    #     dataset,dataset_size = self.get_dataset(files)
    #     train_dataset = dataset.take(self.split*dataset_size)
    #     valid_dataset = dataset.skip()
    #     train_dataset = train_dataset.batch(self.batch_size)
    #     # iterator = dataset.make_one_shot_iterator()
    #     # data =  iterator.get_next()
    #     return train_dataset
    
    # def get_eval_dataset(self):
    #     files = self.input_files
    #     dataset = self.get_dataset(files)
    #     dataset = dataset.batch(self.batch_size)  # .repeat()
    #     # iterator = dataset.make_one_shot_iterator()
    #     # data =  iterator.get_next()
    #     return dataset

    def get_pridiction_dataset(self):
        files = self.input_files
        dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")
        dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=10)
        dataset = dataset.map(self.to_tuple_prediction).batch(self.batch_size)
        return dataset
