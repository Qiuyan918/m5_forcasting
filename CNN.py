import os

import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()

from data_frame import DataFrame
from tf_base_model import TFBaseModel
from tf_utils import (
    time_distributed_dense_layer, temporal_convolution_layer,
    sequence_mean, sequence_rmse, shape
)

import random as rn
def seed_everything(seed):
    
    rn.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)
    
os.listdir('data/processed/')

def mse(y, y_hat):
    square_error =  tf.reduce_mean(tf.square(y_hat - y))
    return square_error

def mae(y, y_hat):
    mae =  tf.reduce_mean(tf.abs(y_hat - y))
    return mae

def mse_all(y, y_hat):
    y_hat = tf.reduce_sum(y_hat, axis=0)
    y = tf.reduce_sum(y, axis=0)
    square_error =  tf.reduce_mean(tf.square(y_hat - y))
    return square_error

def transfer_to_hierarchy(array, hierarchies, batch_size):

    index = [tf.constant(0), array]
    array_copy = tf.identity(array)

    def condition(index, array):
        return tf.less(index, tf.shape(hierarchies)[1])

    def body(index, array):
        hierarchy = tf.gather(hierarchies, index, axis=1)
        __, hierarchy = tf.unique(hierarchy)
        array_hierarchy = tf.unsorted_segment_sum(array_copy, 
                                                  hierarchy,
                                                  num_segments=tf.size(tf.unique(hierarchy)[0]))
        
#         array_hierarchy = array_hierarchy * tf.cast((30490.0 
#                                                      / tf.cast(batch_size, tf.float32) 
#                                                      / tf.cast((tf.reduce_max(hierarchy) + 1), tf.float32))
#                                                     , tf.float32)
        
        array = tf.concat([array, array_hierarchy], 0)
        return tf.add(index, 1), array

    array = tf.while_loop(condition, body, index)[1]
    return array

def transfer_to_hierarchy_weights(array, hierarchies):

    index = [tf.constant(0), array]
    array_copy = tf.identity(array)

    def condition(index, array):
        return tf.less(index, tf.shape(hierarchies)[1])

    def body(index, array):
        hierarchy = tf.gather(hierarchies, index, axis=1)
        __, hierarchy = tf.unique(hierarchy)
        array_hierarchy = tf.unsorted_segment_sum(array_copy, 
                                                  hierarchy,
                                                  num_segments=tf.size(tf.unique(hierarchy)[0]))
                
        array = tf.concat([array, array_hierarchy], 0)
        return tf.add(index, 1), array

    array = tf.while_loop(condition, body, index)[1]
    return array

def convert_to_scale(history_seq, seq_len):

    scales = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    index = [tf.constant(0), scales]
    history_seq_len = tf.shape(history_seq)[0]

    def condition(index, scales):
        return tf.less(index, history_seq_len)

    def body(index, scales):
        series = tf.gather(history_seq, index, axis=0)
        series_len = tf.gather(seq_len, index, axis=0) # tf.cast(seq_len[index], tf.int32)
        series = series[:series_len]
        
        pred = series[1:] 
        true = series[:-1] 
        scale = tf.sqrt(tf.reduce_mean(tf.square(pred - true)))
        scales = scales.write(index, scale)

        return tf.add(index, 1), scales
    
    __, scales = tf.while_loop(condition, body, index)
    scales = scales.stack()
    return scales

def rmsse(y_true, y_pred, hierarchies, x_encode, x_encode_len, weights, batch_size):
    y_true = transfer_to_hierarchy(y_true, hierarchies, batch_size)
    y_pred = transfer_to_hierarchy(y_pred, hierarchies, batch_size)
    weights = transfer_to_hierarchy_weights(weights, hierarchies)
#     x_encode = transfer_to_hierarchy(x_encode, hierarchies, batch_size)
    
#     scales = convert_to_scale(x_encode, x_encode_len)
#     scales = tf.where(tf.equal(scales, np.nan), tf.ones_like(scales), scales)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=1))   
#     rmse = tf.where(tf.equal(rmse, np.nan), tf.ones_like(rmse), rmse)
    
#     rmsse = tf.divide(rmse, scales)
    wrmse = tf.reduce_sum(weights * rmse) / tf.reduce_sum(weights)
    return  wrmse

class DataReader(object):

    def __init__(self, data_dir):
        data_cols = [
            'x',           
            'store_id',
            'item_id',           
            'state_id',
            'dept_id',
            'cat_id',   
            
            'wday',
            'month',
            'event_name_1', 
            'event_type_1',
            'snap',           
            'x_lags', 
            'xy_lags', 
            'ts',    
            'sell_price', 
            'sell_price_first_digit', # to try
            'sell_price_last_digit', 
            'start_date', 
            'weights',
            
            'hierarchy_data',            
            'all_id']
        
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]

        self.test_df = DataFrame(columns=data_cols, data=data)
#         self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95)

        self.num_store = self.test_df['store_id'].max() + 1
        self.num_item = self.test_df['item_id'].max() + 1
        self.num_state = self.test_df['state_id'].max() + 1
        self.num_dept = self.test_df['dept_id'].max() + 1
        self.num_cat = self.test_df['cat_id'].max() + 1       
        self.num_wday = self.test_df['wday'].max() + 1
        self.num_month = self.test_df['month'].max() + 1
        self.num_event_name_1 = self.test_df['event_name_1'].max() + 1
        self.num_event_type_1 = self.test_df['event_type_1'].max() + 1

#         print 'train size', len(self.train_df)
#         print 'val size', len(self.val_df)
        print 'test size', len(self.test_df)

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=True,
            num_epochs=10000,
            mode='train'
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=True,
            num_epochs=10000,
            mode='val'
        )

    def test_batch_generator(self, batch_size, mode):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=True,
            num_epochs=1,
            mode=mode
        )

    def batch_generator(self, batch_size, df, mode, shuffle=True, num_epochs=10000):
        batch_gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=(mode in ('val_test', 'test'))
        )
        for batch in batch_gen:
            num_decode_steps = 28

            full_seq_len = batch['x'].shape[1] - num_decode_steps 
            max_encode_length = full_seq_len
            
            x_for_mean = np.zeros([len(batch), max_encode_length])
            x_len_for_mean = np.zeros([len(batch)])
            x = np.zeros([len(batch), max_encode_length])
            y = np.zeros([len(batch), num_decode_steps])
            x_lags = np.zeros([len(batch), max_encode_length, batch['x_lags'].shape[2] + batch['xy_lags'].shape[2]])
            y_lags = np.zeros([len(batch), num_decode_steps, batch['xy_lags'].shape[2]])
            x_len = np.zeros([len(batch)])
            y_len = np.zeros([len(batch)])
            x_day = np.zeros([len(batch), max_encode_length])
            y_day = np.zeros([len(batch), num_decode_steps])
            y_id = np.zeros([len(batch), num_decode_steps])
            x_ts = np.zeros([len(batch), max_encode_length, batch['ts'].shape[2]])
            x_snap = np.zeros([len(batch), max_encode_length])
            y_snap = np.zeros([len(batch), num_decode_steps])
            x_event_name_1 = np.zeros([len(batch), max_encode_length])
            y_event_name_1 = np.zeros([len(batch), num_decode_steps])            
            x_event_type_1 = np.zeros([len(batch), max_encode_length])
            y_event_type_1 = np.zeros([len(batch), num_decode_steps])   
            x_sell_price = np.zeros([len(batch), max_encode_length])
            y_sell_price = np.zeros([len(batch), num_decode_steps])  
            x_sell_price_last_digit = np.zeros([len(batch), max_encode_length])
            y_sell_price_last_digit = np.zeros([len(batch), num_decode_steps]) 
            x_wday = np.zeros([len(batch), max_encode_length])
            y_wday = np.zeros([len(batch), num_decode_steps]) 
            
            for i, (data, start_idx, x_lag, xy_lag, ts, snap, event_name_1, event_type_1,
                    sell_price, sell_price_last_digit, wday) in enumerate(zip(
                    batch['x'], batch['start_date'], batch['x_lags'],
                    batch['xy_lags'], batch['ts'], batch['snap'],
                    batch['event_name_1'], batch['event_type_1'], batch['sell_price'],
                    batch['sell_price_last_digit'], batch['wday']
                )
            ):
                val_window = 365
                train_window = 365
                seq_len_for_mean = full_seq_len - start_idx - num_decode_steps
                
                if mode == 'train':
                    seq_len = full_seq_len - start_idx - num_decode_steps # 1913 - 28 = 1885
                    if seq_len == 0:
                        rand_encode_len = 0
                        weights[i] = 0
                    elif seq_len <= train_window:
                        rand_encode_len = seq_len
                    else:
                        rand_encode_len = np.random.randint(seq_len - train_window, seq_len)
                    rand_decode_len = min(seq_len - rand_encode_len, num_decode_steps)

                elif mode == 'val':
                    start_idx = start_idx + num_decode_steps
                    seq_len = full_seq_len - start_idx - num_decode_steps # 1885
                    if seq_len <= num_decode_steps:
                        rand_encode_len = 0
                        weights[i] = 0
                    elif seq_len <= val_window :
                        rand_encode_len = seq_len
                    else:
                        rand_encode_len = np.random.randint(seq_len - val_window, seq_len + 1)
                    rand_decode_len = num_decode_steps

                elif mode == 'val_test':
                    seq_len = full_seq_len - start_idx
                    rand_encode_len = seq_len - num_decode_steps
                    rand_decode_len = num_decode_steps    
                                        
                elif mode == 'test':
                    seq_len = full_seq_len - start_idx
                    rand_encode_len = seq_len
                    rand_decode_len = num_decode_steps
                    
                elif mode == 'submission':
                    rand_encode_len = batch['x'].shape[1] - start_idx
                    rand_decode_len = num_decode_steps                    
                    
                end_idx = start_idx + rand_encode_len
                
                x_for_mean[i, :seq_len_for_mean] = data[start_idx:(start_idx+seq_len_for_mean)]         
                x_len_for_mean[i] = seq_len_for_mean
                            
                x[i, :rand_encode_len] = data[start_idx: end_idx]
                x_lags[i, :rand_encode_len, :x_lag.shape[1]] = x_lag[start_idx: end_idx, :]
                x_lags[i, :rand_encode_len, x_lag.shape[1]:] = xy_lag[start_idx: end_idx, :]
                x_ts[i, :rand_encode_len, :] = ts[start_idx: end_idx, :]
                x_len[i] = end_idx - start_idx
                x_day[i, :rand_encode_len] = (np.arange(batch['x'].shape[1])/1000)[start_idx: end_idx]
                x_snap[i, :rand_encode_len] = snap[start_idx: end_idx]
                x_event_name_1[i, :rand_encode_len] = event_name_1[start_idx: end_idx]
                x_event_type_1[i, :rand_encode_len] = event_type_1[start_idx: end_idx]
                x_sell_price[i, :rand_encode_len] = sell_price[start_idx: end_idx]
                x_sell_price_last_digit[i, :rand_encode_len] = sell_price_last_digit[start_idx: end_idx]
                x_wday[i, :rand_encode_len] = wday[start_idx: end_idx]
                
                y[i, :rand_decode_len] = data[end_idx: end_idx + rand_decode_len]
                y_lags[i, :rand_decode_len, :] = xy_lag[end_idx: end_idx + rand_decode_len, :]
                y_len[i] = rand_decode_len   
                y_day[i, :rand_decode_len] = (np.arange(batch['x'].shape[1])/1000)[end_idx: end_idx + rand_decode_len]
                y_snap[i, :rand_decode_len] = snap[end_idx: end_idx + rand_decode_len]
                y_event_name_1[i, :rand_decode_len] = event_name_1[end_idx: end_idx + rand_decode_len]
                y_event_type_1[i, :rand_decode_len] = event_type_1[end_idx: end_idx + rand_decode_len]
                y_sell_price[i, :rand_decode_len] = sell_price[end_idx: end_idx + rand_decode_len]
                y_sell_price_last_digit[i, :rand_decode_len] = sell_price_last_digit[end_idx: end_idx + rand_decode_len]
                y_wday[i, :rand_decode_len] = wday[end_idx: end_idx + rand_decode_len]                    
                    
#             batch['x_scales'] = x_scales     
            batch['x_for_mean'] = x_for_mean
            batch['x_len_for_mean'] = x_len_for_mean
            batch['x'] = x
            batch['y'] = y
            batch['x_lags'] = x_lags
            batch['y_lags'] = y_lags
            batch['x_ts'] = x_ts
            batch['x_len'] = x_len
            batch['y_len'] = y_len
            batch['x_day'] = x_day
            batch['y_day'] = y_day
            batch['x_snap'] = x_snap
            batch['y_snap'] = y_snap
            batch['x_event_name_1'] = x_event_name_1
            batch['y_event_name_1'] = y_event_name_1
            batch['x_event_type_1'] = x_event_type_1
            batch['y_event_type_1'] = y_event_type_1   
            batch['x_sell_price'] = x_sell_price
            batch['y_sell_price'] = y_sell_price  
            batch['x_sell_price_last_digit'] = x_sell_price_last_digit
            batch['y_sell_price_last_digit'] = y_sell_price_last_digit 
            batch['x_wday'] = x_wday
            batch['y_wday'] = y_wday            
            yield batch
            


class cnn(TFBaseModel):

    def __init__(
        self,
        residual_channels=32,
        skip_channels=32,
        dilations=[2**i for i in range(8)]*3,
        filter_widths=[2 for i in range(8)]*3,
        num_decode_steps=28,
        **kwargs
    ):
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilations = dilations
        self.filter_widths = filter_widths
        self.num_decode_steps = num_decode_steps
        super(cnn, self).__init__(**kwargs)

    def get_input_sequences(self):
        # self.x_scales = tf.placeholder(tf.float32, [None, None])
        self.x = tf.placeholder(tf.float32, [None, None])
        self.y = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.x_len = tf.placeholder(tf.int32, [None])
        self.y_len = tf.placeholder(tf.int32, [None])
        self.all_id = tf.placeholder(tf.int32, [None])
        
        self.x_for_mean = tf.placeholder(tf.float32, [None, None])
        self.x_len_for_mean = tf.placeholder(tf.int32, [None])

        self.x_ts = tf.placeholder(tf.float32, [None, None, 10])
        self.x_lags = tf.placeholder(tf.float32, [None, None, 13])
        self.y_lags = tf.placeholder(tf.float32, [None, self.num_decode_steps, 8])
        self.x_day = tf.placeholder(tf.float32, [None, None])
        self.y_day = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.x_snap = tf.placeholder(tf.float32, [None, None])
        self.y_snap = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.x_event_name_1 = tf.placeholder(tf.float32, [None, None])
        self.y_event_name_1 = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.x_event_type_1 = tf.placeholder(tf.float32, [None, None])
        self.y_event_type_1 = tf.placeholder(tf.float32, [None, self.num_decode_steps])        
        self.x_sell_price = tf.placeholder(tf.float32, [None, None])
        self.y_sell_price = tf.placeholder(tf.float32, [None, self.num_decode_steps])                
        self.x_sell_price_last_digit = tf.placeholder(tf.int32, [None, None])
        self.y_sell_price_last_digit = tf.placeholder(tf.int32, [None, self.num_decode_steps])             
        self.x_wday = tf.placeholder(tf.int32, [None, None])
        self.y_wday = tf.placeholder(tf.int32, [None, self.num_decode_steps])   
        
        self.store_id = tf.placeholder(tf.int32, [None])
        self.state_id = tf.placeholder(tf.int32, [None])
        self.dept_id = tf.placeholder(tf.int32, [None])
        self.cat_id = tf.placeholder(tf.int32, [None])
        self.item_id = tf.placeholder(tf.int32, [None])
        self.hierarchy_data = tf.placeholder(tf.float32, [None, 11])
        
        self.weights = tf.placeholder(tf.float32, [None])

        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        item_id_embeddings = tf.get_variable(
            name='item_id_embeddings',
            shape=[self.reader.num_item, 20], 
            dtype=tf.float32
        )
        item_id = tf.nn.embedding_lookup(item_id_embeddings, self.item_id)

        self.x_mean = tf.expand_dims(sequence_mean(self.x_for_mean, self.x_len_for_mean), 1)
        self.x_centered = self.x - self.x_mean
        self.y_centered = self.y - self.x_mean
        self.x_ts_centered = self.x_ts - tf.expand_dims(self.x_mean, 2)
        self.x_lags_centered = self.x_lags - tf.expand_dims(self.x_mean, 2)
        self.y_lags_centered = self.y_lags - tf.expand_dims(self.x_mean, 2)
        self.x_is_zero = tf.cast(tf.equal(self.x, tf.zeros_like(self.x)), tf.float32)

        self.encode_features = tf.concat([
            self.x_ts_centered,
            self.x_lags_centered,
#             tf.expand_dims(self.x_day, 2),
            tf.one_hot(self.x_sell_price_last_digit, 9),
#             tf.one_hot(self.x_wday, 6),            
            tf.expand_dims(self.x_snap, 2),
            tf.expand_dims(self.x_is_zero, 2),
            tf.expand_dims(self.x_event_name_1, 2),
            tf.expand_dims(self.x_event_type_1, 2),     
            tf.expand_dims(self.x_sell_price, 2),                        
            tf.tile(tf.expand_dims(self.x_mean, 2), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.store_id, self.reader.num_store), 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.state_id, self.reader.num_state), 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.dept_id, self.reader.num_dept), 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.cat_id, self.reader.num_cat), 1), (1, tf.shape(self.x)[1], 1)),
            tf.tile(tf.expand_dims(item_id, 1), (1, tf.shape(self.x)[1], 1)),
        ], axis=2)

        decode_idx = tf.tile(tf.expand_dims(tf.range(self.num_decode_steps), 0), (tf.shape(self.y)[0], 1))
        self.decode_features = tf.concat([
            self.y_lags_centered,
#             tf.expand_dims(self.y_day, 2),
            tf.one_hot(self.y_sell_price_last_digit, 9),
#             tf.one_hot(self.y_wday, 6),             
            tf.expand_dims(self.y_snap, 2),
            tf.expand_dims(self.y_event_name_1, 2),
            tf.expand_dims(self.y_event_type_1, 2),    
            tf.expand_dims(self.y_sell_price, 2),                  
            tf.one_hot(decode_idx, self.num_decode_steps),
            tf.tile(tf.expand_dims(self.x_mean, 2), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.store_id, self.reader.num_store), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.state_id, self.reader.num_state), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.dept_id, self.reader.num_dept), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.cat_id, self.reader.num_cat), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(item_id, 1), (1, self.num_decode_steps, 1)),
        ], axis=2)

        return tf.expand_dims(self.x_centered, 2)

    def encode(self, x, features):
        x = tf.concat([x, features], axis=2)

        h = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='x-init',
        )
        c = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='c-init',
        )

        conv_inputs = [h]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)[:-1]):
            dilated_conv = temporal_convolution_layer(
                inputs=h,
                output_units=4*self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-encode-{}'.format(i),
            )
            input_gate, conv_filter, conv_gate, emit_gate = tf.split(dilated_conv, 4, axis=2)

            c = tf.nn.sigmoid(input_gate)*c + tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)
            h = tf.nn.sigmoid(emit_gate)*tf.nn.tanh(c)
            conv_inputs.append(h)

        return conv_inputs

    def initialize_decode_params(self, x, features):
        x = tf.concat([x, features], axis=2)

        h = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='h-init-decode',
        )
        c = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='c-init-decode',
        )

        skip_outputs = []
        conv_inputs = [h]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
            dilated_conv = temporal_convolution_layer(
                inputs=h,
                output_units=4*self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-decode-{}'.format(i),
            )
            input_gate, conv_filter, conv_gate, emit_gate = tf.split(dilated_conv, 4, axis=2)

            c = tf.nn.sigmoid(input_gate)*c + tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)
            h = tf.nn.sigmoid(emit_gate)*tf.nn.tanh(c)

            skip_outputs.append(h)
            conv_inputs.append(h)

        skip_outputs = tf.concat(skip_outputs, axis=2)
        h = time_distributed_dense_layer(skip_outputs, 128, scope='dense-decode-1', activation=tf.nn.relu)
        y_hat = time_distributed_dense_layer(h, 2, scope='dense-decode-2')
        return y_hat
    
    def decode(self, x, conv_inputs, features):
        batch_size = tf.shape(x)[0]

        # initialize state tensor arrays
        state_queues = []
        for i, (conv_input, dilation) in enumerate(zip(conv_inputs, self.dilations)):
            batch_idx = tf.range(batch_size)
            batch_idx = tf.tile(tf.expand_dims(batch_idx, 1), (1, dilation))
            batch_idx = tf.reshape(batch_idx, [-1])

            temporal_idx = tf.expand_dims(self.x_len, 1) + tf.expand_dims(tf.range(dilation), 0)
            temporal_idx = tf.reshape(temporal_idx, [-1])

            idx = tf.stack([batch_idx, temporal_idx], axis=1)
            padding = tf.zeros([batch_size, dilation + 1, shape(conv_input, 2)])
            conv_input = tf.concat([padding, conv_input], axis=1)
            slices = tf.reshape(tf.gather_nd(conv_input, idx), (batch_size, dilation, shape(conv_input, 2)))

            layer_ta = tf.TensorArray(dtype=tf.float32, size=dilation + self.num_decode_steps)
            layer_ta = layer_ta.unstack(tf.transpose(slices, (1, 0, 2)))
            state_queues.append(layer_ta)

        # initialize feature tensor array
        features_ta = tf.TensorArray(dtype=tf.float32, size=self.num_decode_steps)
        features_ta = features_ta.unstack(tf.transpose(features, (1, 0, 2)))

        # initialize output tensor array
        emit_ta = tf.TensorArray(size=self.num_decode_steps, dtype=tf.float32)

        # initialize other loop vars
        elements_finished = 0 >= self.y_len
        time = tf.constant(0, dtype=tf.int32)

        # get initial x input
        current_idx = tf.stack([tf.range(tf.shape(self.x_len)[0]), self.x_len - 1], axis=1)
        initial_input = tf.gather_nd(x, current_idx)

        def loop_fn(time, current_input, queues):
            current_features = features_ta.read(time)
            current_input = tf.concat([current_input, current_features], axis=1)

            with tf.variable_scope('h-init-decode', reuse=True):
                w_x_proj = tf.get_variable('weights')
                b_x_proj = tf.get_variable('biases')
                h = tf.nn.tanh(tf.matmul(current_input, w_x_proj) + b_x_proj)

            with tf.variable_scope('c-init-decode', reuse=True):
                w_x_proj = tf.get_variable('weights')
                b_x_proj = tf.get_variable('biases')
                c = tf.nn.tanh(tf.matmul(current_input, w_x_proj) + b_x_proj)

            skip_outputs, updated_queues = [], []
            for i, (queue, dilation) in enumerate(zip(queues, self.dilations)):

                state = queue.read(time)
                with tf.variable_scope('dilated-conv-decode-{}'.format(i), reuse=True):
                    w_conv = tf.get_variable('weights')
                    b_conv = tf.get_variable('biases')
                    dilated_conv = tf.matmul(state, w_conv[0, :, :]) + tf.matmul(h, w_conv[1, :, :]) + b_conv

                input_gate, conv_filter, conv_gate, emit_gate = tf.split(dilated_conv, 4, axis=1)

                c = tf.nn.sigmoid(input_gate)*c + tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)
                h = tf.nn.sigmoid(emit_gate)*tf.nn.tanh(c)

                skip_outputs.append(h)
                updated_queues.append(queue.write(time + dilation, h))

            skip_outputs = tf.concat(skip_outputs, axis=1)
            with tf.variable_scope('dense-decode-1', reuse=True):
                w_h = tf.get_variable('weights')
                b_h = tf.get_variable('biases')
                h = tf.nn.relu(tf.matmul(skip_outputs, w_h) + b_h)

            with tf.variable_scope('dense-decode-2', reuse=True):
                w_y = tf.get_variable('weights')
                b_y = tf.get_variable('biases')
                y_hat = tf.matmul(h, w_y) + b_y

            elements_finished = (time >= self.y_len)
            finished = tf.reduce_all(elements_finished)

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, 2], dtype=tf.float32),
                lambda: y_hat
            )
            next_elements_finished = (time >= self.num_decode_steps - 1)

            return (next_elements_finished, next_input, updated_queues)

        def condition(unused_time, elements_finished, *_):
            return tf.logical_not(tf.reduce_all(elements_finished))

        def body(time, elements_finished, emit_ta, *state_queues):
            (next_finished, emit_output, state_queues) = loop_fn(time, initial_input, state_queues)

            emit = tf.where(elements_finished, tf.zeros_like(emit_output), emit_output)
            emit_ta = emit_ta.write(time, emit)

            elements_finished = tf.logical_or(elements_finished, next_finished)
            return [time + 1, elements_finished, emit_ta] + list(state_queues)

        returned = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[time, elements_finished, emit_ta] + state_queues
        )

        outputs_ta = returned[2]
        y_hat = tf.transpose(outputs_ta.stack(), (1, 0, 2))
        return y_hat

    def calculate_loss(self):
        x = self.get_input_sequences()

        conv_inputs = self.encode(x, features=self.encode_features)
        decode_x = tf.concat([x, 1.0 - tf.expand_dims(self.x_is_zero, 2)], axis=2)
        self.initialize_decode_params(decode_x, features=self.decode_features)

        y_hat = self.decode(decode_x, conv_inputs, features=self.decode_features)
        y_hat, p = tf.unstack(y_hat, axis=2, num=2)
        y_hat = tf.nn.sigmoid(p)*(y_hat + self.x_mean)
        self.loss = rmsse(tf.exp(self.y)-1, tf.exp(y_hat)-1, self.hierarchy_data, self.x_for_mean, self.x_len_for_mean, self.weights, tf.shape(x)[0])
        
#         self.loss = mse(tf.exp(self.y)-1, tf.exp(y_hat)-1)

        self.prediction_tensors = {
            'preds': tf.nn.relu(y_hat),
            'ids': self.all_id,
        }

        return self.loss

dr = DataReader(data_dir='data/processed/')
base_dir = './'
nn = cnn(
    reader=dr,
    log_dir=os.path.join(base_dir, 'logs'),
    checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
    prediction_dir=os.path.join(base_dir, 'predictions'),
    submission_dir=os.path.join(base_dir, 'submissions'),
    optimizer='adam',
    learning_rates=[.0005, .00025, .000125],
    beta1_decays=[.9, .9, .9],
    batch_sizes=[64, 128, 256],
    num_training_steps=200000, # 200000
    patiences=[5000, 5000, 5000],
    warm_start_init_step=0,
    regularization_constant=0.0,
    keep_prob=1.0,
    enable_parameter_averaging=True,
    min_steps_to_checkpoint=500,
    log_interval=50,
    validation_batch_size=4*64,
    grad_clip=20,
    residual_channels=32,
    skip_channels=32,
    dilations=[2**i for i in range(9)]*3,
    filter_widths=[2 for i in range(9)]*3,
    num_decode_steps=28,
    loss_averaging_window=200
)

seed_everything(123)
nn.fit()

nn.restore()
nn.predict(mode='val_test')
nn.predict(mode='test')