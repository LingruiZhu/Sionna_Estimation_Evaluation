from tkinter import X
import numpy as np
import tensorflow as tf
import sionna

from keras.layers import Layer
from sionna.ofdm import RemoveNulledSubcarriers
from sionna.utils import flatten_last_dims, flatten_dims, split_dim


class LeastSquareEqualizer(Layer):
    def __init__(self,
                 resource_grid,
                 stream_management,
                 whiten_interference=True,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._whiten_interference = whiten_interference
        self._removed_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")
        self._data_ind = data_ind[...,:num_data_symbols]


    def call(self, inputs):
        y, h_hat, no = inputs
        # y has the shape [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size]
        # h_hat has the shape [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_ofdm_symbols, fft_size]

        y_eff = self._removed_nulled_scs(y)             # remove nulled subcarriers from y (guards, dc)
        y_dt = tf.transpose(y_eff, [0, 1, 3, 4 ,2])     # New shape [batch_size, num_rx, num_ofdm_symbols, num_eff_scs, num_rx_ant]
        y_dt = tf.cast(y_dt, self.dtype)                

        perm = [1, 3, 4, 0, 2, 5, 6]                    # channel new shape [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant, ...                                    
        h_dt = tf.transpose(h_hat, perm)                # ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt = flatten_dims(h_dt, 3, 0)                 # flatten the first three dimensions

        ind_desired = self._stream_management.detection_desired_ind     # gather desired and undesired chanenls
        ind_undesired = self._stream_management.detection_undesired_ind
        h_dt_desired = tf.gather(h_dt, ind_desired, axis=0)
        h_dt_undesired = tf.gather(h_dt, ind_undesired, axis=0)
        h_dt_desired = split_dim(h_dt_desired, [self._stream_management.num_rx, -1], 0)
        h_dt_undesired = split_dim(h_dt_undesired, [self._stream_management.num_rx, -1], 0)
        perm = [2, 0, 4, 5, 3, 1]
        h_dt_desired = tf.transpose(h_dt_desired, perm)
        h_dt_desired = tf.cast(h_dt_desired, self._dtype)
        h_dt_undesired = tf.transpose(h_dt_undesired, perm)
        h_dt_desired = tf.squeeze(h_dt_desired, axis=-1)
        # In the end, the dimension [batch_size, num_rx, num_ofdm_symbols, num_eff_scs, num_rx_ant, num_streams_per_rx]

        # Least square estimation
        x_hat = tf.math.divide(y_dt, h_dt_desired)
        x_hat = tf.transpose(x_hat, [1, 4, 2, 3, 0])
        x_hat = flatten_dims(x_hat, 2, 0)
        stream_ind = self._stream_management.stream_ind
        x_hat = tf.gather(x_hat, stream_ind, axis=0)

        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        x_hat = split_dim(x_hat, [num_tx, num_streams], 0)
        x_hat = flatten_dims(x_hat, 2, 2)
        x_hat = tf.gather(x_hat, self._data_ind, batch_dims=2, axis=2)
        x_hat = tf.transpose(x_hat, [3, 0, 1, 2])
        
        no_eff = no * tf.ones_like(x_hat, dtype=np.float32)

        return x_hat, no_eff







