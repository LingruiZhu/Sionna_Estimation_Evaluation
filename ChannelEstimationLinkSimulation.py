import numpy as np
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf

import sionna as sn
from sionna.ofdm import PilotPattern
from sionna.mimo import StreamManagement
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder, LDPCBPDecoder
from sionna.fec.interleaving import RandomInterleaver, Deinterleaver
from sionna.utils import BinarySource, ebnodb2no, hard_decisions
from sionna.ofdm import ResourceGrid, LMMSEEqualizer, ResourceGridMapper, LSChannelEstimator
from sionna.channel import ApplyOFDMChannel

from Simulation_Parameters import Simulation_Parameter
from parameter_configuration import get_default_resource_grid
from channel_data.read_channel_data import read_channel_data



class ChannelEstimationLinkSimulation:
    def __init__(self, sim_paras:Simulation_Parameter):
        # Set up parameters
        self.resource_grid = sim_paras.resource_grid
        self.num_data_symbols = self.resource_grid.num_data_symbols
        self.num_bits_per_symbol = sim_paras.num_bits_per_symbol
        self.batch_size = sim_paras.batch_size
        self.code_rate = sim_paras.code_rate
        self.num_code_bits = int(self.num_data_symbols * self.num_bits_per_symbol)
        self.num_info_bits = int(self.num_code_bits * self.code_rate)
        self.carrier_frequency = sim_paras.carrier_frequency
        self.delay_spread = sim_paras.delay_spread

        # parameters for MIMO, but here everything is single. 
        self.num_UE = 1                     # single user and base station
        self.num_BS = 1
        self.num_UE_ANT = 1                 # both with single antenna
        self.num_BS_ANT = 1
        self.rx_tx_association = np.array([[1]])
        self.stream_management = StreamManagement(self.rx_tx_association, num_streams_per_tx=self.num_UE_ANT)
            
        self.__initialize_transmitter()     # Set up transmitter
        self.__initialize_receiver()        # Set up receiver

    

    def __initialize_transmitter(self):
        self.source = BinarySource()
        self.ldpc_encoder = LDPC5GEncoder(k=self.num_info_bits, n=self.num_code_bits)
        self.constellation = Constellation("qam", num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.resource_grid_mapper = ResourceGridMapper(self.resource_grid)
        self.interleaver = RandomInterleaver()


    def __initialize_receiver(self):
        self.ls_estimator = LSChannelEstimator(self.resource_grid, interpolation_type="nn")
        self.lmmse_equalizer = LMMSEEqualizer(self.resource_grid, self.stream_management)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
        self.deinterleaver = Deinterleaver(self.interleaver)
        self.ldpc_decoder = LDPC5GDecoder(self.ldpc_encoder, hard_out=True)
    

    def set_channel_matrix(self, channel_matrix_list):
        """provide the real channel matrix for propagation.
           The matrcies will be stored in a list and given to the simulation

        Args:
            channel_matrix_list (_type_): _description_
        """
        self.channel_matrix_list = channel_matrix_list
    

    def set_channel_estimation(self, channel_estimation_list):
        self.channel_estimation_list = channel_estimation_list


    def update_mcs(self, modulation_order:int, code_rate:float):
        self.num_bits_per_symbol = modulation_order
        self.code_rate = code_rate
        self.num_code_bits = int(self.num_data_symbols * self.num_bits_per_symbol)
        self.num_info_bits = int(self.num_code_bits * self.code_rate)

        self.__initialize_transmitter()
        self.__initialize_receiver


    def snr_to_noise_variance(self, ebno_dB):
        no = sn.utils.ebnodb2no(ebno_dB,
                        num_bits_per_symbol=self.num_bits_per_symbol,
                        coderate=self.code_rate,
                        resource_grid=self.resource_grid)
        return no


    def transmit(self, batch_size:int=None):
        if batch_size == None:
            batch_size = self.batch_size
        info_bits = self.source([batch_size, self.num_UE, self.resource_grid.num_streams_per_tx, self.num_info_bits])
        codewords = self.ldpc_encoder(info_bits)
        codewords_interleaved = self.interleaver(codewords)
        symbols = self.mapper(codewords_interleaved)
        symbols_rg = self.resource_grid_mapper(symbols)
        return symbols_rg, info_bits
    

    def go_through_channel(self, tx_symbols, ebno_db):
        no = self.snr_to_noise_variance(ebno_db)
        rx_symbols, h_freq = self.channel([tx_symbols, no])
        return rx_symbols, h_freq
        

    def receive(self, rx_symbols, channel_estimation_matrix, ebno_db): 
        no = self.snr_to_noise_variance(ebno_db)
        channel_estimation, error_variance = self.ls_estimator([rx_symbols, no])
        channel_estimation_matrix = tf.cast(channel_estimation_matrix, "complex64")
        equalized_symbols, no_eff = self.lmmse_equalizer([rx_symbols, channel_estimation_matrix, error_variance, no])
        llr = self.demapper([equalized_symbols, no_eff])
        llr_deintlv = self.deinterleaver(llr)
        decoded_bits = self.ldpc_decoder(llr_deintlv)
        return decoded_bits


    def run(self, ebno_db):
        # set up simulation parameters
        tx_symbols, info_bits = self.transmit()
        rx_symbols, channel_freq = self.go_through_channel(tx_symbols, ebno_db)
        decoded_bits = self.receive(rx_symbols, ebno_db=10)
        ber = sn.utils.compute_ber(info_bits, decoded_bits)
        bler = sn.utils.compute_bler(info_bits, decoded_bits)
        return ber, bler

    
    def go_through_channel_single_PRB(self, channel_matrix, tx_symbols, ebno_db):
        no = self.snr_to_noise_variance(ebno_db) 
        applied_channel = ApplyOFDMChannel()
        channel_matrix = tf.cast(channel_matrix, "complex64")
        rx_symbols = applied_channel([tx_symbols, channel_matrix, no])
        return rx_symbols
    

    def simulate_single_channel_sample(self, channel_matrix, channel_estimation_matrix, num_trials, ebno_db):
        tx_symbols, info_bits = self.transmit(batch_size=num_trials)
        rx_symbols = self.go_through_channel_single_PRB(channel_matrix, tx_symbols, ebno_db)
        decoded_bits = self.receive(rx_symbols, channel_estimation_matrix, ebno_db)
        return info_bits, decoded_bits
    

    def simulate_estimation_performance(self, num_trials:int, snrs:list):
        bers_list = list()
        blers_list = list()
        for snr in snrs:
            tx_bits_set, rx_bits_set = None, None
            for channel_mat, channel_est_mat in zip(self.channel_matrix_list, self.channel_estimation_list):
                channel_mat_piled = tf.repeat(channel_mat, repeats=num_trials, axis=0)
                channel_est_piled = tf.repeat(channel_est_mat, repeats=num_trials, axis=0)
                tx_bits_single, rx_bits_single = self.simulate_single_channel_sample(channel_mat_piled, channel_est_piled, num_trials, ebno_db=snr)
                if tx_bits_set is None:
                    tx_bits_set, rx_bits_set = tx_bits_single, rx_bits_single
                else:
                    tx_bits_set = tf.concat([tx_bits_set, tx_bits_single], axis=0)
                    rx_bits_set = tf.concat([rx_bits_set, rx_bits_single], axis=0)
            ber = sn.utils.metrics.compute_ber(tx_bits_set, rx_bits_set)
            bler = sn.utils.metrics.compute_bler(tx_bits_set, rx_bits_set)
            bers_list.append(ber)
            blers_list.append(bler)
        return bers_list, blers_list


if __name__ == "__main__":
    resource_grid = get_default_resource_grid(num_symbols=14, num_subcarriers=72)
    sim_paras = Simulation_Parameter(resource_grid=resource_grid,
                                     batch_size=1,
                                     num_bits_per_symbol=2,
                                     code_rate=0.5,
                                     carrier_frequency=2.1e9,
                                     ue_speed = 10)
    
    data_file = "channel_data/CDL_channel.h5"
    channel_data = read_channel_data(data_file, start_point=0, num_PRB=10)
    channel_sample = channel_data[0:1,...,:72]          # kind of stupid slicing method
    
    channel_mat_list = list()
    channel_mat_list.append(channel_sample)
    channel_mat_list.append(channel_sample)

    channel_est_list = list()
    channel_est_list.append(channel_sample)
    channel_est_list.append(channel_sample)
    
    snrs = [-10, -5, 0]
    link_simulation = ChannelEstimationLinkSimulation(sim_paras)
    link_simulation.set_channel_matrix(channel_mat_list)
    link_simulation.set_channel_estimation(channel_est_list)
    ber_list, bler_list = link_simulation.simulate_estimation_performance(num_trials=2, snrs=snrs)    

    print(ber_list)
    print(bler_list)
