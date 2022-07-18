from dataclasses import dataclass
from sionna.ofdm import ResourceGrid
from enum import Enum



class Simulation_Parameter:
    def __init__(self, 
                resource_grid:ResourceGrid=None,
                batch_size:int=1,
                num_bits_per_symbol:int=None,
                code_rate:float=None,
                carrier_frequency:float=None,
                ue_speed:float=None,
                delay_spread:float=None):
        self.resource_grid = resource_grid
        self.batch_size = batch_size
        self.num_bits_per_symbol = num_bits_per_symbol
        self.code_rate = code_rate
        self.carrier_frequency = carrier_frequency
        self.ue_speed = ue_speed
        self.delay_spread = delay_spread


    def set_code_rate(self, code_rate):
        self.code_rate = code_rate


    def set_num_bits_per_symbol(self, num_bits_per_symbol):
        self.num_bits_per_symbol = num_bits_per_symbol
    

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    

    def set_channel_model(self, channel_model):
        self.channel_model = channel_model


def get_default_parameters(num_ofdm_symbols:int=14,
                           fft_size:int=76,
                           subcarrier_spacing:float=30e3,
                           num_tx:int=1,
                           num_rx:int=1,
                           num_stream_per_tx=1,
                           cyclic_prefix_length=6,
                           pilot_pattern="kronecker",
                           pilot_ofdm_symbol_indices=[2, 11]
                           ):
    default_resouce_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                                fft_size=fft_size,
                                subcarrier_spacing=subcarrier_spacing,
                                num_tx=num_tx,
                                num_streams_per_tx=num_stream_per_tx,
                                cyclic_prefix_length=cyclic_prefix_length,
                                pilot_pattern=pilot_pattern,
                                pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
    default_paras = Simulation_Parameter(resource_grid=default_resouce_grid,
                                         batch_size=1,
                                         num_bits_per_symbol=4,
                                         code_rate=0.5,
                                         carrier_frequency=2.6e9,
                                         ue_speed=10,
                                         delay_spread=100e-9)
    
    return default_paras
    


if __name__ == "__main__":
    """just test if can use functions set_code_rate and set_nums_bits_per_symbol correctly.
    """
    resouce_grid = ResourceGrid(num_ofdm_symbols=14,
                                fft_size=76,
                                subcarrier_spacing=15e3,
                                num_tx=1,
                                num_streams_per_tx=1,
                                cyclic_prefix_length=6,
                                pilot_pattern="kronecker",
                                pilot_ofdm_symbol_indices=[2, 11])
    batch_size = 100
    num_bits_per_symbol = 4
    code_rate = 0.5
    carrier_frequency = 2.6e9
    ue_speed = 10
    delay_spread = 100e-9#

    sim_paras = Simulation_Parameter(resouce_grid, batch_size, num_bits_per_symbol, code_rate, carrier_frequency, ue_speed, delay_spread)

    print(sim_paras.num_bits_per_symbol)
    print(sim_paras.code_rate)

    new_modulation_order = 8
    new_code_rate = 0.75

    sim_paras.set_code_rate(new_code_rate)
    sim_paras.set_num_bits_per_symbol(new_modulation_order)

    print(sim_paras.num_bits_per_symbol)
    print(sim_paras.code_rate)

