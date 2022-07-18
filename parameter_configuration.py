from operator import sub
import numpy as np
import sionna as sn

from sionna.ofdm import ResourceGrid
from sionna.ofdm.pilot_pattern import PilotPattern
from sionna.utils import QAMSource


def get_pilot_mask(num_pilots:int):
    num_tx = 1
    num_stream_per_tx = 1
    num_subcarriers = 72
    num_symbols = 14

    if num_pilots == 48:
        idx = [14*i for i in range(1, 72,6)]+[4+14*(i) for i in range(4, 72,6)]+[7+14*(i) for i in range(1, 72,6)]+[11+14*(i) for i in range(4, 72,6)]
    r = [x//14 for x in idx]
    c = [x%14 for x in idx]
    shape = [num_tx, num_stream_per_tx, num_symbols, num_subcarriers]
    
    mask = np.zeros(shape, bool)
    for row_index, col_index in zip(r, c):
        mask[..., col_index, row_index] = True
    pilots = np.zeros(shape, np.complex64)

    num_tx = mask.shape[0]
    num_stream_per_tx = mask.shape[1]
    pilots_shape = [num_tx, num_stream_per_tx, num_pilots]
    qam_source = QAMSource(num_bits_per_symbol=2)
    pilots = qam_source(pilots_shape)
    return mask, pilots



def get_default_pilot_parttern(num_pilots:int=48):
    default_mask, default_pilots = get_pilot_mask(num_pilots)
    default_pilot_pattern = PilotPattern(default_mask, default_pilots)
    return default_pilot_pattern



def get_default_resource_grid(num_symbols:int=72, num_subcarriers:int=14):
    default_pilot_pattern = get_default_pilot_parttern(num_pilots=48)
    default_resource_grid = ResourceGrid(num_ofdm_symbols=num_symbols,
                                        fft_size = num_subcarriers,
                                        subcarrier_spacing=15e3,
                                        pilot_pattern=default_pilot_pattern
                                        )
    return default_resource_grid
#TODO: to check if pilot pattern (masks and pilots) are set correctly!!!

def test():
    resource_grid_default = get_default_resource_grid()
    print(resource_grid_default.fft_size)


if __name__ == "__main__":
    test()

