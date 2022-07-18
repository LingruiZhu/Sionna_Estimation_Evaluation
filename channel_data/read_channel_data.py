import numpy as np
import h5py

def read_channel_data(file_path:str, start_point:int=0, num_PRB:int=100):
    data_file = h5py.File(file_path, "r")
    channel_data = data_file['channel_data'][:]     # use this line to convert Dataset to np.ndarray
    output_channel = channel_data[start_point:start_point+num_PRB,:,:,:,:,:,:]
    return output_channel


if __name__ == "__main__":
    data_file = "channel_data/CDL_channel.h5"
    channel_data = read_channel_data(data_file)
    print(channel_data.shape)
