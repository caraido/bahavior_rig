audio_settings = {
    'fs': 3e5,  # sample rate TODO: find a better frequency for the fft e.g. 2^18 ~ 262k
    'fMin': 200,
    'fMax': 40000,
    'nFreq': 1e2,  # number of frequencies to plot
    'fScale': 'log',  # frequency spacing, linear or log
    'window': .0032,  # length of window in seconds ~ this is 960 -> 1024?
    'overlap': .675,  # fractional overlap
    'correction': True,  # whether to correct for 1/f noise
    'readRate': 1,  # how frequently to read data from the Daq's off-board data buffer
    # TODO: rename readRate to readPeriod

    # notes on parameters:

    # window, overlap, fMin, fMax are taken from deepsqueak
    # fs needs to be at least twice the highest frequency of interest, ideally higher
    # window*fs should be a power of 2 (or at least even) for optimal computation of fft
    # the higher the readRate, the better the performance, but the latency of plotting increases

    # nFreq determines the number of frequencies that are plotted (by cubic interpolation), not calculated
    # the browser also performs some interpolation, so nFreq should be as low as possible to see features of interest
}
