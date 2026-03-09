import EMGFlow

# Set processing paths
# NOTE: This locates paths relative to your current working directory
#       (Relative to the terminal, not the file running)
path_names = EMGFlow.make_paths()

# Set key parameters
sampling_rate = 2000
cols = ['EMG_zyg', 'EMG_cor']
notch_vals = [(50,5), (150,25), (250,25), (350,25), (400,25), (450,25), (550,25), (650,25), (750,25), (850,25), (950,25)]
notch_sc = [(317, 25)]
notch_sc_reg = '(\\\\08\\\\|\\\\11\\\\)'

# Run processing workflow
EMGFlow.notch_filter_signals(path_names['raw'], path_names['notch'], cols, sampling_rate, notch_vals)
EMGFlow.notch_filter_signals(path_names['notch'], path_names['notch'], cols, sampling_rate, notch_sc, expression=notch_sc_reg)
EMGFlow.bandpass_filter_signals(path_names['notch'], path_names['bandpass'], cols, sampling_rate)
EMGFlow.rectify_signals(path_names['bandpass'], path_names['fwr'], cols)

# Run feature extraction
EMGFlow.extract_features(path_names, cols, sampling_rate)