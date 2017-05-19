import numpy as np
import h5py
import os
import sys

import autoencoder


sys.argv = ['']


def setup_module():
    f = h5py.File('tests/test_data.mat', 'w')
    f['filt_AI_mat'] = np.random.random((8, 256, 256))
    f.close()


def teardown_module():
    os.remove('tests/test_data.mat')


def test_autoencoder_rnn(tmpdir):
    for num_layers in [1, 2]:
        for layer_type in ["gru"]:
            test_args = {"size": 4, "drop_frac": 0.25, "epochs": 1, "N_train":
                         5, "lr": 1e-3, "batch_size": 5, "embedding": 2,
                         "n_cycles": 16, "num_layers": num_layers,
                         "layer_type": layer_type, "log_dir": tmpdir,
                         "data_path": "tests/test_data.mat", "overwrite": True}
            test_args['sim_type'] = autoencoder.get_run_id(**test_args)
            autoencoder.main(test_args)
