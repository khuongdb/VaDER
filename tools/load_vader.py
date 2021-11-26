import os
import sys
import argparse
import numpy as np
import pandas as pd
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from typing import Tuple
from collections import Counter
from vader import VADER
from vader.utils.data_utils import generate_wtensor_from_xtensor
from vader.utils.plot_utils import plot_z_scores, plot_loss_history
from vader.utils.clustering_utils import ClusteringUtils

if __name__ == "__main__":
    """
    The script runs VaDER model with a given set of hyperparameters on given data.
    It computes clustering for the given data and writes it to a report file.

    Example:
    python load_vader.py --input_data_file=../data/ADNI/Xnorm.csv
                         --data_reader_script=addons/data_reader/adni_norm_data.py
                         --output_path=../vader_results/clustering/
                         --load_vader_path=../model/ADNI_vader_model                      
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str, help="a .csv file with input data", required=True)
    parser.add_argument("--input_weights_file", type=str, help="a .csv file with flags for missing values")
    parser.add_argument("--data_reader_script", type=str, help="python script declaring data reader class")
    parser.add_argument("--load_path", type=str, required=True, help="model load path")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_data_file):
        print("ERROR: input data file does not exist")
        sys.exit(1)

    if args.input_weights_file and not os.path.exists(args.input_weights_file):
        print("ERROR: weights data file does not exist")
        sys.exit(2)

    if args.data_reader_script and not os.path.exists(args.data_reader_script):
        print("ERROR: data reader file does not exist")
        sys.exit(3)

    if not os.path.exists(args.load_path):
        print("ERROR: vader model does not exist")
        sys.exit(4)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # dynamically import data reader
    data_reader_spec = importlib.util.spec_from_file_location("data_reader", args.data_reader_script)
    data_reader_module = importlib.util.module_from_spec(data_reader_spec)
    data_reader_spec.loader.exec_module(data_reader_module)
    data_reader = data_reader_module.DataReader()

    x_tensor = data_reader.read_data(args.input_data_file)
    w_tensor = generate_wtensor_from_xtensor(x_tensor)
    input_data = np.nan_to_num(x_tensor)
    input_weights = w_tensor
    features = data_reader.features
    time_points = data_reader.time_points
    x_label = data_reader.time_point_meaning
    ids_list = data_reader.ids_list

    vader = VADER.load_model(args.load_path, input_data, input_weights)
    n_hidden = [str(layer_size) for layer_size in vader.n_hidden]
    report_suffix = f"k{str(vader.K)}" \
                    f"_n_hidden{'_'.join(n_hidden)}" \
                    f"_learning_rate{str(vader.learning_rate)}" \
                    f"_batch_size{str(vader.batch_size)}" \
                    f"_n_epoch{str(vader.n_epoch)}" \
                    f"_seed{str(args.seed)}"
    plot_file_path = os.path.join(args.output_path, f"z_scores_trajectories_{report_suffix}.pdf")
    clustering_file_path = os.path.join(args.output_path, f"clustering_{report_suffix}.csv")
    clustering = vader.cluster(input_data, input_weights)
    pd.Series(list(clustering), index=ids_list, dtype=np.int64, name='Cluster').to_csv(clustering_file_path)

    if features and time_points:
        fig = plot_z_scores(x_tensor, clustering, list(features), time_points, x_label=x_label)
        fig.savefig(plot_file_path)
