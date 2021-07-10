import logging
import time

from sdg.pytorch_code.agnostic_model import agnostic_model
from sdg.pytorch_code.training import train_model
from sdg.pytorch_code.training import fine_tune
from sdg.pytorch_code.earlystopping import stopping_args
from sdg.pytorch_code.propagation import PPRExact, PPRPowerIteration, SDG
from sdg.data.io import load_dataset


if __name__ == '__main__':

    logging.basicConfig(
            format='%(asctime)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO)

    graph_name = 'cora_ml'  # - altenative dataset 'citeseer' and 'pubmed' - #
    graph = load_dataset(graph_name)
    graph.standardize(select_lcc=True)

    # - Train PPNP for the initial inputs for SDG - #
    start_time = time.time()

    prop_ppnp = PPRExact(graph.adj_matrix, alpha=0.1)
    # prop_appnp = PPRPowerIteration(graph.adj_matrix, alpha=0.1, niter=10)

    model_args = {
        'hiddenunits': [64],
        'drop_prob': 0.5,
        'propagation': prop_ppnp}

    idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114}
    reg_lambda = 5e-3
    learning_rate = 0.01

    test = False
    device = 'cuda'
    print_interval = 20

    model, result = train_model(
            graph_name, agnostic_model, graph, model_args, learning_rate, reg_lambda,
            idx_split_args, stopping_args, test, device, None, print_interval)

    print('Training PPNP costs: ' + str(time.time() - start_time) + ' sec.')

    # - SDG receives PPNP and fine-tunes on the updated graph - #
    start_time = time.time()

    sdg = SDG(graph.adj_matrix, alpha=0.1).to(device)

    model_args = {
        'hiddenunits': [64],
        'drop_prob': 0.5,
        'propagation': sdg}

    model, result = fine_tune(
        graph_name, model, graph, model_args, learning_rate, reg_lambda,
        idx_split_args, stopping_args, test, device, None, print_interval)

    print('Generating the new graph + Training SDG costs: ' + str(time.time() - start_time) + ' sec.')