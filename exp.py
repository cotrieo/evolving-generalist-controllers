import json
from utils import generate_morphologies
from evo_process import Algo
import os
import sys


def experiment_run(config):
    runs = config['runs']
    parameter1_range, parameter2_range = config['parameter1'], config['parameter2']
    step_sizes = config['step_sizes']
    variations = generate_morphologies(parameter1_range, parameter2_range, step_sizes)

    for i in range(runs):
        cluster_count = 0
        generations = config['generations']
        folder_name = config['filename']

        while len(variations) != 0:
            cluster_count += 1
            path = f"{folder_name}/"
            os.makedirs(path, exist_ok=True)
            run = Algo(game=config['game'], path=path, xml_path=config['xml'], variations=variations,
                       config=config, generation=generations, run_id=i, cluster_id=cluster_count)
            generation, variations = run.main()
            generations = generations - generation


with open(str(sys.argv[1])) as json_file:
    print('Running experiment for ', sys.argv[1])
    config = json.load(json_file)
    experiment_run(config)
