import argparse
import os
import json
from util import print_evaluations

parser = argparse.ArgumentParser(description='pretty print an evaluation file from trainings')
parser.add_argument('--training-folders', nargs='*', required=True,
                    help='path to the folder containing training information')
args = parser.parse_args()

for training_folder in args.training_folders:
    configuration_file = os.path.join(training_folder, 'configuration.jsonc')
    with open(configuration_file, 'r') as file:
        configuration = json.loads(file.read())

    print_evaluations(training_folder, configuration)
