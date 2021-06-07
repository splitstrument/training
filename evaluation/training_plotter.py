import argparse
import util
from helperutils.boolean_argparse import str2bool

parser = argparse.ArgumentParser(description='Plot information about training session from a csv')
parser.add_argument('--training-folders', required=True, nargs='*',
                    help='paths to folders to plot data for')
parser.add_argument('--save-figure', type=str2bool, default=True,
                    help='whether to save the plots as image instead of showing them')
args = parser.parse_args()

for folder in args.training_folders:
    util.plot_training(folder, args.save_figure)
