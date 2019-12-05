import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--dataset', type=str, default='VisDrone',
                        choices=['VisDrone'], help='dataset name')
    parser.add_argument('--db_root', type=str, default="/home/visitor1/data/Visdrone",
                        help="dataset's root path")
    parser.add_argument('--imgsets', type=str, default=['train', 'val'],
                        nargs='+', help='for train or test')
    parser.add_argument('--padding', type=str, default=['train', 'val'],
                        nargs='+', help='random padding neglect box')
    args = parser.parse_args()
    return args

args = parse_args()

print(args)