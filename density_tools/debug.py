
import fire
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="convert to chip dataset")
    parser.add_argument('--show', type=bool, default=False,
                        help="show image and chip box")
    args = parser.parse_args()
    return args


def train(**k):
    print(k)


if __name__ == '__main__':
    # train()
    args = parse_args()
    print(args)
    fire.Fire(train)