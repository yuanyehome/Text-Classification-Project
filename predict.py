import torch
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate prediction for test dataset.")
    parser.add_argument("--testset", type=str, required=True,
                        help="Location of test file.")
    parser.add_argument("--model", type=str, required=True,
                        help="Location of model file.")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
