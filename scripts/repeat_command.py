import os
import argparse

parser = argparse.ArgumentParser("A script for repeat a command for n times.")
parser.add_argument("--command", type=str, required=True)
parser.add_argument("--times", type=int, required=True)
args = parser.parse_args()
assert args.times > 0

for _ in range(args.times):
    os.system(args.command)
