import numpy as np
import torch as th

import os, argparse

parser = argparse.ArgumentParser(description='check rank')
parser.add_argument('-i', help='input', type=str, required=True)

