import torch
import argparse
from pathlib import Path

data_type = ['xyz','feature_dc_index','quant_cholesky_elements']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args(argv):
    return None

def chamfer_disatance(pc1, pc2):
    return None

def main(argv):
    args = parse_args(argv)
    data_path = args.dataset·
    output_path = args.output
    



