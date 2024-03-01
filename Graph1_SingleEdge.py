#    Graph 1 Single Edge (Baseline)   #
# ------------------------------------#
# Node Features: Speeds from Past Hour
# Edges Included: Type 1
# Edge Types: Not Learned

# import torch
import numpy as np
import pandas as pd
# from torch_geometric.loader import DataLoader
# from torch_geometric.data import InMemoryDataset, Data
# from math import cos, asin, sqrt, pi
# import torch.optim as optim
# from tqdm import tqdm
# import time
# import os
# import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
# import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv
# from datetime import datetime

vds_info = pd.read_csv('./data/vds_info_w_lanes.csv').set_index('vds_id')
# sensor_speed = pd.read_csv('sensor_speed.csv').set_index('vds_id')
# sensor_dist = pd.read_csv('sensor_dist.csv').set_index('Unnamed: 0')
# sensor_conn = pd.read_csv('sensor_conn.csv').set_index('Unnamed: 0')
# non_conn = pd.read_csv('non_conn.csv').set_index('Unnamed: 0')
print(vds_info.shape)