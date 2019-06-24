import pandas as pd
import numpy as np
import numpy.random as rnd
import scipy.stats as stat
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append("../../src/")

from scatter_plots import *

tcga_data = pd.read_csv("eig_stat_tcga.csv",  delimiter="\t")
scatter_from_df(tcga_data,'_T_','_R_','tcga_',200)
