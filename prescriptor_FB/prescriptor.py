import os

import utils
import importlib


from keras.models import model_from_json

from eml.net.reader import keras_reader
from eml.backend import cplex_backend
import docplex.mp.model as cpx
from eml.net.embed import encode
from eml import util

# Read the NN model
# Load the model architecture

net_prefixes = 'nn'


def load_keras_nets(knet):
    # Load scalar output NNs
    with open('nn.json') as fp:
        knet = model_from_json(fp.read())
    # Load the model weights
    wgt_fname = os.path.join('nn.h5')
    knet.load_weights(wgt_fname)

def convert_keras_net(knet, net):
    # Convert scalar-output NNs
    net = keras_reader.read_keras_sequential(knet)

importlib.reload(util)

def build_inout_vars(bounds, bkd, mdl):
    # Build one variable for each network input
    X_vars = []
    for i, bound in enumerate(bounds):
        X_vars.append(mdl.continuous_var(lb=0, ub=1, name='in_' + str(i)))
    return X_vars

# Build a backend object
bkd = cplex_backend.CplexBackend()
# Build a docplex model
mdl = cpx.Model()

knet = None
net = None
load_keras_nets(knet)
convert_keras_net(knet, net)

bounds = list(zip([0] * len(utils.IP_MAX_VALUES), [x + 1 for x in list(utils.IP_MAX_VALUES.values())]))
X_vars = build_inout_vars(bounds, bkd, mdl)
Z_0_var = mdl.continuous_var(lb=0, ub=float('inf'), name='out')
Z_1_var = mdl.sum(X_vars)

encode(bkd,
       net,
       mdl,
       X_vars,
       Z_0_var,
       'nn')

alpha = 0.5

# PROBLEM
R_var = mdl.continuous_var(lb=-float('inf'), ub=float('inf'), name='obj')
mdl.add_constraint(R_var == alpha * Z_0_var + (1 - alpha)  * Z_1_var)

mdl.set_objective('min', R_var)

# mdl.set_time_limit(30)
print('=== Starting the solution process')
sol = mdl.solve()
if sol is None:
    print('No solution found')
else:
    print('=== SOLUTION DATA')
    print('=== PROBLEM')
    print(sol)

