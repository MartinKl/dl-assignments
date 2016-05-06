from blocks.serialization import load
from blocks.filter import VariableFilter

from theano import function

from rnn import Transformer

f = open('model.bin', 'rb')
main_loop = load(f)
model = main_loop.model
outname = 'softmax_log_probabilities_output'
xname = 'x'
soft_outs = None
x = None
i = 0
while (not soft_outs or not x) and i<len(model.variables):
    var_i = model.variables[i]
    soft_outs = var_i if var_i.name == outname else None
    x = var_i if var_i.name == xname else None
    if var_i.name == outname:
        soft_outs = var_i
    elif var_i.name == xname:
        x = var_i
    i += 1

f = function(x, soft_outs)