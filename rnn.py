from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks import Softmax, Rectifier, NDimensionalSoftmax, Linear
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.model import Model

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
from fuel.datasets.hdf5 import H5PYDataset
from fuel.datasets import Dataset
from fuel.transformers import Mapping

from theano.tensor import tensor3

# network parameters
max_epochs = 3
batch_size = 1
test_batch_size = 1
in_dim = 79
h_dim = 79
o_dim = in_dim
REC_NON_LIN = Rectifier
init_weights = IsotropicGaussian(.01)
init_biases = Constant(0)


class Transformer(object):

    @staticmethod
    def transpose_stream(data):
        return (data[0].swapaxes(0,1), data[1].swapaxes(0,1))


if __name__ == '__main__':

    x = tensor3('x')  # the input, given by training data
    y = tensor3('y')  # the targets, given by training data

    x_to_h = Linear(name='i_to_h', input_dim=in_dim, output_dim=h_dim,
                    weights_init=init_weights, biases_init=init_biases, seed=17)

    h = SimpleRecurrent(h_dim, REC_NON_LIN()).apply(x_to_h.apply(x))

    h_to_o = Linear(name='h_to_o', input_dim=h_dim, output_dim=o_dim,
                    weights_init=init_weights, biases_init=init_biases, seed=21)

    final = NDimensionalSoftmax(name='softmax')
    y_hat = final.apply(h_to_o.apply(h), extra_ndim=1)  # the ys computed by the network

    cost = final.categorical_cross_entropy(y=y, x=h, brick=h_to_o, extra_ndim=1).mean()
    cost.name = 'cost'

    cg = ComputationGraph([cost])

    print(cg.parameters)

    # MainLoop
    data_set = H5PYDataset('training_data.hdf5', ('train',), load_in_memory=True)
    data_stream = DataStream.default_stream(data_set, iteration_scheme=SequentialScheme(data_set.num_examples,
                                                                                        batch_size=batch_size))

    data_stream = Mapping(data_stream, Transformer.transpose_stream)

    algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Scale(learning_rate=0.1))
    test_set = H5PYDataset('training_data.hdf5', ('test',), load_in_memory=True)
    data_stream_test = DataStream.default_stream(test_set,
                                                 iteration_scheme=SequentialScheme(test_set.num_examples,
                                                                                   batch_size=test_batch_size))

    monitor = DataStreamMonitoring(variables=[cost], data_stream=data_stream_test, prefix='test')
    tr_monitor = TrainingDataMonitoring(variables=[cost], prefix='train', after_epoch=True)
    checkpoint = Checkpoint('model.bin')  #, save_separately=['model'])

    main_loop = MainLoop(data_stream=data_stream, model=cg, algorithm=algorithm,
                         extensions=[monitor, tr_monitor, FinishAfter(after_n_epochs=max_epochs), Printing(), checkpoint])

    main_loop.run()