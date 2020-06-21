from mlp import NeuralNetwork
from genetic import Eugenicist

import numpy as np

inputs_x = [
	[0, 1],
	[1, 0],
	# [1, 1],
	[0, 0]
]

outputs_x = [
	[1],
	[1],
	# [1],
	[0]
]
inputs_y = []
outputs_y = []

for i in range( len( inputs_x ) ):
	inputs_y.append( np.asarray( [ inputs_x[i] ] ).T )
	outputs_y.append( np.asarray( [ outputs_x[i] ] ).T )


print inputs_y[0]
exit()

nn = NeuralNetwork

def fitness(Network):
	inputs = [[ 0.9 , 0.9 ]]
	inputs = np.asarray(inputs).T
	output =  [[ 1 ]]
	output = np.asarray(output).T
	fitness = ( 1/( output - Network.predict( inputs ) ) )**2
	return fitness[0,0]

parameters = { 'hidden_layers': 1, 'learning_rate': 0.1, 'hidden_layer_nodes': 50  }
eugenicist = Eugenicist( nn, parameters, inputs_y, outputs_y, progeny = 10, generations=5, fitness=fitness, upperlimit=5 )
configuration = eugenicist.propagate()[0]

print configuration['fitness']

nn = nn(**configuration['parameters'])

nn.train(inputs_y, outputs_y, plot=True, verbose=True)