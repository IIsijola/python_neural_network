from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import random

class Layer:

	"""

	Layer class is order to make feed forward and backpropagation somewhat easieer
	It's also the standard approach to neural networks
	The error matrix contains, or should contain, the error for each respective neuron in this layer
	weight matrix is the matrix of weights
	input matrix is the input of the layer
	deltaweight is the change in the weight matrix
	output matrix is the out of the layer ( weight matrix dot input matrix ) which is then squash, with the activation function, then fed to the next layer


	"""

	def __init__(self, activation, name='input_layer'):
		self.weightMatrix = None
		self.inputMatrix  = None
		self.errorMatrix  = None
		self.biasMatrix   = None
		self.deltaWeight  = None
		self.outputMatrix = None
		self.activation   = activation
		self.name 		  = name


	def setInputs(self, inputMatrix):
		if self._check_numpy_matrix(inputMatrix):
			self.inputMatrix = inputMatrix

	def inputSet(self):
		return self.inputMatrix is not None

	def weightSet(self):
		return self.weightMatrix is not None and self.biasMatrix is not None

	def initialise(self, outputNum):
		if self.inputSet() and self.biasMatrix is None and self.weightMatrix is None:
			self._initialiseWeights( outputNum )
			self._initialiseBiases( outputNum )
		# elif:
		# 	raise LayerException("Cannot initialise weights or biases without first setting input for layer.")

	# Initialise weights for the layer
	def _initialiseWeights(self, outputNum):
		input_rows = np.size(self.inputMatrix, 0)
		self.weightMatrix = np.random.randn(  outputNum, input_rows  )

	# Initialise biases for the layer
	def _initialiseBiases(self, outputNum):
		input_rows = np.size(self.inputMatrix, 0)
		self.biasMatrix = np.random.randn(  outputNum, 1  )

	# return output of the layer so that it can be fed forward
	def output(self):
		x = np.dot(self.weightMatrix, self.inputMatrix) + self.biasMatrix
		self.outputMatrix = self.activation( x )
		return self.outputMatrix

	# adjust weights by delta passed back from backpropagtion
	def adjustWeights(self, delta):
		self.weightMatrix = np.add( self.weightMatrix, delta )

	# adjust biases by deltas passed back from backpropagation
	def adjustBiases(self, delta):
		self.biasMatrix = np.add( self.biasMatrix, delta )

	# add error received from backpropagation
	def addError(self, error):
		self.errorMatrix = error

	def _check_numpy_matrix(self, matrix):
		if type(matrix).__module__ == np.__name__:
			return True
		else:
			raise TypeError("Expected a numpy array, not {}".format( matrix ) )

class NeuralNetwork:
# 0 hidden layers and 4-10 neurons are great
# 1 hidden layer and 100 nodes really good too
# 1 hidden layer and 10 nodes really good too
	def __init__(self, hidden_layers = 1, learning_rate = 0.05, output_nodes = 1, hidden_layer_nodes = 50):
		self.num_hidden_layers = hidden_layers
		self.hidden_nodes 	= hidden_layer_nodes
		self.learning_rate 	= learning_rate
		self.output_nodes	= output_nodes
		self.hidden_layers 	= []
		self.layers_ordered = []
		self.error_list		= []
		self.output 		= None

		# instantiate input layer
		self.input_layer = Layer( self._sigmoid )

		# instantiate each of the hidden layers 
		for hidden_layer_number in range( self.num_hidden_layers ):
			self.hidden_layers.append( Layer( self._sigmoid, name = 'hidden_layer_{}'.format( hidden_layer_number ) ) )

		# instatiate each of the output layers
		self.output_layer = Layer( self._sigmoid, name='output_layer' )
		self.layers_ordered = [ self.input_layer ] + self.hidden_layers + [ self.output_layer ]

	def feedforward(self, inputMatrix = None):
		if inputMatrix is not None:
			self.input_layer.setInputs( inputMatrix )
		
		if not self.input_layer.weightSet():
			print "Initialising input weights and biases"
			self.input_layer.initialise( self.hidden_nodes )

		outputs = [ self.input_layer.output() ]

		for layer_num, layer in enumerate(self.hidden_layers):
			layer.setInputs( outputs[-1] )

			if not layer.weightSet():
				layer.initialise( self.hidden_nodes )

			outputs.append( layer.output() )

		self.output_layer.setInputs( outputs[-1] )
		self.output_layer.initialise(self.output_nodes)
		self.output = self.output_layer.output()

		return self.output

	def backpropagation(self, error):

		# print "Number of layers: {}".format( len( self.layers_ordered ) )
		
		# first I need to propagate the error backwards.... or do I?

		#  Total error of the network (i.e) ( target - predicted )
		errors = [ error ]

		# iterating through the list of layers in reversed order

		# tmp = error * self.output * ( 1 - self.output )
		# tmp = self.learning_rate  * np.dot(tmp, self.input_layer.outputMatrix.T)

		# self.output_layer.adjustWeights( tmp )

		# hidden_errors = np.dot(self.output_layer.weightMatrix.T, error)
		# tmp = hidden_errors * self.input_layer.outputMatrix * (1.0 - self.input_layer.outputMatrix)

		# self.input_layer.adjustWeights( self.learning_rate * np.dot(tmp, self.input_layer.inputMatrix.T) )

		# return

		""" ************ IMPORTANT PART HERR ************ """

		# I initially thought that maybe the way i was back propagating was wrong asince my code is mant to work for an undermined amount hidden layers
		# so I set the hidden_layers = 0 so that there would only be two layers in my code: the input layer, and the output layer
		# then I didn the formula 'manually' in accordance to this, https://www.python-course.eu/neural_networks_with_python_numpy.php ,  tutorial
		# but still nothing seems to be improving can anyone please offer some insights here

		for layer in reversed( self.layers_ordered ):

			tmp  = errors[-1] * self._sigmoid_prime( layer.outputMatrix )
			tmp  = self.learning_rate * np.dot( tmp, layer.inputMatrix.T )

			layer.adjustWeights( tmp )
			layer.adjustBiases( errors[-1] * self.learning_rate )

			error = np.dot( layer.weightMatrix.T, errors[-1] )
			errors.append( error )

			# # Back propagtion takes place here 
			# # layer.biasMatrix = (errors[-1].T * self.learning_rate * 1)

			# # # This is were i think my maths goes absolutely wrong
			# # error = np.dot( errors[-1], layer.weightMatrix) * self._sigmoid_prime( self.output )	
			
			# # errors.append( error )
			# # layer.addError( error )


			# # delta = self.learning_rate * error * layer.outputMatrix.T
			
			# print "Input matrix"
			# print layer.inputMatrix
			# print
			# print "Error matrix"
			# print error
			# print
			# print "Delta"
			# print
			# print delta
			# print
			# print "Original weights"
			# print
			# print layer.weightMatrix
			# # layer.adjustWeights( delta )
			# print
			# print "Adjusted weights"
			# print 
			# print layer.weightMatrix

	def train(self, list_of_inputs, list_of_outputs, epoch=1000, plot=False):
		range_of_inputs = range(len(list_of_inputs))

		if plot is not False:
			plt.xlabel("iterations")
			plt.ylabel("Total Error")
			plt.title("Error function ")
			# colors = dict(**mcolors.CSS4_COLORS).values() # This is not working and is fucking irritating me
			colors = [ 'b', 'c', 'r', 'm' ]

		if range_of_inputs != range(len(list_of_outputs)):
			raise Exception("Number of training inputs is different from number of expected training outputs")

		for i in range(epoch):
			choice = random.choice(range_of_inputs)
			x = list_of_inputs[choice]
			y = list_of_outputs[choice]

			pred_output = self.feedforward(x)
			error = y - pred_output

			print "My prediction:{}".format(pred_output)
			print "Total Error:{}".format(error**2)

			self.backpropagation(error/len(range_of_inputs))

			if plot is not False:
				try:
					color = colors[choice]
					# print color
					plt.plot(i, error[0, 0], color + "+" )
				except Exception, e:
					print e
					plt.plot(i, error[0, 0])

		plt.show()


	def predict(self, inputMatrix=None):
		return self.feedforward(inputMatrix)

	def error(self, out):
		return self.error_list

	def _sigmoid(self, matrix):
		return 1. / ( 1. + np.exp(-matrix) )

	def _sigmoid_prime(self, matrix):
		return matrix * ( 1 - matrix )

