import random
from math import exp
from copy import deepcopy


class NeuralNetworkException(Exception):
	pass


class MatrixException(Exception):
	pass


class LayerException(Exception):
	pass


class Matrix:

	def __init__(self, rows, columns):
		self.rows 	 = rows # Number of rows in the matrix
		self.columns = columns # Number of columns in the matrix
		# Do not remove
		self.matrix	 = [ ] # The actual Matrix which is just a list of lists

		for row in range(self.rows):
			# Append a list of n( self.columns ) for each row in the Matrix forming a size self.rows x self.columns matrix
			self.matrix.append( [ 0  for column in range(self.columns) ] )
			# self.matrix.append( [ 0 ] * self.columns )

	def setValue(self, row, column, value):

		if not isinstance(value, (int, long, float)):
			raise MatrixException("The value supplied is not a number")

		if self.rows > row and self.columns > column:
			self.matrix[row][column] = value
		else:
			raise MatrixException("Cannot set value, '{}', at {} {}: Out of index".format( value, row, column) )

	def scalarMultiplication(self, scalar):
		if not isinstance(scalar, (int, long, float)):
			raise MatrixException("The scalar supplied is not a number")

		for row in range(self.rows):
			for column in range(self.columns):
				self.matrix[row][column] *= scalar

	# 	Could potentially use the map function to achieve the same results without repeating code

	def transpose(self):
		# Tranposing a matrix changes it's shape

		rows 	= self.columns
		columns = self.rows
		matrix 	= []


		# Need to improve this

		for row in range(rows):
			matrix.append([ 0 for column in xrange(columns) ])

		for row in range(self.rows):
			for column in range(self.columns):
				matrix[column][row] = self.matrix[row][column]

		self.matrix 	= matrix
		self.rows 		= rows
		self.columns	= columns

	def add(self, matrix):
		if not isinstance(matrix, Matrix):
			raise MatrixException("Matrices can only add other Matrices")
		elif matrix.getRows() != self.rows or matrix.getColumns() != self.columns:
			raise MatrixException("Addition operation cannot be performed between matrices of shape {} and {}".format( self.shape(), matrix.shape() ))

		for row in range(self.rows):
			for column in range(self.columns):
				self.matrix[row][column] += matrix.matrix[row][column]

	# 	Map could also be performed here also

	def elementWiseMultiplication(self, matrix):
		if not isinstance(matrix, Matrix):
			raise MatrixException("Matrices can only add other Matrices")
		elif matrix.getRows() != self.rows or matrix.getColumns() != self.columns:
			raise MatrixException("Addition operation can only be performed on matrices of the same shape")

		for row in range(self.rows):
			for column in range(self.columns):
				self.matrix[row][column] *= matrix.matrix[row][column]

	def subtract(self, matrix):
		matrix.scalarMultiplication(-1)
		self.add(matrix)
		matrix.scalarMultiplication(-1)

	def multiply(self, matrix):
		if not isinstance(matrix, Matrix):
			raise MatrixException("Matrix multiplication is between two matrices a matrix was not provided")
		elif self.getColumns() != matrix.getRows():
			raise MatrixException("Cannot multiply matrices of shape {} with matrix of shape {}".format(self.shape(), matrix.shape()))

		columns = matrix.getColumns()
		productMatrix = Matrix(self.rows, columns)


		for i in range(len(self.matrix)):
			for j in range(len(matrix.matrix[0])):
				for k in range(len(matrix.matrix)):
					productMatrix.matrix[i][j] += self.matrix[i][k] * matrix.matrix[k][j]

		return productMatrix


	def getRows(self):
		return self.rows

	def getColumns(self):
		return self.columns

	def map(self, function):

		for row in range(self.rows):
			for column in range(self.columns):
				self.matrix[row][column] = function(self.matrix[row][column])

	def shape(self):
		return "{}x{}".format(self.rows, self.columns)

	def __str__(self):

		returnString = "Matrix of {} rows, and {} columns\r\n[\r\n".format(self.rows, self.columns)

		for row in range(self.rows):
			returnString += " ".join([ str(value) for value in self.matrix[row] ] )
			returnString += "\r\n"

		returnString += "]"

		return returnString

	def randomUniform(self, element):
		return random.uniform(-1,1)

	def randomize(self):
		self.map(self.randomUniform)


class Layer:

	def __init__(self, activation, name='Input'):
		self.weightMatrix = None
		self.inputMatrix  = None
		self.errorMatrix  = None
		self.biasMatrix   = None
		self.deltaWeight  = None
		self.outputMatrix = None
		self.activation   = activation
		self.name 		  = name

	def setInputs(self, inputMatrix):
		if isinstance(inputMatrix, Matrix):
			self.inputMatrix = inputMatrix
		elif isinstance(inputMatrix, list):
			self.inputMatrix = Matrix(len(inputMatrix), 1)

			for i, x in enumerate(inputMatrix):
				self.inputMatrix.setValue(i,0, x)

		else:
			raise LayerException("Cannot set inputs to {}: Unsupported datatype, please supply a list or an nx1 Matrix".format(inputMatrix))

	def initialiseWeights(self, outputNum):
		if self.weightMatrix is not None: return
		self.weightMatrix = Matrix(outputNum, self.inputMatrix.getRows())
		self.weightMatrix.randomize()
		self.biasMatrix = Matrix(outputNum, 1)
		self.biasMatrix.randomize()


	def output(self):
		outputMatrix = Matrix.multiply(self.weightMatrix, self.inputMatrix)
		outputMatrix.add(self.biasMatrix)
		outputMatrix.map(self.activation)
		self.outputMatrix = outputMatrix
		return outputMatrix

	def modifyWeights(self, additionMatrix):
		self.weightMatrix.add(additionMatrix)
	def modifyBiases(self, additionMatrix):
		self.biasMatrix.add(additionMatrix)
	def addError(self, errorMatrix):
		self.errorMatrix = errorMatrix

class NeuralNetwork:

	def __init__(self, inputs, hiddenLayers, outputNodes, nodes=3):

		self.hiddenLayers = []
		self.inputLayer   = None
		self.outputLayer  = None
		self.outputNodes  = 0
		self.LearningRate = -0.1
		self.NeuronList	  = None
		self.Nodes 		  = nodes
		self.output 	  = None

		if not isinstance(inputs, list):
			raise NeuralNetworkException("Invalid data type")
		elif not isinstance(outputNodes, int):
			raise NeuralNetworkException("Invalid data type")
		elif not isinstance(hiddenLayers, int):
			raise NeuralNetworkException("Invalid data type")

		# Need to redesign this
		# the issue here is that with the current implementation, the user would have to supply the first
		# input in their list of inputs to the initialiser and the return to the train method

		self.inputLayer = Layer(self.sigmoid)
		self.inputLayer.setInputs(inputs)
		self.inputLayer.initialiseWeights(self.Nodes)

		self.outputNodes = outputNodes

		for i in range(hiddenLayers):
			self.hiddenLayers.append(Layer(self.sigmoid, name='hidden_layer_{}'.format(i)))

		self.outputLayer = Layer(self.sigmoid, name='Output')

		# self.outputLayer.initialiseWeights(outputNodes)

	def feedforward(self):
		outputs = [ self.inputLayer.output() ]

		for i, layer in enumerate(self.hiddenLayers):

			# print outputs[i]
			layer.setInputs(outputs[i])
			layer.initialiseWeights(self.Nodes)
			outputs.append(layer.output())

		self.outputLayer.setInputs(outputs[-1])
		self.outputLayer.initialiseWeights(self.outputNodes)

		# print self.outputLayer.weightMatrix

		outputs.append(self.outputLayer.output())

		# print "Final output of the feedforward -> {}".format(outputs[-1])
		self.output = outputs[-1]
		return self.output

	def train(self, inputs, outputs):

		if not isinstance(inputs, list) or not isinstance(outputs, list):
			raise NeuralNetworkException("The input or output are not the right datatypes they must be lists")
		# Check if the list of inputs supplied is equal to the lists of outputs
		if len(inputs) != len(outputs):
			raise NeuralNetworkException("The number of inputs and outputs differ")

		for i in xrange(10000):

			outputMatrices = []

			for output in outputs:
				outputMatrices.append(Matrix(len(output), 1))

				for ij, x in enumerate(output):
					outputMatrices[-1].setValue(ij, 0, x)

			for ix, x in enumerate(inputs):
				self.inputLayer.setInputs(x)
				output = self.feedforward()
				outputMatrices[ix].subtract(output)  # Calculating the error here -> target -  output
				print "Total net error: {}".format(outputMatrices[ix].matrix[0][0])
				self.output.map( self.sigmoid_prime )
				self.backpropagate(outputMatrices[ix])  # Backpropagate the error through the layers

	def backpropagate(self, errorMatrix):

		errors = [errorMatrix]  # list of errors

		# print errors[0]
		self.NeuronList = [self.inputLayer] + self.hiddenLayers + [self.outputLayer]

		# print self.NeuronList

		for i, layer in enumerate(reversed(self.NeuronList)):

			# layer.weightMatrix.transpose()

			# error = layer.weightMatrix.multiply(errors[-1]) # this is wrong

			error = errors[-1].multiply(layer.weightMatrix)
			error.scalarMultiplication( self.output.matrix[0][0] )

			# layer.outputMatrix.transpose()
			if layer.outputMatrix.getRows() == 1 and layer.outputMatrix.getColumns() == 1:
				error.scalarMultiplication( layer.outputMatrix.matrix[0][0] )
				delta = error
			else:
				delta = error.multiply( layer.outputMatrix )
			delta.scalarMultiplication( self.LearningRate )
			# layer.outputMatrix.transpose()

			errors.append(error)
			layer.addError(errors[-1])

			# layer.weightMatrix.transpose()



	def predict(self, inputValue):
		self.inputLayer.setInputs(inputValue)
		output = self.feedforward()
		# print output

		return output

	def cost(self, yic, y):
		return 0.5*((yic - y)**2)

	def filler_activator(self, x):
		return x
	def sigmoid(self, x):
		return 1/(1+exp(-x))

	def sigmoid_prime(self, y):
		return y*(1 - y)


# NeuralNetwork(self, inputs, hiddenLayers, outputNodes, nodes=3):

NN = NeuralNetwork([0, 1], 2, 1, nodes=2)

inputs = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1]
]

outputs = [
	[0],
	[1],
	[1],
	[0]
]

# outputs = [
# 	[1],
# 	[0],
# 	[0],
# 	[1]
# ]


NN.train(inputs, outputs)
output = NN.predict([0, 0])
print output
output = NN.predict([0, 1])
print output
output = NN.predict([1, 1])
print output
