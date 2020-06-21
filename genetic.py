import random
import numpy as np 
from copy import deepcopy

class Eugenicist:

	def __init__( self, network, parameters, inputs, outputs, progeny = 100000, generations = 10, fitness = None, upperlimit = 2000, lowerlimit=0 ):
		self.progeny 		= progeny
		self.generations 	= generations
		self.parameters		= parameters
		self.neuralnet		= network
		self.networklist	= []
		self.inputs			= inputs
		self.outputs		= outputs
		self.numlist		= range(lowerlimit, upperlimit)
		self.threshold		= 0
		self.iterations		= 0
		self.fitness_func	= None

		if not isinstance( self.parameters, dict ):
			raise TypeError("Parameters are meant to be dictionaries")
		if callable( self.neuralnet	) is False:
			raise TypeError("Neural network is meant to be callable")
		if fitness is not None:
			if callable(fitness): 
				self.fitness_func = fitness
			else:
				raise TypeError("fitness function is meant to be callable")

	def propagate(self):
		print "Progeny: {}".format( self.progeny )
		print "iterations: {}".format( self.iterations )

		if self.iterations == 0:
			for i in range( self.progeny ):
				# Step one: create a population with random configurations
					parameters = self.random_progeny( )
					print parameters
					self.networklist.append(  { 'Network': self.neuralnet(**parameters), 'parameters': parameters } )

					# Step two: get the fitness of each member
					self.networklist[i]['Network'].train( self.inputs, self.outputs )
					self.networklist[i]['fitness'] = self.fitness( self.networklist[i]['Network'] )
					del self.networklist[i]['Network'] # save some memory


		if self.generations == self.iterations or len(self.networklist) <=3:
			fittest = self.fittest()
			return fittest


		# Number of progeny to keep the rest are killed like Abrahams sacrifice of Isaac to god.
		self.progeny = int( round( 0.9 * len(self.networklist) ) )

		# Step three: rank members in terms of fitness and then breed
		fittest = self.fittest()
		print "Fittest Network Score:{}".format(fittest[0]['fitness'])
		print "Network Configuration:"
		print fittest[0]['parameters']
		del self.networklist
		self.networklist = []

		print "Crossing Over"

		for offspringXX in fittest:
			for offspringXY in fittest:
				if offspringXX == offspringXY:
					continue
				progeny_parameters = self.crossover( offspringXX['parameters'], offspringXY['parameters'] )
				if random.randint(0,10) == 10: # every 10th child is mutated
					progeny_parameters = self.mutate( progeny_parameters )
				self.networklist.append( { 'Network': self.neuralnet(**progeny_parameters), 'parameters': progeny_parameters, 'fitness': 0 } )
				self.networklist[-1]['Network'].train( self.inputs, self.outputs )
				self.networklist[-1]['fitness'] = self.fitness( self.networklist[-1]['Network'] )
				if self.networklist[-1]['fitness'] < ( offspringXY['fitness']+ offspringXX['fitness'])/2:
					print "Child was weak and is sacrificed to cthulu"
					del self.networklist[-1]
		self.iterations += 1
		return self.propagate()

	def random_progeny(self, random_parameters = None):
		if random_parameters is None:
			random_parameters = self.parameters

		parameters = {}
		for parameter, variable in random_parameters.iteritems():
			if isinstance( variable, list ):
				parameters[ parameter ] = random.choice( variable )
			elif isinstance( variable, int ):
				parameters[ parameter ] = random.choice( self.numlist )
			elif isinstance( variable, float ):
				parameters[ parameter ] = random.random()
			elif isinstance( variable, bool ):
				parameters[ parameter ] = random.choice( [ True, False ] )
			else:
				raise TypeError("One of the parameters has an unrecognised data type: {}".format(type( variable )))

		return parameters

	def mutate(self, parameters):
		# previous algorithm used but might cause some issues with threading and is a bit fat
		# temp_parameters = deepcopy( self.parameters )
		# parameter_to_modify = temp_parameters.keys()[ random.randint( 0, len( temp_parameters.keys() ) - 1 ) ]
		# self.parameters = { parameter_to_modify: parameters[ parameter_to_modify ] }
		# parameters[ parameter_to_modify ] = self.random_progeny()[ parameter_to_modify ]
		# self.parameters = temp_parameters
		# del temp_parameters

		# much more elegant solution
		parameter, value = random.choice( parameters.items() )
		parameters[ parameter ] = self.random_progeny( { parameter: value } )[ parameter ]
		return parameters


	def crossover(self, parameterXX, parameterXY ):
		parameters = {}

		for parameter in self.parameters.keys():
			parameters[ parameter ] = random.choice( [ parameterXX[ parameter ], parameterXY[ parameter ] ] )
		return parameters

	def fitness(self, Network):
		if not self.fitness_func:
			""" Filler fitness function used for testing purposes soley now """
			return random.randint(0, 10)
		else:
			fitness = self.fitness_func( Network )
			if fitness == np.nan:
				fitness = 0
			return fitness

	def fittest(self):
		return sorted(self.networklist, key = lambda i: i['fitness'], reverse=True)[:self.progeny]


# class Network:

# 	def __init__(self, **kwargs):
# 		return
# 		print kwargs

# 	def train(self, *args):
# 		return
# 		print args

# parameters = { 'hidden_layers': 1, 'learning_rate': 0.1, 'hidden_layers_nodes': 50  }
# eugenicist = Eugenicist( Network, parameters, [], [] )

# # eugenicist.propagate()

# XX = eugenicist.random_progeny()
# XY = eugenicist.random_progeny()


# print XX
# print XY

# print eugenicist.crossover( XX, XY )

# print 
# print "Mutation"
# print eugenicist.mutate( XY )