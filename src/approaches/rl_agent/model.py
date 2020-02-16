from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten

class ModelSelector():

	def __init__(self,input_size,output_size):
		self.input_size = input_size
		self.output_size = output_size


	def return_model(self, architecture):
		if architecture == 'dense':
			model = DenseSimple(self.input_size, self.output_size)
			return model.build()

		elif architecture == 'dense-deep':
			model = DenseDeep(self.input_size, self.output_size)
			return model.build()

		elif architecture == 'dense-wide':
			model = DenseWide(self.input_size, self.output_size)
			return model.build()

		elif architecture == 'resnet-50':
			model = ResNetFifty(self.input_size, self.output_size)
			return model.build()

		else:
			raise ValueError("You have supplied an unknown architecture.")


class DenseSimple():

	def __init__(self,input_size,output_size):
		self.input_size = input_size
		self.output_size = output_size

	
	def build(self):
		# building of simp
	    model = Sequential()
	    # model.add(Flatten(input_shape=(1,) + (self.input_size,)))
	    model.add(Dense(40,input_dim=self.input_size))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    # model.add(Dense(40))
	    # model.add(Activation('relu'))
	    # model.add(Dense(40))
	    # model.add(Activation('relu'))
	    # model.add(Dense(40))
	    # model.add(Activation('relu'))
	    # model.add(Dense(40))
	    # model.add(Activation('relu'))
	    # model.add(Dense(40))
	    # model.add(Activation('relu'))
	    # model.add(Dense(40))
	    # model.add(Activation('relu'))
	    # model.add(Dense(40))
	    # model.add(Activation('relu'))
	    model.add(Dense(self.output_size))
	    model.add(Activation('softmax'))

	    return model

class DenseDeep():

	def __init__(self,input_size,output_size):
		self.input_size = input_size
		self.output_size = output_size


		
	def build(self):
		# building of simp
	    model = Sequential()
	    model.add(Flatten(input_shape=(1,) + (self.input_size,)))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(40))
	    model.add(Activation('relu'))
	    model.add(Dense(self.output_size))
	    model.add(Activation('softmax'))

	    return model

class DenseWide():

	def __init__(self,input_size,output_size):
		self.input_size = input_size
		self.output_size = output_size

	
	def build(self):
		# building of simp
	    model = Sequential()
	    model.add(Flatten(input_shape=(1,) + (self.input_size,)))
	    model.add(Dense(80))
	    model.add(Activation('relu'))
	    model.add(Dense(80))
	    model.add(Activation('relu'))
	    model.add(Dense(80))
	    model.add(Activation('relu'))
	    model.add(Dense(80))
	    model.add(Activation('relu'))
	    model.add(Dense(80))
	    model.add(Activation('relu'))
	    model.add(Dense(80))
	    model.add(Activation('relu'))
	    model.add(Dense(80))
	    model.add(Activation('relu'))
	    model.add(Dense(80))
	    model.add(Activation('relu'))
	    model.add(Dense(80))
	    model.add(Activation('relu'))
	    model.add(Dense(80))
	    model.add(Activation('relu'))
	    model.add(Dense(self.output_size))
	    model.add(Activation('softmax'))


class ResNetFifty():
	def build(self):
		raise RuntimeError("Res net now available")