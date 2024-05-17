import torch

class Tensor(torch.Tensor):
	def __new__(self, array, dtype=torch.float32, requires_grad=True):
		return torch.tensor(array, dtype=dtype, requires_grad=requires_grad)		

class Weights_Bias_Activation:
	def __init__(self):
		self.weights: torch.Tensor = None
		self.bias: torch.Tensor = None
		"""
		How to self.weights:
		self.weights = torch.Tensor([[1, 2],    ^
									 [3, 4],    |  y-axis: the amount of node 
									 [5, 6],    | 		   in the next layer
									 <---->     V     
							x-axis: the amount of node in 
									the current layer

		How to self.bias:
		It just needs to have the same amount of dimension to the next layer
		"""
		self.activation_func: "lambda func" or "others" = torch.sigmoid # setting a  default activation function # type: ignore

class Layer:
	def __init__(self):
		pass

class NeuralNetwork:
	def __init__(self):
		self.layers = {}
		"""
		Ex of self.layers =
		{
			"input_layer": None, "output_layer": torch.Tensor([[0],
															   [0],
															   [0]])
		}
		"""

		self.info_to_run = {}
		"""
		Ex of self.info_to_run =
		{("layer1", "layer2"): Weights_n_Bias() <-- has weight(s) and bias(s) in pytorch matrices, and activation function attributes}
		      --key--              --value--
		"""
		# for backpropagation
		self.output_layer = None
		self.loss_func = self._mean_squared_error
		self.expectation = None

		self.saved_layers = {}

	def __getitem__(self, key):
		if type(key) == tuple:
			return self.info_to_run[key] # self.info_to_run.__getitem__(key)
		else:
			self.key_accessed = key
			return self.layers[key] # self.layers.__getitem__(key)

	def __setitem__(self, key, value):
		if type(key) == tuple:
			self.info_to_run[key] = value
		else:
			self.layers[key] = value

	def create_layer(self, layer_name: str, node_amount: int, data_type=torch.float32) -> object:
		self.layers[layer_name] = torch.tensor([[0]] * node_amount, dtype=data_type)		
		self.created_layer_name = layer_name

		return self

	# establish an output layer for backpropagation purposes
	def is_output(self) -> None:
		self.output_layer = self.created_layer_name

	def connect_layer(self, layer_from: str, layer_to: str) -> None:
		self.info_to_run[(layer_from, layer_to)] = Weights_Bias_Activation()

	def _compute_layer(self, layer_from: str, layer_to: str) -> None:
		_weights, _bias, _activation_func = self.info_to_run[(layer_from, layer_to)].__dict__.values() # __dict__ gathers the attributes and values of the class as a dict

		assert type(_weights) == torch.Tensor
		assert type(_bias) == torch.Tensor or _bias == None
		assert type(self.layers[layer_to]) == torch.Tensor
		# assert type(self.layers[layer_from] == torch.Tensor)

		self.layers[layer_to] += _activation_func(self.layers[layer_to] + torch.matmul(_weights, self.layers[layer_from]) + (_bias if _bias != None else torch.tensor([0])))
				
	def forward(self) -> None:
		for connection in self.info_to_run: # iterate through the keys of self.info_to_run
			self._compute_layer(*connection)

	# built-in loss function
	def _mean_squared_error(self, expectation: torch.Tensor) -> torch.Tensor:
		assert type(self.layers[self.output_layer]) == torch.Tensor
		assert type(expectation) == torch.Tensor

		self.layers[self.output_layer] - expectation
		return sum(sum((lambda x: x ** 2)(self.layers[self.output_layer] - expectation)))
	
	def backpropagate(self, learning_rate: int or float) -> None: # type: ignore
		assert type(learning_rate) == int or type(learning_rate) == float
		assert type(self.expectation) == torch.Tensor

		self.loss: torch.Tensor = self.loss_func(self.expectation)
		assert type(self.loss) == torch.Tensor
		
		self.loss.backward()

		for Wei_Bias_Actv in self.info_to_run.values():
			_weights: torch.Tensor = Wei_Bias_Actv.weights
			assert type(_weights) == torch.Tensor
			# print(_weights)

			tempt = _weights - _weights.grad * learning_rate
			Wei_Bias_Actv.weights: torch.Tensor = tempt.detach().clone() # type: ignore
			Wei_Bias_Actv.weights.requires_grad: bool = True # type: ignore

			if (_bias := Wei_Bias_Actv.bias) != None:
				assert type(_bias) == torch.Tensor
				temp = _bias - (_bias.grad * learning_rate)
				Wei_Bias_Actv.bias: torch.Tensor = temp.detach().clone() # type: ignore
				Wei_Bias_Actv.bias.requires_grad: bool = True # type: ignore
				
	
	def save(self) -> None:
		# torch.Tensor is a mutable data type which is very troublesome.
		self.saved_layers = {key: value.clone() for key, value in self.layers.items()}
	
	def load(self) -> None:
		self.layers = {key: value.clone() for key, value in self.saved_layers.items()}


if __name__ == "__main__":
	NN = NeuralNetwork()

	# creating layers
	NN.create_layer("input layer", 2)
	NN.create_layer("hidden layer", 2)
	NN.create_layer("output layer", 2).is_output()

	print(f'{NN.layers}')


	# setting up the inputs
	# NN["input layer"] = torch.tensor([[3],
	# 								  [1]], dtype=torch.float32)
	
	NN["input layer"] = torch.tensor([[-1],
								 	  [4]], dtype=torch.float32)

	# connecting the layers
	NN.connect_layer("input layer", "hidden layer")
	NN.connect_layer("hidden layer", "output layer")

	# setting up the weight(s), bias(es), and the activation function between ("input layer" --> "hidden layer"), and ("hidden layer" --> "output layer")
	# NN[("input layer", "hidden layer")].weights = Tensor([[6, -2],
	# 													  [-3, 5]])

	# NN[("hidden layer", "output layer")].weights = Tensor([[1, 0.25],
	# 													   [-2, 2]])

	NN[("input layer", "hidden layer")].weights = Tensor([[2, 2],
														  [2, 2]])

	NN[("hidden layer", "output layer")].weights = Tensor([[2, 2],
														   [2, 2]])

	# NN[("hidden layer", "output layer")].bias = Tensor([[10],
	# 													[0]])
	NN.save()
	print(f'Before: {NN[("hidden layer", "output layer")].bias = }')
	NN.forward()
	print(NN.layers)
	print(f'After: {NN[("hidden layer", "output layer")].bias = }')
	NN.expectation = torch.tensor([[0],
								   [1]])
	NN.backpropagate(0.5)
	NN.load()
	NN.forward()
	print(f'{NN["input layer", "hidden layer"].weights = }')
	print(f'{NN["hidden layer", "output layer"].weights = }')
	print(NN.layers)
	NN.backpropagate(0.5)
	NN.load()
	NN.forward()
	print(f'{NN["input layer", "hidden layer"].weights = }')
	print(f'{NN["hidden layer", "output layer"].weights = }')
	print(NN.layers)
	NN.backpropagate(0.5)
	NN.load()
	NN.forward()
	print(f'{NN["input layer", "hidden layer"].weights = }')
	print(f'{NN["hidden layer", "output layer"].weights = }')
	print(NN.layers)




















	# print(f'{NN._mean_squared_error(NN.expectation)}')
	# print(f"{NN.expectation = }")
	# print(f'Before: {NN[("hidden layer", "output layer")].weights}')
	# print(f'Before: {NN[("hidden layer", "output layer")].bias}')
	# NN.backpropagate(0.5)
	# print(f'After: {NN[("hidden layer", "output layer")].weights}')
	# print(f'After: {NN[("hidden layer", "output layer")].bias}')
	# print(f'After: {NN[("hidden layer", "output layer")].bias.requires_grad}')
	# NN.load()
	# print(f"{NN.expectation = }")
	# # NN.get_error()
	# print(NN.layers)
	# NN.forward()

	# print("After----")
	# print(f'{NN["input layer"] = }')
	# print(f'{NN["hidden layer"] = }')
	# print(f'{NN["output layer"] = }\n')

	# print(f'{NN[("hidden layer", "output layer")].bias = }\n')
	# print(f"{NN.expectation = }")
	# print(f'{NN._mean_squared_error(NN.expectation)}')
	# NN.backpropagate(0.5)


	



# class Test:
# 	def __init__(self):
# 		self.info = {"a": 1}
# 	def __getattr__(self, attr):
# 		return self.info[attr]
# 	def __setattr__(self, attr, value):
# 		self.info[attr] = value
# 	def keys(self):
# 		return self.info.keys()
# 	def values(self):
# 		return self.info.values()
# 	def items(self):
# 		return self.info.items()







# import torch
# import torch.nn as nn

# class NeuralNetwork(nn.Module):
# 	def __init__(self):
# 		super().__init__()
