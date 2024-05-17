import json
import torch
from PIL import Image
import random
from NeuralNetwork import NeuralNetwork, Tensor

# file_list = os.listdir(path)

class SetUp:
	def __init__(self, dimension: ("depth: int", "width: int")) -> None: # type: ignore
		self.bee_path: str = "Training Data\\Raw Bee"
		self.three_path: str = "Training Data\\Raw Three"
		self.test_path: str = "Training Data\\Test Dataset"
		self.training_source_json_path: str = "Training Data\\training_source.json"
		self.testing_source_json_path: str = "Training Data\\testing_source.json"

		with open(self.training_source_json_path, "r") as f_training:
			self.training: dict = json.load(f_training)

		with open(self.testing_source_json_path, "r") as f_testing:
			self.testing: dict = json.load(f_testing)

		self.hidden_layer_depth: int = dimension[0]
		self.hidden_layer_width: int = dimension[1]

		self.NN = NeuralNetwork()

		# 50 iterations/test
		self.NN.create_layer("input layer", 2500)

		for x in range(1, self.hidden_layer_depth + 1):
			self.NN.create_layer(f"hidden layer {x}", self.hidden_layer_width)

		self.NN.create_layer("output layer", 2).is_output()

		for indx, layer_name in enumerate((all_layers := tuple(self.NN.layers.keys()))[:-1]):
			connection_id = (layer_name, all_layers[indx + 1])

			self.NN.connect_layer(*connection_id)
			random_weight = lambda: random.choice([i/10 for i in range(-50, 51)])
			self.NN[connection_id].weights = Tensor(
				[
					[random_weight() for e in range(len(self.NN[layer_name]))]
					for i in range(len(self.NN[all_layers[indx + 1]]))
				]
			)
			self.NN[connection_id].bias = Tensor(
				[
					[random_weight()]
					for i in range(len(self.NN[all_layers[indx + 1]]))
				]
			)
		self.NN.save()


	def run(self) -> None:
		checkpoint_number = 1
		count = 0
		performance: list([{f"Checkpoint {int}": "percentage accuracy %"}]) = [] # type: ignore

		for classification, name in {key: self.training[key] for key in list(self.training.keys())[:1000]}.items():
		# for classification, name in {f"Three training #{i}": "12346.png" for i in range(200)}.items():
			self.NN.load()
			count += 1
			# 0 = Three; 1 = Bee
			id = (0 if "Three" in classification else 1)
			self.pass_input(
					self.get_pixel(path := (f'Training Data\\{("Raw Bee" if id else "Raw Three")}\\{name}'))
				)
			self.NN.forward()
			self.NN.expectation = (torch.tensor([[1], [0]]) if id else torch.tensor([[0], [1]]))
			self.learn(("bee" if id else "three"))
			if count % 50 == 0:
				correct: int = 0
				for key, value in self.testing.items():
					self.pass_input(
						self.get_pixel(path := (f'Training Data\\Test Dataset\\{value}'))
					)
					self.NN.forward()
					bee, three =  self.NN["output layer"]
					if bee > three and "Bee" in key:
						correct += 1
					elif three > bee and "Three" in key:
						correct += 1
					
					self.NN.load()

			
				performance.append(
					{f"Checkpoint {checkpoint_number}": f"{correct}%"}
				)

				checkpoint_number += 1
		print("-" * 10)
		print("~Performance~")
		for dictionary in performance:
			key, value = tuple(dictionary.items())[0]
			print(f"{key}: {value}")

	def pass_input(self, array: list) -> None: # type: ignore
		self.NN["input layer"] = torch.tensor([[i] for i in array])

	def get_output(self) -> torch.Tensor:
		return self.NN["output layer"]

	def learn(self, expectation: str) -> None: # type: ignore
		self.NN.expectation = (torch.tensor([[1], [0]]) if expectation == "bee" else torch.tensor([[0], [1]]))
		self.NN.backpropagate(0.1)

	def get_pixel(self, path: str) -> torch.Tensor:
		img = Image.open(path)
		img = img.resize((50, 50), Image.LANCZOS).convert('L')
		return torch.tensor(img.getdata(), dtype=torch.float32)
		


if __name__ == "__main__":
	SetUp = SetUp((60, 1)) # depth, width
	SetUp.run()