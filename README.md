# The AI behind the EE 
In this folder, it contains the files responsible for running of the AI used for the EE project. The main parts in this folder are the "Training Data" folder, "NeuralNetwork.py", and "setup.py". The rest are redundant and shouldn't be touched. This project uses PyTorch, Pillow, and some other libraries, but the first 2 are the one to check for.
## Running the built-in model
Inside the "setup.py" file contains the code to configure an artificial neural network (ANN), run it, and train it. To manipulate the amount of layer and the quantity of node per layer, modify the parameters on line 118. The first value is the depth, while the second value is the width. This is indicated by the comment next to it.
```
if __name__ == "__main__":
	SetUp = SetUp((60, 1)) # depth, width
	SetUp.run()
```

As for the "NeuralNetwork.py", it is what the "setup.py" uses to build, run, and train the neural network. "NeuralNetwork.py" only uses Pytorch unlike "setup.py". It's best to think that "setup.py" is an abstraction of "NeuralNetwork.py", which is the program that works in the background to get the ANN running. Like the "setup.py" program, it has a built-in AI that can be ran, though said AI is merely a prototype, so don't expect too much from it.

## Setting up a Neural Network from scratch with "NeuralNetwork.py"
First, initalize the NeuralNetwork object
```
NN = NeuralNetwork()
```
Next, create the layers. 
```
NN.create_layer("input layer", 2)
NN.create_layer("hidden layer", 2)
NN.create_layer("output layer", 2).is_output()
```
It should be mentioned that the datatype of the parameters are:
```
NN.create_layer(str, int) -> NeuralNetwork()
```
And it will output the object itself, which is NN--NeuralNetwork()--in this case. Another note is that when the layer is created, it will create a matrix filled with 0s. You will have to manually assign a matrix to the input layer using torch.tensor(), with the tensor's datatype setted as ```dtype=torch.float32``` like the example below:
```
NN["input layer"] = torch.tensor([[3],
                                  [1]], dtype=torch.float32)
```

Also, the output layer need to be specified using the is_output() method. This is for the program to know which layer to use to compute the loss function.

Then, the weights and biases need to be setted up
