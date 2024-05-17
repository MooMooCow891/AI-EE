import os
import shutil
import json

raw_bee_path = "Raw Bee"
bee_path = 'C:\\Users\\Lenovo\\Tony\\Code\\AI Img guesser test\\Original Source\\Bee\\Used'
raw_three_path = "Raw Three"
three_path = 'C:\\Users\\Lenovo\\Tony\\Code\\AI Img guesser test\\Original Source\\3\\Used'
test_path = "Test Dataset"

# <!-- Doesn't work for some reason -->
# with open("chosen_bee.txt", "r") as training_bee_txt:
# 	bee_training_info = training_bee_txt.read().replace("'", '"')

# with open("test_bee.txt", "r") as testing_bee_txt:
# 	bee_test_info = training_bee_txt.read().replace("'", '"')

# with open("chosen_three.txt", "r") as training_three_txt:
# 	three_training_info = training_three_txt.read().replace("'", '"')

# with open("test_three.txt", "r") as testing_three_txt:
# 	three_test_info = testing_three_txt.read().replace("'", '"')


# with open("chosen_bee.txt", "w") as training_bee_txt:
# 	training_bee_txt.write(bee_training_info)

# with open("test_bee.txt", "w") as testing_bee_txt:
# 	testing_bee_txt.write(bee_test_info)

# with open("chosen_three.txt", "w") as training_three_txt:
# 	training_three_txt.write(three_training_info)

# with open("test_three.txt", "w") as testing_three_txt:
# 	testing_three_txt.write(three_test_info)

# <!-- Automatically changes single-quotation marks to double-quotation marks to fix json errors -->
# def fix(path):
# 	f_read = open(path, "r")
# 	info = f_read.read().replace("'", '"')
# 	f_read.close()

# 	f_write = open(path, "w")
# 	f_write.write(info)
# 	f_write.close()

# for i in iter(["chosen_bee.txt", "test_bee.txt", "chosen_three.txt", "test_three.txt"]):
# 	fix(i)

def get_list(path: str) -> list:
	f = open(path, "r")
	array = json.loads(f.read())
	f.close()

	return array

def move(source: str, destination: str, array: list) -> None:
	for file_name in array:
		shutil.copy(f"{source}\\{file_name}", f"{destination}\\{file_name}")

for indx, i in enumerate([(bee_path, raw_bee_path, get_list("chosen_bee.txt")), (three_path, raw_three_path, get_list("chosen_three.txt")), (bee_path, test_path, get_list("test_bee.txt")), (three_path, test_path, get_list("test_three.txt"))]):
	print(indx)
	move(*i)