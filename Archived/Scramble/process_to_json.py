# Shoutout to Microsoft Copilot for providing some of code templates for me to base upon

import os
import shutil
import random
import json


json_training_path = "training_source.json"
json_test_path = "testing_source.json"

bee_file_list = os.listdir("Raw Bee")
three_file_list = os.listdir("Raw Three")

bee_id_list = [f"Bee Training #{i}" for i in range(1, len(bee_file_list) + 1)]
three_id_list = [f"Three Training #{i}" for i in range(1, len(three_file_list) + 1)]

bee_file_dict = dict(zip(bee_id_list, bee_file_list))
three_file_dict = dict(zip(three_id_list, three_file_list))

combined_file_dict = {**bee_file_dict, **three_file_dict}
key_list = list(combined_file_dict.keys())
random.shuffle(key_list)

training_dict = {key: combined_file_dict[key] for key in key_list}

test_file_list = os.listdir("Test Dataset")
testing_dict = dict(zip([f"Bee Test #{i}" for i in range(1, 51)] + [f"Three Test #{i}" for i in range(1, 51)], test_file_list))

with open(json_training_path, "w") as f_train:
	json.dump(training_dict, f_train)

with open(json_test_path, "w") as f_test:
	json.dump(testing_dict, f_test)

# bee_id = 0
# three_id = 0
# for i in order:
#     if i == 0:
#         bee_id += 1

#         bee_file_name = next(bee_file_iter)
#         source_path = f"Raw Bee\\{bee_file_name}"
#         destination_path = f"Final\\{bee_file_name}"

#         shutil.copy(source_path, destination_path)
#         os.rename(destination_path, f"Final\\Bee {bee_id}.png")
#     else:
#         three_id += 1

#         three_file_name = next(three_file_iter)
#         source_path = f"Raw Three\\{three_file_name}"
#         destination_path = f"Final\\{three_file_name}"

#         shutil.copy(source_path, destination_path)
#         os.rename(destination_path, f"Final\\Three {three_id}.png")


# # rename bee_path
# for indx, filename in enumerate(bee_path):
#     new_filename = f"bee{indx + 1}.png"  # Modify the renaming pattern as needed
#     os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_filename))

# # rename three_path
# for indx, filename in enumerate(file_list):
#     new_filename = f"new_file_{indx}.txt"  # Modify the renaming pattern as needed
#     os.rename(bee_path + filename, bee_path + new_filename)

# combined_path = bee_file_list + three_file_list


# # Print a success message
# print(f"{len(file_list)} files in the directory were successfully renamed.")


