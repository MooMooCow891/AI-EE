import json

with open("Training Data\\training_source copy.json", "r") as f:
    stuff: dict = json.load(f)

three = 1
bee = 1

new_dict = {}
for key, value in stuff.items():
    if "Three" in key:
        new_dict[f"Three Training #{three}"] = value
        three += 1
    else:
        new_dict[f"Bee Training #{bee}"] = value
        bee += 1

with open("training_source.json", "w") as f_write:
    json.dump(new_dict, f_write)