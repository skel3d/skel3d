import json

# Read the first JSON file
with open("/objaverse/valid_paths.json", "r") as file:
    data1 = json.load(file)

# Read the second JSON file
with open("/home/user/objaverse-rendering_zero/output/valid_paths.json", "r") as file:
    data2 = json.load(file)

# Find the common elements
common_elements = list(set(data1) & set(data2))


print(len(common_elements))

# Write the common elements into a third JSON file
with open("common_elements.json", "w") as file:
    json.dump(common_elements, file)
