#############################
## author: G. Räuber, 2024 ##
#############################

import json

dict_inputs = {}
dict_inputs['features'] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
dict_inputs['hyperparameters'] = []

with open("inputs.json", 'w') as json_file:
    json.dump(dict_inputs, json_file, indent=4)
