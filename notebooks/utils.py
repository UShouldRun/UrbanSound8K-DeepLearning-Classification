import json

def read_config(config_path: str) -> dict:
    """
    Reads a JSON file and returns the data as a dictionary.
    """
    with open(config_path, 'r') as json_file:
        data = json.load(json_file)
    return data
