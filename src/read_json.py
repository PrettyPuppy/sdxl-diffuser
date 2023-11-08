import json

def read_json(FileName):
    """
    Reads a JSON file and returns the data.
    
    Args:
    - filename (str): The name of the JSON file to read.
    
    Returns:
    - dict: The data from the JSON file.
    """
    # Load JSON file
    with open(FileName) as jsonFile:
        jsonData = json.load(jsonFile)

    # Return
    return jsonData
