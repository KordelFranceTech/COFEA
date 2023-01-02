class PickleException(Exception):
    def __init__(self):
        print("Pickle file could not be loaded due to incorrect parameters")