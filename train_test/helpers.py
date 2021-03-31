import re


def sorted_alphanumeric(path):
    """The function takes a directory path, and sorts the file names in that directory

    #Arguments
        path (str): Path of the directory in which the name of the files to be sorted
    #Returns
        sorted_path (list): List that contains the sorted file names in that directory
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    sorted_path = sorted(path, key=alphanum_key)
    return sorted_path
