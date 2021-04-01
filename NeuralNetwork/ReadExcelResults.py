import os
import re
import pandas as pd

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def get_file_names():
    file_names = []
    print(os.getcwd())
    for file in os.listdir("NeuralNetwork/Results/April_ImprovedAE"):
        filename, file_extension = os.path.splitext(file)
        file_names.append(filename)
    file_names.sort(key=alphanum_key)

    return file_names
if __name__ =="__main__":
    result_files = get_file_names()
    last_file_loss = result_files[-2]
    df = pd.read_csv("NeuralNetwork/Results/April_ImprovedAE/" + last_file_loss + ".csv")
    best_epoch = df.idxmin().values[0]
    print("test")