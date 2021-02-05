import os
import json
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

root_path="training_files/"
model_name="kwyk_single_input_21-02-03_04-57"
file_name="data-"+ model_name +".json"

def plot_vars(dic):
    """ plot items in dic"""
    fig, ax = plt.subplots(1,len(dic), figsize=[11, 3])
    for i, key in enumerate(dic.keys()):
        data = dic[key]
        ax[i].plot(np.linspace(0,len(data),len(data)),data)
        ax[i].set_title(key)
        ax[i].set_xlabel("batch number")
        
    
path=os.path.join(root_path,model_name,file_name)

with open(path) as f:
    data=json.load(f)

plot_vars(data)
plt.show()


