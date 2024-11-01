import matplotlib.pyplot as plt
import json

from lab1_utils import multiple_diagnostic


## Load learning rate scheduler results
file_name = "./plots/cifar10-opt-diagnostic"
with open(file_name + ".json", "r") as file:
    loaded_file = json.load(file)

multiple_diagnostic(loaded_file)
plt.savefig(file_name + ".pdf")
print("Plotted in", file_name + ".pdf")


## Load transforms results
file_name = "./plots/cifar10-transforms-diagnostic"
with open(file_name + ".json", "r") as file:
    loaded_file = json.load(file)

multiple_diagnostic(loaded_file)
plt.savefig(file_name + ".pdf")
print("Plotted in", file_name + ".pdf")


## Load full resnet with different training settings results
file_name = "./plots/cifar10-resnet-aug"
with open(file_name + ".json", "r") as file:
    loaded_file = json.load(file)

multiple_diagnostic(loaded_file)
plt.savefig(file_name + ".pdf")
print("Plotted in", file_name + ".pdf")


## 
