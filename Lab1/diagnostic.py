import matplotlib.pyplot as plt
import json

from lab1_utils import multiple_diagnostic


## MNIST: load performance for different nets
file_name = "./plots/mnist-benchmark-nets"
with open(file_name + ".json", "r") as file:
    loaded_file = json.load(file)

multiple_diagnostic(loaded_file, fig_title="Performance with different Nets")
plt.savefig(file_name + ".pdf")
print("Plotted in", file_name + ".pdf")


## MNIST: load performance for different optimizers
file_name = "./plots/mnist-benchmark-opts"
with open(file_name + ".json", "r") as file:
    loaded_file = json.load(file)

multiple_diagnostic(loaded_file, fig_title="Performance with different optimizers")
plt.savefig(file_name + ".pdf")
print("Plotted in", file_name + ".pdf")



## Load learning rate scheduler results
file_name = "./plots/cifar10-opt-diagnostic"
with open(file_name + ".json", "r") as file:
    loaded_file = json.load(file)

multiple_diagnostic(loaded_file, fig_title="Performance with learning rate schedulers")
plt.savefig(file_name + ".pdf")
print("Plotted in", file_name + ".pdf")


## Load transforms results
file_name = "./plots/cifar10-transforms-diagnostic"
with open(file_name + ".json", "r") as file:
    loaded_file = json.load(file)

multiple_diagnostic(loaded_file, fig_title="Performance with image augmetations")
plt.savefig(file_name + ".pdf")
print("Plotted in", file_name + ".pdf")


## Load full resnet with different training settings results
file_name = "./plots/cifar10-resnet-aug"
with open(file_name + ".json", "r") as file:
    loaded_file = json.load(file)

multiple_diagnostic(loaded_file, fig_title="Image augmentation performance with ResNet18")
plt.savefig(file_name + ".pdf")
print("Plotted in", file_name + ".pdf")


## 
