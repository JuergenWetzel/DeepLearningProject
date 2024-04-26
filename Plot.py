import math

import matplotlib.pyplot as plt
import pandas as pd


def plot(x_values, y_values_list, x_label, y_label, legend_labels, filename=None):
    plt.figure(figsize=(10, 6))
    for i, y_values in enumerate(y_values_list):
        plt.plot(x_values, y_values, label=legend_labels[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    if filename is not None:
        plt.savefig(filename)


df = pd.read_csv('accuracy_bin_long.csv')
column_names = df.columns.tolist()
csv = [[] for _ in range(len(column_names))]
for index, row in df.iterrows():
    for i, column_name in enumerate(column_names):
        csv[i].append(row[column_name])

xlabel = csv[0]
acc = csv[1]
loss = csv[2]
epochs = csv[3]
acc_cat = csv[4]
acc_dog = csv[5]

xlog = []
for i in csv[0]:
    xlog.append(math.log10(i))

plot(xlog, [acc, acc_cat, acc_dog], "log10 weight", "accuracy", ["test total", "test cats", "test dogs"], "acclog.eps")
plot(xlog, [loss], "log10 weight", "loss", ["train loss"], "losslog.eps")

plot(xlabel, [acc, acc_cat, acc_dog], "weight", "accuracy", ["test total", "test cats", "test dogs"], "acc.eps")
plot(xlabel, [loss], "weight", "loss", ["train loss"], "loss.eps")

