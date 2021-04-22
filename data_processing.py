import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def label_rectify(data):
	data[1:-10] += 5
	data[-10:] += 1
	return data


def data_preparation(file):
	epoch = []
	top1 = []
	top5 = []
	loss = []
	delete_title = list(range(4))
	for i in range(4):
		delete_title[i] = file.readline()
	for i in range(39):
		line = file.readline()
		words = line.split()
		epoch.append(int(words[7]))
		top1.append(float(words[11]))
		top5.append(float(words[15]))
		loss.append(float(words[19]))
	data = [epoch, top1, top5, loss]
	data = np.array(data)
	label_rectify(data[0])
	data = pd.DataFrame(data=data, index=['epoch', 'top1', 'top5', 'loss'])
	return data.T


root_dir = "."
Folder = "diagram"
images = os.path.join(root_dir, Folder)
os.makedirs(images, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
	path = os.path.join(images, fig_id + "." + fig_extension)
	print("Saving figure", fig_id)
	if tight_layout:
		plt.tight_layout()
	plt.savefig(path, format=fig_extension, dpi=resolution)
