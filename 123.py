import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

labels=['class 4', 'class 5', 'class 6']
data = pd.DataFrame({'girl': [352, 350, 339], 'boy': [333, 345, 343]},
                    index=labels)
print(data['girl'])
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(5,4))
rects1 = ax.bar(x - width/2, data['girl'], width, label='girl')
rects2 = ax.bar(x + width/2, data['boy'], width, label='boy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set(ylim=(300, 370))
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
