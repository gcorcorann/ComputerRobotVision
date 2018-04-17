import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

confusion = np.zeros((4,4))

data = [0.8317, 0.1683, 0, 0, 0.0194, 0.9419, 0.0194, 0.0194,
        0, 0.0455, 0.9091, 0.0455, 0, 0.1084, 0.0482, 0.8434]

i = 0
for row in range(4):
    for col in range(4):
        confusion[row][col] = data[i]
        i += 1

# plot confusion matrix
fig, ax = plt.subplots(figsize=(6,5))
cax = ax.matshow(confusion)
fig.colorbar(cax)
all_categories = ['low attention', 'medium attention', 'high attention',
        'very high attention']
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
fig.tight_layout()
fig.savefig('../outputs/confusion.png')
