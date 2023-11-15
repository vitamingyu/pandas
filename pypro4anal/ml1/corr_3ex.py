import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import numpy as np

url = 'https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/Advertising.csv'
df = pd.read_csv(urllib.request.urlopen(url), delimiter=',', na_values=' ')
data= pd.DataFrame(df, columns=('tv', 'radio', 'newspaper'))

# hitmap
import seaborn as sns
sns.heatmap(data.corr(method='spearman'))
plt.show()

# heatmap에 텍스트 표시 추가사항 적용해 보기
corr = data.corr(method='spearman')
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool_)  # 상관계수값 표시
mask[np.triu_indices_from(mask)] = True
# Draw the heatmap with the mask and correct aspect ratio
vmax = np.abs(corr.values[~mask]).max()
fig, ax = plt.subplots()     # Set up the matplotlib figure

sns.heatmap(corr, mask=mask, vmin=-vmax, vmax=vmax, square=True, linecolor="lightgray", linewidths=1, ax=ax)

for i in range(len(corr)):
    ax.text(i + 0.5, len(corr) - (i + 0.5), corr.columns[i], ha="center", va="center", rotation=45)
    for j in range(i + 1, len(corr)):
        s = "{:.3f}".format(corr.values[i, j])
        ax.text(j + 0.5, len(corr) - (i + 0.5), s, ha="center", va="center")
ax.axis("off")
plt.show()
