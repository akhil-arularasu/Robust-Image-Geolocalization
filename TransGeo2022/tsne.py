from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('result_lvl4C+NGaussian2/grd_global_descriptor.npy', 'rb') as f:
    transforms = np.load(f)

labels = np.zeros(transforms.shape[0])

model = TSNE(n_components = 2, random_state = 0)

tsne_data = model.fit_transform(transforms)

tsne_data = np.vstack((tsne_data.T, labels)).T
tsne_df = pd.DataFrame(data = tsne_data,columns =("Dim_1", "Dim_2", "label"))

sn.FacetGrid(tsne_df, hue ="label", height=10, aspect=1).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
 
plt.savefig('TSNEC+NModelGaussianTestDataGround.png')