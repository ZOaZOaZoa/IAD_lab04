# %%
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pingouin
import seaborn as sns

wine=load_wine()
n_classes = len(wine['target_names'])
data_shape = wine['data'].shape

w_pd = pd.DataFrame(np.c_[wine['data'], wine['target']], columns=wine['feature_names']+['target'])
w_pd = w_pd.rename(columns={'od280/od315_of_diluted_wines': 'od280/od315'})

X_train, X_test, y_train, y_test = train_test_split(w_pd.iloc[:,:-1], w_pd['target'], test_size=0.3, random_state=42)
print(w_pd)
print(f'Количество классов {n_classes}, форма данных {data_shape}')

# %%
print(wine['DESCR'])

# %%
plt.figure(figsize=(10, 12.8))
for i, column in enumerate(X_train.columns):
    plt.subplot(4,4,i+1)
    plt.boxplot(X_train[column])
    plt.title(column)
plt.show()

# %%
w_corr = X_train.corr()
w_pcorr = X_train.pcorr()
plt.figure(figsize=(10, 9))
sns.heatmap(w_corr.round(2), vmin=-1, vmax=1, annot=True)
plt.title('Парные корреляции')
plt.show()

plt.figure(figsize=(10, 9))
sns.heatmap(w_pcorr.round(2), vmin=-1, vmax=1, annot=True)
plt.title('Частные корреляции')
plt.show()

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X_train)

pca = PCA()
x_pca = pca.fit_transform(x_scaled, y_train)
data_delta = pca.explained_variance_ratio_
x_pca_pd = pd.DataFrame(x_pca, columns=[ f'f{i}' for i in range(len(X_train.columns))])

print(f'Информативность компонент: {(100*data_delta).round(2)}%')
print(f'Информативность: {100*sum(data_delta[:3]):.2f}%')

def scatter3d(data_pd, columns, classes, classes_names):
    ax = plt.axes(projection='3d')

    x = data_pd[columns[0]]
    y = data_pd[columns[1]]
    z = data_pd[columns[2]]

    scatter1 = ax.scatter(x, y, z, c=classes, marker='o', edgecolors=['000']*len(x))
    legend1 = ax.legend(*[scatter1.legend_elements()[0],classes_names], 
                        title="Legend", loc='upper left')
    ax.add_artist(legend1)
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[2])
    plt.legend()
    plt.show()

scatter3d(x_pca_pd, columns=['f1','f2','f3'], classes=y_train, classes_names=wine['target_names'])


