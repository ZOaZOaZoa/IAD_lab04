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

# %% [markdown]
# ## Разведочный анализ

# %%
plt.figure(figsize=(10, 12.8))
for i, column in enumerate(X_train.columns):
    plt.subplot(4,4,i+1)
    plt.boxplot(X_train[column])
    plt.title(column)
plt.show()

# %% [markdown]
# Многие переменные имеют выбросы, но, их, судя по диаграмме не очень много. Часть диаграмм симметричны и имеют не очень длинные усы. Некоторые же несимметричны или имеют длинные усы.

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

# %% [markdown]
# Некоторые характеристики имеют сильные💪 парные корреляции. Это значит, что можно попробовать предсказывать эти значения через линейную регрессию. Однако, большая часть параметров заметной линейной связи не показывают.

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

pca = PCA()
x_pca = pca.fit_transform(x_scaled, y_train)
x_test_pca = pca.transform(x_test_scaled)
data_delta = pca.explained_variance_ratio_
x_pca_pd = pd.DataFrame(x_pca, columns=[ f'f{i}' for i in range(len(X_train.columns))])
x_test_pca_pd = pd.DataFrame(x_test_pca, columns=[ f'f{i}' for i in range(len(X_test.columns))])

print(f'Информативность компонент: {(data_delta).round(2)}')
print(f'Информативность: {100*sum(data_delta[:3]):.2f}%')

def scatter3d(data_pd, columns, classes=None, classes_names=None, title=''):
    ax = plt.axes(projection='3d')

    x = data_pd[columns[0]]
    y = data_pd[columns[1]]
    z = data_pd[columns[2]]

    if classes_names is not None:
        scatter1 = ax.scatter(x, y, z, c=classes, marker='o', edgecolors=['000']*len(x))
        legend1 = ax.legend(*[scatter1.legend_elements()[0],classes_names], 
                            title="Legend", loc='upper left')
        ax.add_artist(legend1)
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[2])
    plt.title(title)

scatter3d(x_pca_pd, columns=['f1','f2','f3'], classes=y_train, classes_names=wine['target_names'], title=f'МГК с информативностью {sum(data_delta[:3]):.2f}')
plt.show()

# %% [markdown]
# Полученное представление имеет не слишком большую информативность (окло 67%), но по ней всё равно можно сделать некоторые выводы, учитывая то, что она показывает не всю информацию о выборке. Так, здесь видно, что первый класс довольно хорошо должен отделяться от двух других, по скольку он находится в отдельном кластере. Также, по картинке можно предположить, что первый класс является линейно отделимым. Для задачи кластеризации видно два кластера.

# %%
from sklearn.manifold import TSNE

t_sne = TSNE()
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(x_scaled)
tsne_data =pd.DataFrame(tsne_data,columns=['x1','x2'])
sns.scatterplot(data=tsne_data, x='x1', y='x2', c=y_train)
plt.title('Метод t-sne')
plt.legend()
plt.show()

# %% [markdown]
# В данном методе видно, что 3 класса довольно хорошо визуально разделимы. Значит, их классификация должна быть возможна и, при этом, с довольно большой хорошей разделимостью. Также на данном представлении видно 3 кластера, которые явно могут быть выделены.

# %%
from sklearn import linear_model

features = X_train.shape[1] - 1
n = X_train.shape[0]

y_columns = X_train.columns
l_regs = dict()
l_columns = dict()
data = dict()

#train
for col in y_columns:
    l_columns[col] = list(X_train.columns)
    l_columns[col].remove(col)
    X = X_train[l_columns[col]].values.reshape(-1, features)
    y = X_train[col].values.reshape(-1, 1)

    data[col] = (X, y)

    l_reg = linear_model.LinearRegression()
    l_reg.fit(X, y)
    l_regs[col] = l_reg


R_squared = dict()
std_coeffs = dict()
y_pred = dict()
RMSE = dict()

def get_std_coeffs(coeffs, std_pd, col, l_columns):
    scale = np.array([ std_pd[i] / std_pd[col] for i in l_columns[col] ])
    return pd.Series(data=(coeffs*scale).flatten(), index=l_columns[col])

#test
train_std = X_train.std(ddof=1)
for col in y_columns:
    X = X_train[l_columns[col]].values.reshape(-1, features)
    y = X_train[col].values.reshape(-1, 1)
    test_X = X_test[l_columns[col]].values.reshape(-1, features)
    test_y = X_test[col].values.reshape(-1, 1)
    
    R_squared[col] = l_regs[col].score(X, y)
    std_coeffs[col] = get_std_coeffs(l_regs[col].coef_, train_std, col, l_columns)
    y_pred = l_regs[col].predict(test_X)
    err = test_y - y_pred
    RMSE[col] = np.sqrt(err.T.dot(err) / n).item()
    

print("Коэффициенты детерминации по обучающей выборке")
print(pd.Series(R_squared))
print('\nRMSE по тестовой выборке')
print(pd.Series(RMSE))

print('\nКоэффициенты стандартизированных уравнений построенных регрессий')
print(pd.concat(std_coeffs, axis=1).round(2))

# %% [markdown]
# По полученным метрикам видно, что для нескольких параметров получились неплохие линейные регрессии (od280/od315, total_phenols, flavonoids), имеющие детерминацию более 0.7. Эти значения можно предсказывать по остальным. Также, стоит обратить внимание, что для некоторых параметров, получивших малое значение детерминации получены хорошие значения RMSE. Например, к ним можно отнести параметр hue. В целом, такое наблюдается для параметров имеющих выбросы. Также, получены стандартизированные уравнения для каждой из регрессий. По ним видно, что многие параметры в этих регрессиях малозначимы, и их можно исключить из регрессии упростив модель без значительного ухудшения качества.

# %% [markdown]
# ## Классификация

# %%
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn import tree

scores1 = ['accuracy', 'recall_micro', 'recall_macro', 'precision_micro', 'precision_macro', 'f1_micro', 'f1_macro']
n_classes = len(set(y_train))

def models_predict(models, X, n_classes):
    proba = np.zeros((X.shape[0], n_classes))
    for model in models:
        proba += model.predict_proba(X)
    proba /= len(models)
    y_pred = proba.argmax(axis=1)
    return y_pred
    
cross_validation_res = cross_validate(tree.DecisionTreeClassifier(max_depth=4), X_train, y_train, cv = 10, scoring = scores1, return_estimator=True)
scores=pd.DataFrame(cross_validation_res).drop(['fit_time', 'score_time', 'estimator'], axis=1).mean(axis=0)
models = cross_validation_res['estimator']
y_pred = models_predict(models, X_test, n_classes)
print('Средние метрики для валидационных выборок')
print(scores, '\n')

print('Метрики по тестовой выборке')
print(classification_report(y_pred, y_test)) 

plt.scatter(x='f1', y='f2', data=x_pca_pd, c=y_train)
plt.title('Обучающая выборка')
plt.show()


plt.scatter(x='f1', y='f2', data=x_test_pca_pd, c=y_pred)
plt.title('Классификация по тестовой выборке.\nМодели по кросс-валидации.\nДерево решений.')
plt.show()


plt.scatter(x='f1', y='f2', data=x_test_pca_pd, c=y_test)
plt.title('Истинное распределение классов тестовой выборки')
plt.show()

# %%
from catboost import CatBoostClassifier
from catboost import metrics
custom_metric=[metrics.Accuracy(), metrics.F1(), metrics.Precision(), metrics.Recall()]

clf = CatBoostClassifier(
    random_seed=0,
    verbose=False,
    custom_metric = custom_metric
)
params = {'learning_rate': np.arange(0.03, 0.1, 0.01),
          'iterations': range(2, 10),
        'depth': range(2,6),
        'l2_leaf_reg': [1, 2]}

import os
import sys
a = sys.stdout
f = open(os.devnull, 'w')
grig_clf = clf.grid_search(params,X_train,y=y_train,cv=10,verbose=False,log_cout=f)
f.close()
print('Найденные лучшие параметры')

best_params = grig_clf.get('params')
print(pd.Series(best_params))

# %%
best_clf = CatBoostClassifier(
    random_seed=0,
    verbose=False,
    custom_metric = custom_metric,
    **best_params
)
best_clf.fit(X_train, y_train)
y_pred=best_clf.predict(X_test)

print('Метрики по тестовой выборке')
print(classification_report(y_pred, y_test)) 

plt.scatter(x='f1', y='f2', data=x_pca_pd, c=y_train)
plt.title('Обучающая выборка')
plt.show()


plt.scatter(x='f1', y='f2', data=x_test_pca_pd, c=y_pred)
plt.title('Классификация по тестовой выборке.\nЛучшие параметры по grid_search.\nГрадиентный бустинг')
plt.show()


plt.scatter(x='f1', y='f2', data=x_test_pca_pd, c=y_test)
plt.title('Истинное распределение классов тестовой выборки')
plt.show()

# %% [markdown]
# ## Кластеризация

# %%
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def plot_clusters_KMeans(data_x, n_clusters, metric='euclidean', as_subplot = False):
    '''Кластеризация по методу K-средних и построение результатов на 3д графике по МГК'''
    pca = PCA()
    x_pca = pca.fit_transform(data_x)
    x_pca_pd = pd.DataFrame(x_pca, columns=[ f'f{i}' for i in range(data_x.shape[1])])

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(x_pca_pd)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    
    if as_subplot:
        ax = plt.figure(figsize=(12.8,6)).add_subplot(1, 2, 1, projection='3d')
    else:
        ax = plt.axes(projection='3d')
    z = x_pca_pd['f1']
    x = x_pca_pd['f2']
    y = x_pca_pd['f3']
    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax.scatter(x, y, z, c=colors, marker='o', edgecolors=['000']*len(labels))
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    ax.set_title(f'Сформированные кластеры (Заданное кол-во кластеров: {n_clusters}).\nMetric: {metric}, KMeans.\nПредставление МГК. Информативность {sum(pca.explained_variance_ratio_[:3]):.3f}')
    return (labels, inertia)

def plot_silhouettes(silhouette_vals, labels):
    '''Построение графиков силуэтов'''
    n_clusters = len(set(labels))
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_vals[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
        )
        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10 # 10 for the 0 samples
    silhouette_avg = silhouette_score(data_x, labels)
    plt.title(f"График силуэтов.\nСреднее значение силуэта: {silhouette_avg:.3f}")
    plt.xlabel("Значения силуэтов")
    plt.ylabel("Номер кластера")
    plt.xlim(-1, 1)
    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([]) # Clear the yaxis labels / ticks

def plot_wcss(n_clusters_var, inertias):
    '''Построение графика суммарного квадратичного отклонения'''
    plt.plot(n_clusters_var, inertias, '-bo')
    plt.ylabel('wcss')
    plt.xlabel('Количество кластеров')
    plt.title('Метод локтя.')

scaler = StandardScaler()
data_x = scaler.fit_transform(w_pd.iloc[:,:-1])

inertias = []
n_clusters_var = range(2, 6)
for clusters in n_clusters_var:
    labels, inertia = plot_clusters_KMeans(data_x, clusters, as_subplot=True)
    inertias.append(inertia)
    plt.subplot(1, 2, 2)
    silhouette_vals = silhouette_samples(data_x, labels)
    plot_silhouettes(silhouette_vals, labels)
    plt.show()
    
plot_wcss(n_clusters_var, inertias)
plt.show()


