#1)Let us first import all the necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2)Loading the dataset
# To import the dataset we will use Pandas library.It is the best Python library to play with the dataset and has a lot of functionalities.
df = pd.read_csv('../datafile/HR_comma_sep.csv')
print(df.head())

print('\n\n')
print(df.columns)

print('\n\n')
print(df.columns.tolist())

print('\n')
print('shape of the dataset=',df.shape)

print('\n\n')
print(df.corr().to_string())

print('\n\n')

#Visualising correlation using Seaborn library

# correlation = df.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(data=correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
# plt.title('correlation between certain feature')
# plt.show()

# Doing some visualisation before moving onto PCA
print(df['sales'].unique())

print('\n\n')
print(df.groupby('sales').sum())

groupby_sales = df.groupby('sales').mean()
print(groupby_sales.to_string())

print('\n\n')
IT=groupby_sales['satisfaction_level'].IT
# IT=groupby_sales['satisfaction_level']
print(IT)
RandD=groupby_sales['satisfaction_level'].RandD
print(RandD)
accounting=groupby_sales['satisfaction_level'].accounting

hr=groupby_sales['satisfaction_level'].hr
management=groupby_sales['satisfaction_level'].management
marketing=groupby_sales['satisfaction_level'].marketing
product_mng=groupby_sales['satisfaction_level'].product_mng
sales=groupby_sales['satisfaction_level'].sales
support=groupby_sales['satisfaction_level'].support
technical=groupby_sales['satisfaction_level'].technical
print(technical)
print('\n\n')
import pandas as pd



# dict = {'satisfaction_level':[sales, accounting, hr, technical, support, management,
#        IT, product_mng, marketing, RandD],
#        'department':['sales', 'accounting', 'hr', 'technical', 'support', 'management','IT', 'product_mng', 'marketing', 'RandD']}
# df=pd.DataFrame(dict, columns=['department','satisfaction_level'])
# print(df)

# sns.barplot(data=df,x='satisfaction_level',y='department')

# plt.show()
print('\n\n')
#Principal Component Analysis
print(df.head().to_string())
print('\n')
df_drop = df.drop(labels=['sales' , 'salary'],axis=1)
print(df_drop.head().to_string())

cols = df_drop.columns.tolist()
print(cols)

#Here we are converting columns of the dataframe to list so it would be easier for us to reshuffle the columns.We are going to use cols.insert method
print(cols.insert(0,cols.pop(cols.index('left'))))
print(cols)

print('\n\n')
df_drop = df_drop.reindex(columns=cols)
print(df_drop.head().to_string())

# By using df_drop.reindex(columns= cols) we are converting list to columns again

# Now we are separating features of our dataframe from the labels.
print('\n')
x = df_drop.iloc[:,1:8].values
print(x)

y = df_drop.iloc[:,0].values
print(y)

print(np.shape(x))
print(np.shape(y))

print('\n\n')
#4) Data Standardisation¶
from sklearn.preprocessing import StandardScaler
x_std = StandardScaler().fit_transform(x)
print(x_std)

# 5) Computing Eigenvectors and Eigenvalues:
# Before computing Eigen vectors and values we need to calculate covariance matrix.

print('\n\n')
# Covariance matrix
mean_vec = np.mean(x_std,axis=0)
cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec))/x_std.shape[0]-1
print('\n')
print('numpy covariance matrix:',np.cov(x_std.T))
sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different features')
sns.heatmap(cov_mat, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('correlation between different features')
plt.show()

#Eigen decomposition of the covariance matrix¶
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

print('\n\n')
#6) Selecting Principal Components¶
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

print('\n\n')
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(7), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Projection Matrix
matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1),
                      eig_pairs[1][1].reshape(7,1)
                    ))
print('Matrix W:\n', matrix_w)
Y = x_std.dot(matrix_w)
print(Y)

##PCA in scikit-learn
from sklearn.decomposition import pca
pca = PCA().fit(x_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

from sklearn.decomposition import PCA
sklearn_pca = PCA(n_components=6)
Y_sklearn = sklearn_pca.fit_transform(x_std)
print(Y_sklearn)

print(Y_sklearn.shape)
