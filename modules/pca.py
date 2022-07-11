import numpy as np


class PCA(object):
    def __init__(self, n_components=2):
        self.X_demeaned = None
        self.explained_variance_ = None  #sorted eigenvalues
        self.components_ = None  #sorted eigenvectors
        self.n_components = n_components

    def fit(self, X):
        # mean-center the data on each row (axis=1)
        self.X_demeaned = X - np.mean(X.T, axis=1)

        # get the covariance matrix of the mean-centered features
        covariance_matrix = np.cov(self.X_demeaned.T)

        # get the eigenvalues and eigenvectors of the covariance matrix
        # use np.eigh() since we assume that the matrix is symmetric,
        # making the eigendecomposition faster to calculate than np.eig()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # sort the eigenvalues and eigenvectors in descending order
        idx_sorted = np.argsort(eigenvalues)
        idx_sorted_desc = idx_sorted[::-1]
        self.explained_variance_ = eigenvalues[idx_sorted_desc]
        self.components_ = eigenvectors[:,idx_sorted_desc]

    def transform(self, X):
        eigenvecs_subset = self.components_[:,:self.n_components]

        # transform the data by multiplying the transpose of the eigenvectors
        # with the transpose of the mean-centered data
        X_reduced = eigenvecs_subset.T.dot(self.X_demeaned.T).T
        return X_reduced
