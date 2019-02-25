import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    #TODO: Implement PCA by extracting eigenvector#
    #You may need to sort the eigenvalues to get  #
    #             the top K of them.              #
    ###############################################
    
    #get N,D
    N=X.shape[0]
    D=X.shape[1]
    #get mean
    #X_mean=np.mean(X,axis=1,keepdims=True)
    #X_normal=X-X_mean
    
    #compute cov matrix
    cov=(X.T.dot(X))/(N-1)
    #compute eigenvalue, eigenvector
    eigenvalue,eigenvector=np.linalg.eig(cov)
 
    #sort eigenvalue & eigenvector
    eigen_index=eigenvalue.argsort()[::-1]
    eigenvalue = eigenvalue[eigen_index]
    eigenvector = eigenvector[:,eigen_index]
    #keep first K eigenvector
    eigenvector_K = eigenvector[:,0:K]
    P = eigenvector_K.T
    #keep first K eigenvalue
    T=eigenvalue[0:K]
    

    ###############################################
    #              End of your code               #
    ###############################################
    
    return (P, T)
