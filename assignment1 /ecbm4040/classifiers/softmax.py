import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
        score = X[i].dot(W)
        #normalize
        score_norm=score-np.max(score)
        loss_row = -np.log(np.exp(score_norm[y[i]])/np.sum(np.exp(score_norm)))
        loss+=loss_row#add up loss
        for j in range (num_classes):
            out=np.exp(score_norm[j])/sum(np.exp(score_norm))
            if j == y[i]:#when yi, dW=-1
                dW[:,j]+=(-1+out)*X[i] 
            else: 
                dW[:,j]+=out*X[i]
    

    loss=loss/num_train
    #regularization
    loss+=reg*np.sum(W*W)
    dW = dW/num_train + reg*2*W 
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]

    score=X.dot(W)
    
    #logC=-max(fj)    
    max_score=np.max(score,axis=1,keepdims=True).reshape(-1,1)#get max score for 0~N
    score_norm = score - np.max(score, axis = 1).reshape(-1, 1)#normalize
    out = np.exp(score_norm) / np.sum(np.exp(score_norm), axis = 1).reshape(-1, 1)
    loss = - np.sum(np.log(out[range(num_train), list(y)]))
    
    s = out.copy()
    s[range(num_train), list(y)] += -1
    dW = (X.T).dot(s)
  
    loss= loss/num_train
    dW = dW/num_train
    #regularization
    loss +=reg *np.sum(W*W)
    dW+=reg*2*W 

    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
