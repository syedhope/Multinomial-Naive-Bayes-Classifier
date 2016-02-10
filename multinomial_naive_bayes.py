import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1
        
    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        
        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words,n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being 
            # active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 
            # corresponds to the fifth feature!
        
        ###########################
        # Calculating the Prior Probabilities for the classes
        prior[0],prior[1] =  (float(np.size(y)-np.count_nonzero(y))/np.size(y)),(float(np.count_nonzero(y))/np.size(y))
        # pos_count and neg_count gives the count for each word for a particular class
        pos_count,neg_count = np.zeros(n_words),np.zeros(n_words)
        # examining each word and fining the above mentioned values
        for c_w in range (x.shape[1]):        
            for c_d in range(x.shape[0]):
                if y[c_d] == 0: # if positive
                    pos_count[c_w] += x[c_d,c_w]
                else: # if negative
                    neg_count[c_w] += x[c_d,c_w]
        # Finding likelihood for each word with respective to a class
        for c_w in range (x.shape[1]):
            likelihood[c_w,0] = float(pos_count[c_w])/np.sum(pos_count)
            likelihood[c_w,1] = float(neg_count[c_w])/np.sum(neg_count)
        ###########################

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes): 
            # log probabilities
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
        
