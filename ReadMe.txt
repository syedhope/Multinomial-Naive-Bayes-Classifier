Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features.

Preprocessed text has been stored in .review files so that each line contains a review
document; each token (e.g., year:2) represents a word and its frequency in the document. The last token
(e.g., #label#:negative) in each line indicates the polarity (label) of the document.

The algorithm classifies the data into positive and negative reviews and then finds the accuracy of the system for the dataset.