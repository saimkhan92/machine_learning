import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()

class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """
        
        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, returns the majority label.  If
        there's a tie, returns the median of the majority labels (as implemented
        in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"

        l1=[]
        dic={}
        count=1
        for i in item_indices:
            l1.append(self._y[i])
        for i in l1:
            if i not in dic.keys():
                dic[i]=count
            else:
                dic[i]=count+1
        dic_max=max(dic.values())
        max_list=[k for k,v in dic.items() if v==dic_max]
        if len(max_list)==1:
            return max_list[0]
        else:
            return numpy.median(numpy.array(max_list))

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """
        example_new=numpy.array(example).reshape((1,-1))
        dist, ind = self._kdtree.query(example_new, k=self._k)
        majority_label=self.majority(ind[0].reshape(len(ind[0])).tolist())
        return majority_label


    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier. Return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        dic={0:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},1:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},2:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},3:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},4:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},5:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},6:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},7:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},8:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},9:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}}

        for i in range(0,len(test_x)):
            real_label=test_y[i]
            calculated_label=int(self.classify(test_x[i]))
            dic[real_label][calculated_label]=dic[real_label][calculated_label]+1
        return dic

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, computes the accuracy of the underlying classifier.
        """
        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")
    #print(knn.majority([39,49,59]))
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in xrange(10)))
    print("".join(["-"] * 90))
    for ii in xrange(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(10)))
    print("Accuracy: %f" % knn.accuracy(confusion))
