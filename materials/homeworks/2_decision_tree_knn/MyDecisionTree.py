import numpy as np
from sklearn.base import BaseEstimator
from collections import Counter
import heapq


median_dif = 0.0


class DecisionTree(BaseEstimator):
    class NotFittedError(Exception):
        pass

    class UnknownCriterionError(Exception):
        pass

    class TreeNode:
        def __init__(self, tree, measure, predict, X, y, depth = 0, class_count = 0):
            self.__tree = tree
            self.__depth = depth
            self.__predicted = None
            self.__feature = None
            self.__threshold = None
            self.__left_child = None
            self.__right_child = None
            self.__class_count = class_count

            if depth >= tree.max_depth or X.shape[0] < tree.min_samples_split:
                self.__predicted = predict(y)
                return

            best_quality = None
            for feature in range(X.shape[1]):
                rearrange = np.argsort(X[:, feature])
                sorted_features = X[rearrange, feature]
                sorted_y = y[rearrange]
                left_criteria = measure(sorted_y)
                right_criteria = np.flip(measure(np.flip(sorted_y)))
                for separator in range(1, sorted_features.shape[0]):
                    if sorted_features[separator] == sorted_features[separator - 1]:
                        continue
                    quality = left_criteria[-1]
                    quality -= separator / sorted_features.shape[0] * left_criteria[separator - 1]
                    quality -= (1 - separator / sorted_features.shape[0]) * right_criteria[separator]
                    if self.__feature is None or quality > best_quality:
                        best_quality = quality
                        self.__feature = feature
                        self.__threshold = (sorted_features[separator] + sorted_features[separator - 1]) / 2

            if self.__feature is not None:
                go_left = (X[:, self.__feature] < self.__threshold)
                self.__left_child = tree.TreeNode(tree, measure, predict, X[go_left], y[go_left], depth + 1,
                                                  class_count)
                self.__right_child = tree.TreeNode(tree, measure, predict, X[~go_left], y[~go_left], depth + 1,
                                                   class_count)
            else:
                self.__predicted = predict(y)

        def predict(self, X, probability = False):
            if not probability:
                predicted = np.zeros(shape = (X.shape[0]))
            else:
                predicted = np.zeros(shape = (X.shape[0], self.__class_count))
            if X.shape[0] != 0:
                if self.__feature is None:
                    if not probability:
                        predicted[:] = self.__predicted[0]
                    else:
                        for tag, p in self.__predicted[1].items():
                            predicted[:, tag] = p
                else:
                    go_left = (X[:, self.__feature] < self.__threshold)
                    predicted[go_left] = self.__left_child.predict(X[go_left], probability = probability)
                    predicted[~go_left] = self.__right_child.predict(X[~go_left], probability = probability)
            return predicted

        def __str__(self):
            res = ''
            if self.__left_child is not None:
                res += self.__left_child.__str__()
            res += '{0}[node ({1}): feature = {2}, threshold = {3}, prediction = {4}]\n'.format(
                '  ' * self.__depth, self.__depth, self.__feature, self.__threshold, self.__predicted)
            if self.__right_child is not None:
                res += self.__right_child.__str__()
            return res

    def __init__(self, max_depth = np.inf, min_samples_split = 2, criterion = 'gini', debug = False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.__debug = debug
        self.__root = None
        self.__class_count = None
        self.median_dif_ = 0

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError('Numbers of objects in X and y are not equal.')
        if len(X.shape) != 2:
            raise ValueError('X must be 2-dimensional.')
        if len(y.shape) != 1:
            raise ValueError('y must be 1-dimensional.')
        measure, predict = self.__get_methods(self.criterion)
        self.__class_count = int(np.max(y) + 1)
        self.__root = self.TreeNode(self, measure, predict, X, y, class_count = self.__class_count)

    def predict(self, X, probability = False):
        if len(X.shape) != 2:
            raise ValueError('X must be 2-dimensional.')
        if self.__root is None:
            raise self.NotFittedError
        return self.__root.predict(X, probability = probability)

    def predict_proba(self, X):
        if len(X.shape) != 2:
            raise ValueError('X must be 2-dimensional.')
        return self.predict(X, probability = True)

    def __str__(self):
        res = 'DecisionTree:\n'
        res += self.__root.__str__()
        return res

    def __get_methods(self, criterion):

        def classic_entropy(y):
            partials = np.zeros(shape = y.shape[0])
            counter = {}
            for i, y_i in enumerate(y):
                counter[y_i] = counter.get(y_i, 0) + 1
                p = np.array(list(counter.values())) / (i + 1)
                partials[i] = -np.sum(p * np.log2(p))
            return partials

        def entropy(y):
            partials = np.zeros(shape = y.shape[0])
            logs = self.__logs
            while len(logs) <= y.shape[0]:
                logs.append(np.log2(len(logs)))
            counter = {}
            cur_log_sum = 0.0
            for i, y_i in enumerate(y):
                count_y_i = counter.get(y_i, 0)
                if count_y_i > 0:
                    cur_log_sum -= count_y_i * logs[count_y_i]
                count_y_i += 1
                cur_log_sum += count_y_i * logs[count_y_i]
                counter[y_i] = count_y_i
                partials[i] = (logs[i + 1] * (i + 1) - cur_log_sum) / (i + 1)
            return partials

        def classic_gini(y):
            partials = np.zeros(shape = y.shape[0])
            counter = {}
            for i, y_i in enumerate(y):
                counter[y_i] = counter.get(y_i, 0) + 1
                p = np.array(list(counter.values())) / (i + 1)
                partials[i] = 1 - np.sum(p * p)
            return partials

        def gini(y):
            partials = np.zeros(shape = y.shape[0])
            counter = {}
            cur_sum_squared = 0.0
            for i, y_i in enumerate(y):
                count_y_i = counter.get(y_i, 0)
                if count_y_i > 0:
                    cur_sum_squared -= count_y_i * count_y_i
                count_y_i += 1
                cur_sum_squared += count_y_i * count_y_i
                counter[y_i] = count_y_i
                partials[i] = 1 - cur_sum_squared / ((i + 1) * (i + 1))
            return partials

        def classic_variance(y):
            partials = np.zeros(shape = y.shape[0])
            for i in range(y.shape[0]):
                prefix = y[:(i + 1)]
                partials[i] = np.mean((prefix - np.mean(prefix)) ** 2)
            return partials

        def variance(y):
            partials = np.zeros(shape = y.shape[0])
            cur_sum = 0.0
            cur_square_sum = 0.0
            for i, y_i in enumerate(y):
                cur_sum += y_i
                cur_square_sum += y_i ** 2
                partials[i] = cur_square_sum / (i + 1) - (cur_sum / (i + 1)) ** 2
            return partials

        def classic_mad_median(y):
            global median_dif
            partials = np.zeros(shape = y.shape[0])
            for i in range(y.shape[0]):
                prefix = y[:(i + 1)]
                partials[i] = np.mean(np.abs(prefix - np.median(prefix)))
            dif = np.max(np.abs(partials - mad_median(y)))
            median_dif = max(median_dif, dif)
            return partials

        def mad_median(y):
            partials = np.zeros(shape = y.shape[0])
            left_sum = 0.0
            left_heap = []
            right_sum = 0.0
            right_heap = []
            for i, num in enumerate(y):
                if len(left_heap) == 0 or num < -left_heap[0]:
                    left_sum += num
                    heapq.heappush(left_heap, -num)
                else:
                    right_sum += num
                    heapq.heappush(right_heap, num)
                if len(left_heap) < len(right_heap):
                    num = heapq.heappop(right_heap)
                    right_sum -= num
                    left_sum += num
                    heapq.heappush(left_heap, -num)
                elif len(left_heap) > len(right_heap) + 1:
                    num = -heapq.heappop(left_heap)
                    left_sum -= num
                    right_sum += num
                    heapq.heappush(right_heap, num)
                if len(left_heap) == len(right_heap):
                    median = (-left_heap[0] + right_heap[0]) / 2
                else:
                    median = -left_heap[0]
                partials[i] = (right_sum - left_sum + (len(left_heap) - len(right_heap)) * median) / (i + 1)
            return partials

        def classify(y):
            probabilities = Counter(y)
            result = None
            biggest_count = None
            for tag, count in probabilities.items():
                if result is None or count > biggest_count:
                    result = tag
                    biggest_count = count
                probabilities[tag] = count / y.shape[0]
            return result, probabilities

        def interpolate(y):
            return y.mean(), None

        if criterion == 'entropy':
            self.__logs = [0]

        criteria = {
            'entropy': (entropy, classify),
            'gini': (gini, classify),
            'variance': (variance, interpolate),
            'mad_median': (mad_median, interpolate),
            'classic_entropy': (classic_entropy, classify),
            'classic_gini': (classic_gini, classify),
            'classic_variance': (classic_variance, interpolate),
            'classic_mad_median': (classic_mad_median, interpolate)
        }
        if criterion not in criteria:
            raise self.UnknownCriterionError(criterion)
        measure, predict = criteria[criterion]
        return measure, predict

    @staticmethod
    def median_dif():
        global median_dif
        return median_dif
