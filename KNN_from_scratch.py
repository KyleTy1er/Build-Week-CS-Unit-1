

Class KNN:
    def __init__(self, k=3)
    self.k = k


    def euclidean_distance(first_row, second_row):
        distance = 0
        # this assumes the last column is the target variable:
        # from 0th column to -1 column:
        for index in range(len(first_row)-1):
            # calculate the ED
            distance += (first_row[index] - second_row[index]) ** 2
            return sqrt(distance)

    def locate_neighbors(train, test, k):
        distances = list()
        for train_row in train:
            dist = euclidean_distance(test, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(k):
            neighbors.append(distances[i][0])
        return neighbors

    # Make a classification prediction with neighbors
    def knn_classifier(train, test, k):
        neighbors = locate_neighbors(train, test, k)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction



