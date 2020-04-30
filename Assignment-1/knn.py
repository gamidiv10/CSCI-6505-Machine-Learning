import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

K = 3
coordinates = []
neighbors = []

a = [3, 4, 5]
a= np.asarray(a)


np.delete(a)

print(a)


def fit(training_data, test_data_point, K, count, labels):
    distance_list = []
    for index in range(len(training_data)):
        dist = distance(test_data_point, training_data[index])
        distance_list.append((training_data[index], dist, labels[index], test_data_point))
    distance_list.sort(key=lambda x: x[1])
    neighbors = distance_list[:K]
    return neighbors

def distance(instance1, instance2):
    instance1 = np.array(instance1) 
    instance2 = np.array(instance2)
    return np.linalg.norm(instance2 - instance1, 2)

def create_dataset(space):
    for x in np.arange(-1.5, 1.5, 0.1):
        for y in np.arange(-1.5, 1.5, 0.1):
            coordinates.append((x, y))
    return coordinates


# for x in np.arange(-1.5, 1.5, 0.1):
#   for y in np.arange(-1.5, 1.5, 0.1):
#     coordinates.append((x, y))


class1 = []
class2 = []
training_data = []
labels = []
training_coordinates = create_dataset(0.1)
# test_data = create_dataset(0.1)
test_data = []

for x in np.arange(-1.5, 1.5, 0.1):
  for y in np.arange(-1.5, 1.5, 0.1):
    test_data.append((x, y))

def scatter(normValue):

    for x in training_coordinates:
        if(np.linalg.norm(x, normValue) <= 1):
            class1.append(x)
            label = "red"
            training_data.append(x)
            labels.append(label)

        else:
            class2.append(x)
            label = "blue"
            training_data.append(x)
            labels.append(label)

    # for x in class1:
    #     plt.scatter(x[0], x[1], c = 'b')

    # for y in class2:
    #     plt.scatter(y[0], y[1], c = 'r')
    # plt.show()
    

# scatter(2)



def neighbor_list():
    count = 0
    count1 = 0
    nei_list = []
    print(len(test_data))
    for index in range(len(test_data)):
        count1 += 1
        neighbors = fit(training_data, test_data[index], K, count, labels)
        count = 0
        for neighbor in neighbors:
            if neighbor[2] == 'blue':
                count += 1
            
        if count >= 2:
            color = 1
        else:
            color = 0
        nei_list.append(color)
    return np.asarray(nei_list)

        
  
# fig = plt.figure(figsize=(7.5,5))
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# ax = fig.add_axes([left, bottom, width, height])
# x_vals = np.arange(-1.5, 1.5, 0.1)
# y_vals = np.arange(-1.5, 1.5, 0.1)
# X, Y = np.meshgrid(x_vals, y_vals)
# Z = neighbor_list()
# cp = plt.contourf(X, Y, Z.reshape(X.shape[0], X.shape[1]))
# plt.colorbar(cp)

# ax.set_title('Contour Plot')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.show()


# print(len(training_data))
