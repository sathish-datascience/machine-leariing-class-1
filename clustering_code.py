from sklearn.cluster import KMeans

# Customer spending
X = [[500], [700], [800], [5000], [5500], [6000],[9000],[9500],[9800]]

model = KMeans(n_clusters=3, random_state=0)

model.fit(X)

labels = model.labels_

print(labels)

import matplotlib.pyplot as plt

plt.scatter(X,labels)
plt.xlabel('spending amount')
plt.ylabel('cluster')
plt.show()

