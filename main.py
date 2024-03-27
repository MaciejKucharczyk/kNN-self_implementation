import pandas
from sklearn import datasets
import numpy as np
import os
import matplotlib.pyplot as plt

iris_data = datasets.load_iris()
X = iris_data['data']
y = iris_data['target']

"""
Listy zawierajace dane potrzebne do testow:
:k_list - lista wartosci k, dla ktorych model zostal testowany
:accuracy_list - lista z ocena poziomu przypisania dla kazdego k
"""
k_list = list(range(1, 151))
accuracy_list = []


# zapis danych ze zbioru w tablicach numpy
X = np.array(X)
y = np.array(y)

# podzial danych na dane treningowe i dane testowe
def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# wyznaczenie odleglosci 
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn(X_train, y_train, X_test, k):
    y_pred = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x) for x in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(most_common)
    return np.array(y_pred)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# uruchomienie modelu kla kazdego k z listy powyzej
for k in k_list:
    print("k = ", k)
    y_pred = knn(X_train, y_train, X_test, k)
    accuracy_list.append(accuracy(y_test, y_pred))
    os.system('clear')

# wyznaczenie optymalnej wartosci k 
best_k = np.argmax(accuracy_list) + 1
print(f"Najlepsze k: {best_k} z dokładnością: {max(accuracy_list)}")

# wyznaczenie listy optymalnych k, jezeli jest ich kilka
best_ks = [k for k, accuracy in enumerate(accuracy_list, start=1) if accuracy == 1.0]
print(f"Wartości k z dokładnością 1.0: {best_ks}")


# wygenerowanie macierzy pomylek
def confusion_matrix(y_true, y_pred, classes):
    """
    Generuje macierz pomyłek.

    :param y_true: rzeczywiste etykiety
    :param y_pred: przewidywane etykiety
    :param classes: lista unikalnych klas
    :return: macierz pomyłek jako lista list
    """
    # Inicjalizacja macierzy pomyłek zerami
    matrix = [[0 for _ in classes] for _ in classes]
    
    # Mapowanie klasy do indeksu
    class_to_index = {cls: index for index, cls in enumerate(classes)}

    # Wypełnianie macierzy pomyłek
    for true, pred in zip(y_true, y_pred):
        true_index = class_to_index[true]
        pred_index = class_to_index[pred]
        matrix[true_index][pred_index] += 1

    return matrix

# wyswietlenie macierzy pomylek
def print_confusion_matrix(matrix, classes):
    """
    Wypisuje macierz pomyłek.

    :param matrix: macierz pomyłek
    :param classes: lista unikalnych klas
    """
    print("Macierz pomyłek:")
    print("     " + " ".join(f"{cls:5}" for cls in classes))
    print("      =================")
    for i, row in enumerate(matrix):
        print(f"{classes[i]:2}"," |", ' '.join(f"{cell:5}" for cell in row))

# Znajdowanie unikalnych klas
classes = np.unique(y)

# Generowanie macierzy pomyłek dla najlepszego k
best_k = np.argmax(accuracy_list) + 1  
y_pred = knn(X_train, y_train, X_test, best_k)
cm = confusion_matrix(y_test, y_pred, classes)

# Wypisywanie macierzy pomyłek
print_confusion_matrix(cm, classes)

# Generowanie wykresu zaleznosi k od accuracy

plt.plot(range(1, 151), accuracy_list)
plt.xlabel('Wartość k')
plt.ylabel('Dokładność')
plt.title('Zależność dokładności od k')
plt.show()


