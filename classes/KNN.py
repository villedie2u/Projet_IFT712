"""
class using the KNN model to predict targets
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class KNNeighbours:
    def __init__(self):
        self.predictions = []
        self.classification_report = 0
        self.accuracy_score = 0
    
    def knn(self, data_train, data_target):
        x_train, x_test, y_train, y_test = train_test_split(data_train, data_target, test_size=0.2)
        paramk = int(input("Please, pick an option. "
                           "0 = Manual choice of K. 1 = Automatic determination of the best K parameter. : "))
        if paramk == 0:
            k_value = int(input("Please, choose a parameter K : "))

        else:
            min_k = 51
            while min_k>50:
                min_k = int(input("Please, choose a minimal number for K (we advise you to choose 2 or 3) : "))
                if min_k>50:
                    print("Parameter too high. Please, choose a number inferior to 50.")

            # Determination of the best K parameter
            print("K parameter tuning. Please, wait...")
            k_list = []
            for i in range(min_k, 50):
                k_list.append(i)
                
            scores_list = []
            
            # 10-fold cross validation
            for k in k_list:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, data_train, data_target, cv=10, scoring='accuracy')
                scores_list.append(scores.mean())
                
            # turning this into a misclassification errors list
            Errors = []
            for i in scores_list:
                Errors.append(1-i)
            
            plt.figure()
            plt.figure(figsize=(15, 10))
            plt.title('Optimal number of neighbors', fontsize=20, fontweight='bold')
            plt.xlabel('K Parameter', fontsize=15)
            plt.ylabel('Misclassification Error', fontsize=15)
            plt.plot(k_list, Errors)
            
            plt.show()
            
            # finding best k
            k_value = k_list[Errors.index(min(Errors))]
            print("The optimal number of neighbors is %d." % k_value)
            
        # Training and evaluating the model's accuracy
        neigh = KNeighborsClassifier(n_neighbors=k_value)
        neigh.fit(x_train, y_train)

        self.predictions = neigh.predict(x_test)
        self.classification_report = classification_report(y_test, self.predictions)
        self.accuracy_score = accuracy_score(y_test, self.predictions)

        self.print_results()
            
    def print_results(self):
        print("predictions:", self.predictions, "\n")
        print("classification_report:", self.classification_report, "\n")
        print("accuracy_score:", self.accuracy_score, "\n")
