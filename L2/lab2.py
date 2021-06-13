from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix, classification_report
import sys

def main(classifier, qtd, flag):
    if classifier == 'knn':
        neigh = KNeighborsClassifier(n_neighbors = 5)
    if classifier == 'bayes':
        neigh = GaussianNB()
    if classifier == 'lda':
        neigh = LinearDiscriminantAnalysis()
    if classifier == 'lr':
        neigh = LogisticRegression(random_state=0,solver='lbfgs', max_iter=1000)
    if classifier == 'perceptron':
        neigh = Perceptron()

    X_train, y_train = load_svmlight_file("dados/train.txt")
    X_test, y_test = load_svmlight_file("dados/test.txt")

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # cut
    X_train = X_train[:int(qtd)]
    y_train = y_train[:int(qtd)]
    X_test = X_test
    y_test = y_test

    neigh.fit(X_train,y_train)

    # predicao do classificador
    print ('Predicting...')
    y_pred = neigh.predict(X_test)

    # mostra o resultado do classificador na base de teste
    print ('Accuracy: ',  neigh.score(X_test, y_test))

    # cria a matriz de confusao
    if flag == "-m":
        cm = confusion_matrix(y_test, y_pred)
        print (cm)
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("Use: lab2.py <classifier> <qtd> [optional]<flag>")
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2], "")