import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  

veri = pd.read_csv("veri-seti.txt", sep="\t", header=None)

# SORU 5 #

# Veri setinizi rastgele olarak %70 eğitim %30 test olacak şekilde ayırınız. 
X = veri[[0,1,2,3,4,5,6,7]]
y = veri[[8]]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=57)


# Eğitim veri seti için Naive bayes sınıflandırıcısını uygulayınız. 
from sklearn.naive_bayes import GaussianNB

# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, y_train.values.ravel())

# Elde ettiğiniz sonucları raporlayınız.  
# Test verisi için performans metriklerini hesaplayınız.

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)

labels = [0,1,2]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot();
