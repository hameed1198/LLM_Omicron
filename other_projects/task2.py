from collections import Counter

def top_words(text, n=5):
    words = text.lower().split()
    return Counter(words).most_common(n)

print(top_words("I love AI and I love Python AI", 3))

# Word Count with Top N words with out using imports
def top_words_no_import(text, n=5):
    words = text.lower().split()
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    # Sort by frequency and return top N
    return sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:n]

print("with out imports:", top_words_no_import("I love AI and I love Python AI", 3))

#Decision Tree Classifier program
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

#decession tree classification with easy example

