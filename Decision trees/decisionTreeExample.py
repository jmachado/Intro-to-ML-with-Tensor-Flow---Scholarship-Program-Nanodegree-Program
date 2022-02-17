# Import the classifier from sklearn
from sklearn.tree import DecisionTreeClassifier

# TODO: Define the classifier, and fit it to the data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Making predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)

#The training accuracy is 1.0
#The test accuracy is 0.815642458101

#--------------------------------------------------------------------------------
#Improve the model

# TODO: Train the model
model = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 6, min_samples_split = 5)
model.fit(X_train, y_train)

# TODO: Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# TODO: Calculate the accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)

#The training accuracy is 0.88202247191
#The test accuracy is 0.860335195531