from sklearn.model_selection import GridSearchCV

#2. Select the parameters:
parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}

#3. Create a scorer:
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
scorer = make_scorer(f1_score)

#4. Create a GridSearch Object with the parameters, and the scorer. Use this object to fit the data.
# Create the object.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
# Fit the data
grid_fit = grid_obj.fit(X, y)


#5. Get the best estimator.
best_clf = grid_fit.best_estimator_
