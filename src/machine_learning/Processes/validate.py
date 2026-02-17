import pandas as pd
from sklearn.metrics import accuracy_score



def evaluateClassifierProduction(clf, X_train, y_train, X_val, y_val,
                            preprocessing_fn=None, kwargs={}):
    new_X_train = pd.DataFrame(X_train)
    new_y_train = pd.Series(y_train)

    new_X_val   = pd.DataFrame(X_val)
    new_y_val   = pd.Series(y_val)

    if preprocessing_fn != None:
        new_X_train, new_y_train, new_X_val, new_y_val = preprocessing_fn(new_X_train, 
                                            new_y_train, new_X_val, new_y_val, **kwargs)
    clf.fit(new_X_train, new_y_train)

    y_preds = clf.predict(new_X_val)

    print('Validation Accuracy is', accuracy_score(new_y_val, y_preds))

    return