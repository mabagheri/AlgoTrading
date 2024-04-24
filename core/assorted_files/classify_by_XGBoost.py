def classify(X_train, y_train, X_test, y_test, configs):
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    X_train_original = X_train.values
    X_test = X_test.values

    n_trees = configs["classification"]["XGBoost: n_trees"]
    md = configs["classification"]["XGBoost: max_depth"]
    lr = configs["classification"]["XGBoost: learning_rate"]

    # split data into train and test sets
    seed = 7
    test_size = 0.01
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X_train_original, y_train, test_size=test_size, random_state=seed)

    # fit model no training data
    # eval_set = [(X_train, y_train), (X_test, y_test)]
    eval_set = [(X_train, y_train)]
    clf = XGBClassifier(max_depth=md, alpha=10, n_estimators=n_trees, learning_rate=lr, eval_metric="logloss")

    # clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="aucpr", eval_set=eval_set, verbose=False)
    clf.fit(X_train, y_train, eval_metric="aucpr", eval_set=eval_set, verbose=False)

    prob_y_train = clf.predict_proba(X_train_original)
    prob_y_test = clf.predict_proba(X_test)

    print("XGBoost Training done !")
    return clf, prob_y_train, prob_y_test
