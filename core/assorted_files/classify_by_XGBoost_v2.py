def classify(X_train, y_train, X_test, y_test, configs):
    # from xgboost import XGBClassifier
    import xgboost as xgb
    # from sklearn.model_selection import train_test_split

    X_train = X_train.values
    X_test = X_test.values

    n_trees = configs["classification"]["XGBoost: n_trees"]
    md = configs["classification"]["XGBoost: max_depth"]
    lr = configs["classification"]["XGBoost: learning_rate"]
    print(f"n_trees: {n_trees}, max_depth: {md}, learning_rate: {lr}")
    # split data into train and test sets
    # seed = 7
    # test_size = 0.01
    # X_train, X_valid, y_train, y_valid = \
    #     train_test_split(X_train_original, y_train, test_size=test_size, random_state=seed)

    # Train a model
    n_rounds = n_trees
    params = {
        'seed': 42,
        'max_depth': md,
        'learning_rate': lr,  # allias: eta
        "tree_method": "hist"  # hist or gpu_hist'

    }

    # fit model on training data
    d_train = xgb.DMatrix(X_train, label=y_train, nthread=2)
    d_test = xgb.DMatrix(X_test, label=y_test, nthread=2)

    model = xgb.train(params, d_train, n_rounds)

    # eval_set = [(X_train, y_train)]
    # clf = XGBClassifier(max_depth=md, alpha=10, n_estimators=n_trees, learning_rate=lr, eval_metric="logloss")

    # clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="aucpr", eval_set=eval_set, verbose=False)
    # clf.fit(X_train, y_train, eval_metric="aucpr", eval_set=eval_set, verbose=False)

    prob_y_train = model.predict(d_train)
    prob_y_test = model.predict(d_test)

    print("XGBoost Training done !")
    return model, prob_y_train, prob_y_test
