def classify(X_train, y_train, X_test, y_test, configs):
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense
    # from keras.utils import np_utils

    nb_classes = len(np.unique(y_train))
    # batch_size = configs["classification"]["FCN: batch_size"] #  int(min(X_train.shape[0]/10, 16))
    # nb_epochs = configs["classification"]["FCN: n_epochs"]

    # y_train = np_utils.to_categorical(y_train, nb_classes)
    # y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.values
    X_test = X_test.values

    X_train = np.asarray(X_train).astype('float32')
    X_test = np.asarray(X_test).astype('float32')

    # X_train = X_train.reshape(X_train.shape + (1, ))
    # X_test = X_test.reshape(X_test.shape + (1, ))

    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=20, )  # batch_size=batch_size

    # evaluate the keras model
    prob_y_train = model.predict(X_train)
    prob_y_test = model.predict(X_test)
    print(prob_y_train)
    print(prob_y_test)

    print("FCN Training done !")
    return model, prob_y_train, prob_y_test

# n_time_steps = Xy.shape[0]
# w = 40
# Xy_vals = Xy.T.values
# all_time_series = []
# for i in range(0, n_time_steps - w):
#     all_time_series.append(Xy_vals[:, i:i+1])
# print(np.array(all_time_series).shape)
