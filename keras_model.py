from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential

# Defining Keras model creator function
def get_clf(meta, hidden_layer_sizes, dropout):
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]
    model = Sequential()
    model.add(Dense(n_features_in_, activation='relu', input_shape=X_shape_[1:]))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(Dense(hidden_layer_size, activation="relu"))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    return model