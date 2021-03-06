import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer


def process_data(
        X,
        training=True,
        label=None,
        cat_features=[],
        ohe=None,
        lb=None,
        scaler=None):
    """ Process the data used in the \
        machine learning pipeline.
    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and\
             label. Columns in `categorical_features`
    label : str
        Name of the label column in `X`. If None,\
             then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or \
            inference/validation mode.
    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise\
             empty   np.array.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[cat_features].values
    X_continuous = X.drop(*[cat_features], axis=1)

    if training is True:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        scaler = StandardScaler()
        X_continuous = scaler.fit_transform(X_continuous)
        X_categorical = ohe.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = ohe.transform(X_categorical)
        X_continuous = scaler.transform(X_continuous)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    return X, y, lb, ohe, scaler
