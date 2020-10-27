using MLDataUtils
mutable struct SupportVectorMachine
    """Support Vector Machine Classifier

    Parameters
    ----------

    eta : float
        Learning rate (betweeb 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    shuffle : bool (default: true)
        Shuffles training data every epoch if True to prevent
        cycles.
    random_state : int
        Random number generator seed for random weight
        initialization.


    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value average over all
        training examples in each epoch.


    """
    eta::Float64
    n_iter::Int


    function SupportVectorMachine(eta::Float64, n_iter::Int)
        eta = 0.001
        n_iter = 1000
        weights = nothing
        b = nothing
    end
end


function fit(X, y)
    """Fit training data.

    Parameters
    ----------

    X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of
        examples and n_features is the number of features.
    y : array-like, shape = [n_examples]
        Target values.

    Returns
    -------
    self : object

    """
    n_samples, n_features = size(X)
    weights = zeros(size(X,1))
    b = 0
    for i in n_iter
        for idx, x_i in enumerate(X)
            condition = y[idx] * dot(x_i, weights) - b >= 1
            if condition:
                weights -= eta * (2 * 0.01 * weights)
            else
                weights -= eta * (2 * 0.01 * weights - dot(x_i,y[idx]))
                b -= eta * y[idx]
            end
        end
    end
end



function predict(features)
    """Return class label after unit step"""
    classification = dot(features, w) + b
    if classification >= 0
        classificationSign = '+'
    else
        classificationSign = '-'
    end
    return classificationSign
end
