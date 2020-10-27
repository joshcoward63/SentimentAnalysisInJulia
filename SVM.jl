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
    shuffle::bool
    random_state::int


    function SupportVectorMachine(eta::Float64, n_iter::Int, shuffle::bool, random_state::Int)
        eta = 0.1
        n_iter = 10
        shuffle = true
        random_state = nothing
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

end

function partial_fit(X, y):
    """Fit training data without reinitializing weights"""
end

function shuffle(X,y)
    """Shuffles training data."""
end

function _initialize_weights(xi, target)
    """Initialize weights to small random numbers"""
end

function _update_weights(xi, target)
    """Apply learning rule. """
end

function net_input(X)
    """Calculate net input"""
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
