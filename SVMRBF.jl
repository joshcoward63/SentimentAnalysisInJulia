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
    C::Int
    kernel::String
    power::Int
    γ::float
    coef::Int
    lagr_mulitpliers::nothing
    support_vectors::nothing
    support_vector_labels::nothing
    intercept::nothing

    function SupportVectorMachine(C::Integer, kernel::String, power::Integer, γ::γ, coef::Integer)
        C = 1
        kernel = rbf_kernel
        power = 4
        γ = nothing
        coef = 4
        lagr_mulitpliers = nothing
        support_vectors = nothing
        support_vector_labels = nothing
        intercept = nothing
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
    #set γ to 1/n_features by default
    if ∉ γ
        γ = 1 / n_features
    end
    #Initialize kernel method with Parameters
    kernel = kernel(power=power;
    γ=γ,
    coef=coef)

    #Calculate kernel matrix
    kernel_matrix = zeros(n_samples, n_samples)
    for i in n_samples
        for j in n_samples
            kernel_matrix[i, j] = kernel(X[i], X[j])
        end
    end

    #Define the quadratic optimization problem
    p = [y * transpose(y) * kernel_matrix]

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
