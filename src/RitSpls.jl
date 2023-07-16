module RitSpls

using ScikitLearn, Statistics, DataFrames, Parameters, ForwardDiff, CSV
import ScikitLearn.GridSearch:GridSearchCV
import ScikitLearnBase: BaseRegressor, BaseEstimator, predict, fit!, fit_transform!, @declare_hyperparameters, is_classifier, clone, transform
import LinearAlgebra: diagm, norm, eigen
import Optim: optimize, minimizer, LBFGS    
    include("_preproc_utils.jl")
    include("_rob_utils.jl")
    include("_sreg_utils.jl")
    include("crossval.jl")
    include("gsspp.jl")
    include("modelmetrics.jl")
    include("spls_algorithms.jl")
    include("spls_sklearn.jl")
    export RitSpls, GSSPP, wrap, autoscale, set_params_dict!, mad
end
