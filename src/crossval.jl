@doc """
LOOCV Leave-One-Out Cross-Validation for SPLS objects

Inputs:
    `etimator`: SPLS object
    `param_grid`: Dict, contains grid of parameters to be evaluated, with parameter
        names as Dict keys and values as arrays.
    `scoring`: not yet implemented (always scores by RMSE)
    `n_jobs`: not yet implemented (only one core so far)
    `refit`: Bool. Refit model at best parameters if true
    `verbose`: Bool.
    `scorer`: not yet implemented

Methods: SPLS's sklearn-compatible  API: `fit!(obj,X)`, `transform!(obj,X)`,
    `predict(obj,X)` and `predict_all(obj,X) with X: Data matrix.

Attributes:
    `best_params_`: Dict with optimal parameter values
    `best_score_`: Value for the score function at optimal set of parameters
    `grid_scores_`: Values for the score function for all grid settings.
    `best_estimator_`: SPLS object at optimal parameter settings
    `grid_predictions_`: Predictions for all grid settings
"""->


@with_kw mutable struct LOOCV <: ScikitLearn.Skcore.BaseSearchCV

    estimator
    param_grid
    scoring = nothing
    n_jobs = 1
    refit = true
    verbose = true
    scorer = nothing
    best_params_ = nothing
    best_score_ = nothing
    grid_scores_ = nothing
    best_estimator_ = nothing
    grid_predictions_ = nothing
end

function LOOCV(estimator,param_grid;kwargs...)

    LOOCV(estimator=estimator,param_grid=param_grid,kwargs...)

end

# using ScikitLearn

function fit!(cvobj::LOOCV,X,y)

    n,p = size(X)
    n_pars_test = length(cvobj.param_grid)
    keyspg = collect(keys(cvobj.param_grid))
    valuespg = collect(values(cvobj.param_grid))
    num_configurations = prod(length.(valuespg))
    if :n_components in keyspg
        wherekey = findall(keyspg .== :n_components)[1]
        maxcomponents = maximum(cvobj.param_grid[:n_components])
        valuespg[wherekey] = [maxcomponents]
    end
    pg = ScikitLearn.Skcore.ParameterGrid(Dict(zip(keyspg,valuespg)))
    lpg = length(pg)

    grid_scores = zeros(n,num_configurations)
    grid_predictions = zeros(n,num_configurations)
    if !cvobj.estimator.all_components
        spls.set_params!(cvobj.estimator,all_components=true)
    end

    for i in 1:n
        if cvobj.verbose
            print("Cross-validation " * string(round(i/n*100,digits=2)) * "% completed")
        end
        idx_train = setdiff(collect(1:n),i)
        for j in 1:lpg
            set_params_dict!(cvobj.estimator,pg[j])
            fit!(cvobj.estimator,X[idx_train,:],y[idx_train])
            ŷ = predict_all(cvobj.estimator,X[i,:])
            grid_predictions[i,((j-1)*maxcomponents + 1):(maxcomponents*j)] = ŷ
        end
    end

    deviations = grid_predictions .- y
    rmse = deviations.^2
    rmse = mapslices(mean,rmse,dims=1)
    rmse = sqrt.(rmse)

    indbest = findall(rmse .== minimum(rmse))[1][2]
    if :n_components in keyspg
        indotherpars = Int(floor(indbest/maxcomponents)) + 1
        indcomponents = indbest % maxcomponents
        print(string(indbest))
        print("\n" * string(maxcomponents))
        print("\n" * string(cvobj.param_grid[:n_components]))
        print("\n" * string(indcomponents))
        bestcomponents = collect(cvobj.param_grid[:n_components])[indcomponents+1]
        best_params = pg[indotherpars]
        best_params[:n_components] = bestcomponents
    else
        best_params = pg[indbest]
    end

    if cvobj.refit
        best_estimator = clone(cvobj.estimator)
        set_params_dict!(best_estimator,best_params)
        fit!(best_estimator,X,y)
    end

    setfield!(cvobj,:best_params_,best_params)
    setfield!(cvobj,:best_score_,rmse[indbest])
    setfield!(cvobj,:grid_scores_,rmse)
    setfield!(cvobj,:best_estimator_,best_estimator)
    setfield!(cvobj,:grid_predictions_,grid_predictions)

end

predict(cvobj::LOOCV,X) = predict(cvobj.best_estimator_,X)
predict_all(cvobj::LOOCV,X) = predict_all(cvobj.best_estimator_,X)
transform(cvobj::LOOCV,X) = transform(cvobj.best_estimator_,X)
