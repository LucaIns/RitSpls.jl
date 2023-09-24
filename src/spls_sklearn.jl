@doc """
SPLS: Sparse Partial Least squares Algorithms


Parameters:
-----------
`eta`: float. Sparsity parameter in [0,1)

`n_components`: int, min 1. Note that if applied on data, n_components shall
    take a value <= minimum(size(x_data))

`weighted`: bool: if true, weight cases according to Hampel function

`verbose`: Bool: to print intermediate set of columns retained

`centre`: How to internally centre data. Accepts strings ("mean","median"),
    functions (e.g. Statistics.mean) or a location vector.

`scale`: How to internally scale data. Accepts strings ("std"),
        functions (e.g. Statistics.std) or a scale vector. Enter "none" both for
        centre and scale to work with the raw data.

`copy`: Bool, whether to copy data

`fit_algorithm`: Either "snipls" or "shelland"

`return_snipls_entities`: bool. When running `shelland`, set to true will not only
    return the Helland PLS vectors aₕ, ãₕ, Hₚ, T, but will also return vectors
    typical to SNIPLS such as R

`all_components`: bool. `true` will return a matrix B that contains all vectors
    of regression coefficients based on [0,1,...,n_components] components

`cutoff_probs`: list: cutoff probabilities for the Hampel function

Values:
-------
The mutable struct called by SPLS() will be populated with model results, such
    as coef_ (regression coefficients), x_scores_, etc., as well as estimated
    locations and scales for both X and y.


Examples
--------

The module is consistent with the ScikitLearn API, e.g.

    import ScikitLearn.GridSearch:GridSearchCV
    gridsearch = GridSearchCV(spls.SPLS(), Dict(:eta => collect(0.9:-0.01:0.1),
                    :n_components => collect(1:4), :verbose => false))
    ScikitLearn.fit!(gridsearch,X,y)
    gridsearch.best_params_
    ScikitLearn.predict(gridsearch.best_estimator_,Xt)

The module contains a Leave-One-Out cross-validation routine which leverages that
an h component model nests all models up to h-1 components in it (when
`all_components = true`). To use:

    loocv = LOOCV(SPLS(), Dict(:eta => 0.9:-0.02:0.1,:n_components => 1:5, :verbose => [false], :scale => ["none"]))
    fit!(loocv,X,y)
    loocv.best_params_

Example of an SPLS call

    snipreg = spls.SPLS()
    spls.set_params_dict!(snipreg,Dict(:eta => 0.5, :fit_algorithm=>"snipls", :n_components => 3, :all_components => true))
    fit!(snipreg,X,y)
    snipreg.x_weights_
    predict(snipreg,Xn)

Written by Sven Serneels.

References
----------

Remark: SNIPLS Algorithm first outlined in:
    Sparse and robust PLS for binary classification,
    I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza,
    Journal of Chemometrics, 30 (2016), 153-162.

""" ->
@with_kw mutable struct SPLS <: BaseRegressor
    eta::Number = .5
    n_components::Int = 1
    weighted = false
    verbose::Bool = true
    centre = "mean"
    scale = "none"
    copy::Bool = true
    fit_algorithm = "snipls"
    return_snipls_entities = true
    all_components = true
    cutoff_probs = [.95,.975,.999]
    X_Types_Accept = [DataFrame,Array{Number,2},Array{Float64,2},
                    Array{Float32,2},Array{Int32,2},Array{Int64,2},
                    Array{Union{Missing, Float64},2}]
    y_Types_Accept = [DataFrame,DataFrameRow{DataFrame,DataFrames.Index},
                    Array{Number,1},Array{Float64,1},Array{Float32,1},
                    Array{Int32,1},Array{Int64,1},Array{Union{Missing, Float64},1}]
    _ṁ = nothing
    _m̈ = nothing
    _cutoffs = nothing
    X0 = nothing
    y0 = nothing
    Xs_ = nothing
    Xa_ = nothing
    ys_ = nothing
    x_weights_ = nothing
    x_Helland_weights_ = nothing
    x_loadings_ = nothing
    C_ = nothing
    Hₚ = nothing
    x_scores_ = nothing
    coef_ = nothing
    coef_scaled_ = nothing
    all_coeffs_ = nothing
    intercept_ = nothing
    all_intercepts_ = nothing
    x_ev_ = nothing
    y_ev_ = nothing
    fitted_ = nothing
    all_fits_ = nothing
    all_fitted = nothing
    residuals_ = nothing
    all_residuals_ = nothing
    x_Rweights_ = nothing
    colret_ = nothing
    x_loc_ = nothing
    y_loc_ = nothing
    x_sca_ = nothing
    y_sca_ = nothing
    x_names = nothing
    y_name = nothing
end

function SPLS(eta,n_components;kwargs...)

    SPLS(eta=eta,n_components=n_components;kwargs...)

end

@declare_hyperparameters(SPLS, [:n_components, :eta, :centre, :scale
])

@doc """

    Dummy function equivalent to directly creating a SNIPLS struct

    """ ->
function call(;kwargs...)

    self = SPLS()
    if length(kwargs) > 0
        allkeys = keys(kwargs)
        for j in allkeys
            setfield!(self,j,kwargs[j])
        end #for
    end #if
    return(self)
end #snipls

@doc """

    Fit SPLS model to data X and y.

    """ ->
function fit!(self::SPLS,X,y)

    @assert self.fit_algorithm in ["snipls","shelland"] "Algorithm has to be `snipls` or `shelland`]"

    if typeof(self.centre)==String
        if self.centre in ["mean","median"]
            self.centre = getfield(Statistics,Symbol(self.centre))
        else
            @assert self.centre=="none" "Only supported strings for median:" * "\n" *
                 "'mean', 'median', 'none'" * "\n" *
                 "Alternatively pass a function"
            # other location estimators can be included
        end
    end

    if typeof(self.scale)==String
        if self.scale in ["std"]
            self.scale = getfield(Statistics,Symbol(self.scale))
        else
            @assert self.scale=="none" "Only supported strings for scale:" * "\n" *
                 "'std','none'" * "\n" *
                 "Alternatively pass a function"
        end
    end

    X_Type = typeof(X)
    y_Type = typeof(y)
    @assert X_Type in self.X_Types_Accept "Supply X data as DataFrame or Matrix"
    @assert y_Type in self.y_Types_Accept "Supply y data as DataFrame or Vector"

    if self.copy
        setfield!(self,:X0,deepcopy(X))
        setfield!(self,:y0,deepcopy(y))
    end

    X0 = X
    y0 = y

    if y_Type == DataFrame
        ynames = true
        y_name = names(y0)
        y0 = y[:,1]
    else
        ynames = false
    end

    if X_Type == DataFrame
        Xnames = true
        X_name = names(X0)
        X0 = Matrix(X0)
    else
        Xnames = false
        X_name = nothing
    end

    allr = nothing
    allintercept = nothing
    allfit = nothing
    allB = nothing

    (n,p) = size(X0)
    ny = size(y0,1)
    @assert ny == n "Number of cases in X and y needs to agree"

    if self.weighted
        cutoffs = quantile.(Chi(p),self.cutoff_probs)
        setfield!(self,:_cutoffs, cutoffs)
    end

    Xa = deepcopy(X0)
    ya = deepcopy(y0)
    if self.copy
        setfield!(self,:X0,X0)
        setfield!(self,:y0,y0)
    end
    ṗ = p

    centring_X = autoscale(Xa,self.centre,self.scale)
    Xs= centring_X.X_as_
    mX = centring_X.col_loc_
    sX = centring_X.col_sca_

    centring_y = autoscale(y0,self.centre,self.scale)
    ys= centring_y.X_as_
    my = centring_y.col_loc_
    sy = centring_y.col_sca_

    S = Xs' * Xs
    s = Xs' * ys
    nys = sum(ys.^2)

    if self.fit_algorithm == "snipls"
        (W,P,R,C,T,B,Xev,yev,colret,allB) = _fit_snipls(self.n_components,self.eta,n,ṗ,Xs,ys,nys,Xnames,X_name,s,
            self.all_components,false,self.verbose)
    elseif self.fit_algorithm == "shelland"
        S = Xs'*Xs
        if self.return_snipls_entities
            (W,P,R,C,T,B,Xev,yev,colret,Ã,Hₚ,allB) = _fit_shelland(self.n_components,self.eta,n,ṗ,Xs,ys,nys,Xnames,X_name,s,S,
            self.return_snipls_entities,self.all_components,self.verbose)
        else
            (W,Ã,T,Hₚ,B,colret,allB) = _fit_shelland(self.n_components,self.eta,n,ṗ,Xs,ys,nys,Xnames,X_name,s,S,
            self.return_snipls_entities,self.all_components,self.verbose)
            P = nothing
            R = nothing
            C = nothing
            Xev = nothing
            yev = nothing

        end
    end

    if self.all_components
        B_rescaled = (sy./sX)' .* allB
    else
        B_rescaled = (sy./sX)' .* B
    end

    yp_rescaled = Xa*B_rescaled

    if self.centre != "none"
        intercept = mapslices(self.centre,y .- yp_rescaled,dims=1)
    else
        intercept = mean(y .- yp_rescaled)
    end

    yfit = yp_rescaled .+ intercept
    r = y .- yfit
    if self.all_components
        allfit = yfit
        yfit = yfit[:,self.n_components]
        allr = r
        r = r[:,self.n_components]
        allB = B_rescaled
        B_rescaled = B_rescaled[:,self.n_components]
        allintercept = intercept
        intercept = intercept[:,self.n_components]
    end
    setfield!(self,:x_weights_,W)
    setfield!(self,:x_Rweights_,R)
    setfield!(self,:x_loadings_,P)
    setfield!(self,:C_,C)
    setfield!(self,:x_scores_,T)
    setfield!(self,:coef_,B_rescaled)
    setfield!(self,:all_coeffs_,allB)
    setfield!(self,:coef_scaled_,B)
    setfield!(self,:intercept_,intercept)
    setfield!(self,:all_intercepts_,allintercept)
    setfield!(self,:x_ev_,Xev)
    setfield!(self,:y_ev_,yev)
    setfield!(self,:fitted_,yfit)
    setfield!(self,:all_fits_,allfit)
    setfield!(self,:residuals_,r)
    setfield!(self,:all_residuals_,allr)
    setfield!(self,:x_Rweights_,R)
    setfield!(self,:colret_,colret)
    setfield!(self,:Xs_,Xs)
    setfield!(self,:ys_,ys)
    setfield!(self,:x_loc_,mX)
    setfield!(self,:y_loc_,my)
    setfield!(self,:x_sca_,sX)
    setfield!(self,:y_sca_,sy)
    if Xnames
        setfield!(self,:x_names,X_name)
    end
    if ynames
        setfield!(self,:y_name,y_name)
    end
    if self.fit_algorithm == "shelland"
        setfield!(self,:x_Helland_weights_,Ã)
        setfield!(self,:Hₚ,Hₚ)
    end

    return(self)

end


@doc """

    Fit SPLS model to data X and y and only return the regression
    coefficients.

    """ ->
function fit(self::SPLS,X,y)

    if self.X0 == nothing
        fit!(self,X,y)
    end

    return(self.coef_)

end

@doc """

    Predict responses for new predictor data.

    """ ->
function predict(self::SPLS,Xn)

    Xn, nxn, pxn = _predict_check(self,Xn)
    (n,p) = size(Xn)
    @assert p==length(self.coef_) "New data must have same number of columns as the ones the model has been trained with"
    return(Xn * self.coef_ .+ self.intercept_[:,1])

end

@doc """

    For a model with `n_components` = k, predict the dependent variable for all
    settings 1:k.

    """ ->
function predict_all(self::SPLS,Xn)

    @assert self.all_components "To predict full set of components, flag `all_components` needs to be `true`"
    Xn, nxn, pxn = _predict_check(self,Xn)
    (n,p) = size(Xn)
    @assert p==length(self.coef_) "New data must have same number of columns as the ones the model has been trained with"
    return((Xn * self.all_coeffs_ .+ self.all_intercepts_))

end

@doc """

    Transform new predictor data to estimated scores.

    """ ->
function transform(self::SPLS,Xn)

    Xn, nxn, pxn = _predict_check(self,Xn)
    (n,p) = size(Xn)
    @assert p==length(self.coef_) "New data must have same number of columns as the ones the model has been trained with"
    Xnc = autoscale(Xn,self.x_loc_,self.x_sca_).X_as_
    return(Xnc*self.x_Rweights_)

end

@doc """

    Get all settings from an existing SPLS struct (also the ones not
        declared as parameters to ScikitLearn)

    """ ->
function get_all_params(self::T, param_names=[]::Array{Any}) where{T}

    if length(param_names)==0

        params_dict = type2dict(self)

    else

        params_dict = Dict{Symbol, Any}()

        for name::Symbol in param_names
            params_dict[name] = getfield(self, name)
        end

    end

    params_dict

end

@doc """

    ScikitLearn similar function to set parameters in an existing SPLS
        struct, yet takes a Dict as an argument.

    Compare:
    ScikitLearn.set_params!(lnrj,eta=.3)
    spls.set_params_dict!(lnrj,Dict(:eta => .3))

    """ ->
function set_params_dict!(self::T, params; param_names=nothing) where {T}

    for (k, v) in params

        if param_names !== nothing && !(k in param_names)

            throw(ArgumentError("An estimator of type $T was passed the invalid hyper-parameter $k. Valid hyper-parameters: $param_names"))

        end

        setfield!(self, k, v)

    end

    if self.verbose
        print(self)
    end

end

clone_param(v::Any) = v # fall-back

function is_classifier(self::SPLS) return(false) end

@doc """

    ScikitLearn compatible function to clone an existing SPLS struct.

    """ ->
function clone(self::T) where {T}

    kw_params = Dict{Symbol, Any}()

    # cloning the values is scikit-learn's default behaviour. It's ok?

    for (k, v) in get_params(self) kw_params[k] = clone_param(v) end

    return T(; kw_params...)

end

function get_params(self::SPLS)
    return spls.get_all_params(self)
end

function set_params!(self::SPLS; params...)
    return spls.set_params_dict!(self,params)
end

function Base.copy!(newcopy::SPLS, self::SPLS)
    # shallow copy - used below
    for f in fieldnames(typeof(newcopy))
        setfield!(newcopy, f, getfield(self, f))
    end
end

function fit_if(self::SPLS,X,y,xi,yi)

    if self.fit_algorithm != "shelland"
        throw("ASV requires Sparse Helland fit, please change fit_algorithm")
    end

    if self.x_Helland_weights_ == nothing
        set_params_dict(self,Dict(:fit_algorithm=>"shelland"))
        fit!(self,X,y)
    end

    IFᵦ, IFã = _if(self,xi,yi)

    return(IFᵦ, IFã)

end

function _if_map(xi,yi,self::SPLS)

    IFB, IFA = _if(self,xi,yi)

    return(IFB[:,self.n_components])

end

function sid(self::SPLS,X,y)

    if self.fit_algorithm != "shelland"
        throw("ASV requires Sparse Helland fit, please change fit_algorithm")
    end

    n,p = size(X)
    all_if = zeros(p,n)
    for i = 1:n
        all_if[:,i] = _if_map(X[i,:],y[i],self).^2
    end
    return(mapslices(sum,all_if,dims=1)/(p-self.n_components))

end

function asv(self::SPLS,X,y)

    if self.fit_algorithm != "shelland"
        throw("ASV requires Sparse Helland fit, please change fit_algorithm")
    end

    n,p = size(X)
    all_if = zeros(p,n)
    for i = 1:n
        all_if[:,i] = _if_map(X[i,:],y[i],self).^2
    end
    return(mapslices(sum,all_if,dims=2)/n)

end

function _if(self::SPLS,xi,yi)

    Xs = self.Xs_
    ys = self.ys_
    n,p = size(Xs)
    s = Xs'*ys/n
    S = Xs'*Xs/n
    IFS = xi*xi'- S
    IFs = xi*yi - s
    IFa = deepcopy(IFs)
    IFat = deepcopy(IFs)
    IFHₚ = zeros(p,p)
    B = zeros(p,self.n_components)
    IFÃ = zeros(p,self.n_components)
    IFB = zeros(p,self.n_components)
    Iₚ = diagm(ones(p))
    Hₚ = zeros(p,p)

    for i = 1:self.n_components

        if i>1

            IFa=(Iₚ-S*Hₚ)*IFs - (S*IFHₚ + IFS*Hₚ)*s
            IFa[findall(abs.(self.x_weights_[:,i]).<1e-6)] .= 0
            IFat = (Iₚ- Hₚ*S)*IFa - (IFHₚ*S + Hₚ*IFS)*self.x_weights_[:,i]

        end

        ãᵢ = self.x_Helland_weights_[:,i]
        nti2 = ãᵢ'*S*ãᵢ #sum((Xs*ãᵢ).^2)
        IFHₚ += (ãᵢ*IFat' + IFat*ãᵢ')/nti2
        IFHₚ -= ãᵢ*ãᵢ'/nti2^2 * (ãᵢ'*S*IFat + ãᵢ'*IFS*ãᵢ + IFat'*S*ãᵢ)
        Hₚ += (ãᵢ*ãᵢ')/nti2
        b = Hₚ*s
        IFb = IFHₚ*s + Hₚ*IFs
        B[:,i] = b
        IFÃ[:,i] = IFat
        IFB[:,i] = IFb
    end

    return(IFB,IFÃ)

end
