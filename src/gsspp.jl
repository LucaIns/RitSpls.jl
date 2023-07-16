    @doc """
    Generalized Spatial Sign Pre-Processing as a scikit-learn compatible object
    that can be used in ML pipelines.

    Inputs:
        `centre`: str or function, location estimator for centring.
                str options: "mean", "median", "none"
        `fun`: str or function, radial transformation function,
                str options: 'ss' (the non-generalized spatial sign, equivalent
                to sklearn's Normalizer) or 'linear_redescending'
    Methods: sklearn API: `fit!(obj,X)`, `transform!(obj,X)` and `fit_transform!(obj,X)` with
        X: Data matrix

    Attributes:
        `gss_`: the generalized spatial signs
        `Xm_`: the centred data
        `centre_`: X location estimate
        `'X_gss_pp_'`: Data Preprocessed by Generalized Spatial Sign
    """->

@with_kw mutable struct GSSPP <: BaseEstimator
    centre = kstepLTS
    fun = quad
    gss_ = nothing
    Xm_ = nothing
    centre_ = nothing
    X_gsspp_ = nothing
    _d_hmed = nothing
    _cutoff = nothing
end

@declare_hyperparameters(GSSPP, [:centre,:fit_algorithm])

function fit!(self::GSSPP,X)

    """
    Calculate and store generalized spatial signs
    """
    X = convert.(AbstractFloat, Matrix(X))
    # X = _check_input(X)
    n,p = size(X)
    if typeof(self.fun) == String
        fun = getfield(RitSpls,Symbol(self.fun))
    else
        fun = self.fun
    end
    if (self.centre==kstepLTS) || (self.centre=="kstepLTS") # it is not columnwise as expected by a fun used in autoscale
       loc = RitSpls.kstepLTS(Matrix(X))
       self.centre = loc
    end
    Xm,loc,sca = autoscale(X,self.centre,"none")    
    gss_ = fun(_norms(Xm),p,n)
    setfield!(self,:gss_,gss_[1])
    setfield!(self,:_d_hmed,gss_[2])
    setfield!(self,:_cutoff,gss_[3])
    setfield!(self,:centre_,loc)
    setfield!(self,:Xm_,Xm)

end

function transform!(self::GSSPP,X)

    """
    Calculate Generalized Spatial Sign Pre-Pprocessed Data
    """

    if getfield(self,:X_gsspp_)==nothing
        fit!(self,X)
    end
    Xgss = self.Xm_ .* self.gss_
    setfield!(self,:X_gsspp_,Xgss)
    return(Xgss)

end

function fit_transform!(self::GSSPP,X)

    fit!(self,X)
    transform!(self,X)

    return(self.X_gsspp_)

end
