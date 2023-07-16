function _find_sparse(s,eta)

    """
    Generate sparse version of vector based on relative η criterion and identify
    informative variables
    """

    s ./= sqrt(sum(s.^2))
    goodies = abs.(s) .- eta*maximum(abs.(s))
    s = goodies .* sign.(s)
    goodies = findall(goodies .> 0)

    return(s,goodies)

end

function sparsify(s,eta,method="relative")

    """
    Generate sparse version of vector based on either relative η or abolsute λ
    criterion and identify informative variables
    """

    if method == "relative"
        (s,goodies) = _find_sparse(s,eta)
    elseif method == "absolute"
        (s,goodies) = _find_sparse_l(s,eta)
    else
        throw(DomainError("Methods accepted are `relative` or `absolute`"))
    end

    elimvars = setdiff(1:length(s),goodies)
    s[elimvars] .= 0

    return((s,goodies,elimvars))

end

function _find_sparse_l(s,lambda)

    """
    Generate sparse version of vector based on abolsute λ
    criterion and identify informative variables
    """

    s ./= sqrt(sum(s.^2))
    goodies = abs.(s) .- lambda/2
    s = goodies .* sign.(s)
    goodies = findall(goodies .> 0)

    return(s,goodies)

end

function dcd(X)

    """
    double centred distance matrix
    """

    X = pairwise(Euclidean(),X,dims=1)
    mdot = mapslices(mean,X,dims=1)
    return(X .- mdot .- mdot' .+ mean(X))

end

function _check_is_fitted(self;method_name::Symbol=:spls)

    """
    Check if SPLS model has been fit for a given SPLS object
    """

    if self.coef_ == nothing
        return false
    else
        return true
    end
end

function _predict_check(self,Xn)

    """
    Check data and SPLS object prior to making makings and change formats
    """

    @assert _check_is_fitted(self) "Please fit model before making predictions"
    Xn_type = typeof(Xn)
    @assert Xn_type in vcat(self.X_Types_Accept,self.y_Types_Accept) "Supply new X data as DataFrame or Matrix"
    if Xn_type == DataFrame
        Xn = Matrix(Xn)
    end
    if Xn_type in [DataFrameRow,DataFrameRow{DataFrames.DataFrame,DataFrames.Index}]
        Xn = Array(Xn)
    end
    nxn = size(Xn)
    if length(nxn)==1 #Transform one new case
        Xn = reshape(Xn,(1,nxn[1]))
        pxn = nxn[1]
        nxn = 1
    else
        pxn = nxn[2]
        nxn = nxn[1]
    end

    return(Xn,nxn,pxn)

end #_predict_check