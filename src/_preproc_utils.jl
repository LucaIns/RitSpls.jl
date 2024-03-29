# See also :
# https://github.com/SvenSerneels/unisimpls/blob/master/src/_preproc_utils.jl

@doc """

    Autoscale a data matrix

    Either pass on functions of location and scale estimators,
    or strings (e.g. Statistics.median or "median") or "none"
    if no centring and/or scaling needed.

    Output is a NamedTuple containing the autoscaled data, and the location
    and scale estimates.

    """ ->
function autoscale(X::Matrix,locest,scalest)

    if typeof(locest)==String
        if locest in ["mean","median"]
            locest = getfield(Statistics,Symbol(locest))
        else
            # other location estimators to be included
            if locest != "none"
                @warn("Only supported strings for median:" * "\n" *
                    "'mean', 'median', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_loc = false
    else
        if typeof(locest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},
                              Array{Float64,2},Array{Int64,2},Array{Float32,2},Array{Int32,2}]
            pre_estimated_loc = true
        else
            pre_estimated_loc = false
        end
    end

    if typeof(scalest)==String
        if scalest == "std"
            scalest = getfield(Statistics,Symbol(scalest))
        else
            # Further scale estimates to be included
            if scalest != "none"
                # Further scale estimates to be included
                @warn("Only supported strings for scale:" * "\n" *
                    "'std', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_sca = false
    else
        if typeof(scalest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},
                            Array{Float64,2},Array{Int64,2},Array{Float32,2},Array{Int32,2}]
            pre_estimated_sca = true
        else
            pre_estimated_sca = false
        end
    end

    (n,p) = size(X)

    if pre_estimated_loc
        x_loc_ = locest
        if length(size(x_loc_))==1
            x_loc_ = x_loc_'
        end
    else
        if locest == "none"
            x_loc_ = zeros(1,p)
        else
            x_loc_ = mapslices(x -> locest(x),X,dims=1)
        end
    end

    if pre_estimated_sca
        x_sca_ = scalest
        if length(size(x_sca_))==1
            x_sca_ = x_sca_'
        end
    else
        if scalest == "none"
            x_sca_ = ones(1,p)
        else
            x_sca_ = mapslices(x -> scalest(x),X,dims=1)
        end
    end

    if (locest == "none" && scalest == "none")
        X_as_ = X
    else
        X_as_ = (X - ones(n,1)*x_loc_) ./ (ones(n,1)*x_sca_)
    end
    return(X_as_ = X_as_, col_loc_ = x_loc_, col_sca_ = x_sca_)

end

@doc """

    Autoscale a data frame

    Either pass on functions of location and scale estimators,
    or strings (e.g. Statistics.median or "median") or "none"
    if no centring and/or scaling needed.

    Output is a NamedTuple containing the autoscaled data, and the location
    and scale estimates.

    """ ->
function autoscale(X::DataFrame,locest,scalest)

    if typeof(locest)==String
        if locest in ["mean","median"]
            locest = getfield(Statistics,Symbol(locest))
        else
            # other location estimators can be included
            if locest != "none"
                @warn("Only supported strings for median:" * "\n" *
                    "'mean', 'median', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_loc = false
    else
        if typeof(locest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},
                            Array{Float64,2},Array{Int64,2},Array{Float32,2},Array{Int32,2}]
            pre_estimated_loc = true
        else
            pre_estimated_loc = false
        end
    end

    if typeof(scalest)==String
        if scalest == "std"
            scalest = getfield(Statistics,Symbol(scalest))
        else
            if scalest != "none"
                # Further scale estimates to be included
                @warn("Only supported strings for scale:" * "\n" *
                    "'std', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_sca = false
    else
        if typeof(scalest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},
                            Array{Float64,2},Array{Int64,2},Array{Float32,2},Array{Int32,2}]
            pre_estimated_sca = true
        else
            pre_estimated_sca = false
        end
    end

    (n,p) = size(X)
    x_names = names(X)

    if pre_estimated_loc
        x_loc_ = locest
        if length(size(x_loc_))>1
            x_loc_ = x_loc_'
        end
    else
        if locest == "none"
            x_loc_ = zeros(1,p)
        else
            x_loc_ = mapslices(x -> locest(x),Array(X),dims=1)
        end
    end

    if pre_estimated_sca
        x_sca_ = scalest
        if length(size(x_sca_))>1
            x_sca_ = x_sca_'
        end
    else
        if scalest == "none"
            x_sca_ = ones(1,p)
        else
            x_sca_ = mapslices(x -> scalest(x),Array(X),dims=1)
        end
    end
    if (locest == "none" && scalest == "none")
        X_as_ = X
    else
        X_as_ = (Array(X) .- x_loc_) ./ (x_sca_)
        X_as_ = DataFrame(X_as_,:auto)
    end
    rename!(X_as_,[Symbol(n) for n in x_names])
    return(X_as_ = X_as_, col_loc_ = x_loc_, col_sca_ = x_sca_)

end

@doc """

    Autoscale a data vector

    Either pass on functions of location and scale estimators,
    or strings (e.g. Statistics.median or "median") or "none"
    if no centring and/or scaling needed.

    Output is a NamedTuple containing the autoscaled data, and the location
    and scale estimates.

    """ ->
function autoscale(X::Vector,locest,scalest)

    if typeof(locest)==String
        if locest in ["mean","median"]
            locest = getfield(Statistics,Symbol(locest))
        else
            # other location estimators to be included
            if locest != "none"
                @warn("Only supported strings for median:" * "\n" *
                    "'mean', 'median', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_loc = false
    else
        if typeof(locest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},Float64,Int64,Float32,Int32]
            pre_estimated_loc = true
        else
            pre_estimated_loc = false
        end
    end

    if typeof(scalest)==String
        if scalest == "std"
            scalest = getfield(Statistics,Symbol(scalest))
        else
            if scalest != "none"
                # Further scale estimates to be included
                @warn("Only supported strings for scale:" * "\n" *
                    "'std', 'none'" * "\n" *
                    "Alternatively pass a function")
            end
        end
        pre_estimated_sca = false
    else
        if typeof(scalest) in [Array{Float64,1},Array{Int64,1},Array{Float32,1},Array{Int32,1},Float64,Int64,Float32,Int32]
            pre_estimated_sca = true
        else
            pre_estimated_sca = false
        end
    end

    if pre_estimated_loc
        x_loc_ = locest
    else
        if locest == "none"
            x_loc_ = 0
        else
            x_loc_ = locest(X)
        end
    end

    if pre_estimated_sca
        x_sca_ = scalest
    else
        if scalest == "none"
            x_sca_ = 1
        else
            x_sca_ = scalest(X)
        end
    end
    if (locest == "none" && scalest == "none")
        X_as_ = X
    else
        X_as_ = (X .- x_loc_) ./ x_sca_
    end
    return(X_as_ = X_as_, col_loc_ = x_loc_, col_sca_ = x_sca_)

end

function _norms(X)

    """
    Casewise norms of a matrix
    """
        return(mapslices(norm,X,dims=2))
end
