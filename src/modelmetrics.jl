function mean_squared_error(y,ŷ,locest="mean")
    if typeof(locest) == String
        locest = getfield(Statistics,Symbol(locest))
    end
    return(locest((y .- ŷ).^2))
end

function r2(y,ŷ,locest="mean")
    if typeof(locest) == String
        locest = getfield(Statistics,Symbol(locest))
    end
    den = sum((y .- ŷ).^2)
    num = sum((y .- locest(y)).^2)
    return(1 .- (den./num))
end

function mean_absolute_error(y,ŷ,locest="mean")
    if typeof(locest) == String
        locest = getfield(Statistics,Symbol(locest))
    end
    return(locest(abs.(y .- ŷ)))
end

function maximum_absolute_error(y,ŷ)
    if typeof(locest) == String
        locest = getfield(Statistics,Symbol(locest))
    end
    return(maximum(abs.(y .- ŷ)))
end

Scorer(object,X,y;fun::Function=r2) = fun(y,predict(object,X))
