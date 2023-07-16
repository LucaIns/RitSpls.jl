# Credits:
  # Spatial median
    # https://github.com/bcbi/MultivariateMedians.jl/blob/master/docs/src/examples.md
  # wrapping
    # https://rdrr.io/cran/cellWise/src/R/Wrap.R
  # GSSPP:
    # https://wis.kuleuven.be/stat/robust/Programs/GSSCM/gsscm-code.r

function l2_norm(z::AbstractVector)
    return sqrt(sum(abs2, z))
end

function l2_distance(x::AbstractVector, y::AbstractVector)
    z = x - y
    return l2_norm(z)
end

function geometric_median(a::AbstractVector{V};
                                  x0 = zeros(T, length(first(a)))) where V <: AbstractVector{T} where T
    function _sum_of_distances(x)
        return sum(l2_distance.(a, Ref(x)))
    end
    result = optimize(_sum_of_distances,
                            x0, 
                            LBFGS(); autodiff = :forward)
    minimiz = minimizer(result)
    return minimiz
end

function l1median(X, args...)
  """
  computes the spatial/geometric/L1 median
  args:
      X: data matrix of size n times p
      args flag to be able to pass on n - has no effect
    returns:
      l1med: p-dimensional vector of the coordinates of the L1 median
  """
  X = convert.(AbstractFloat, X)
  xsvec = mapslices(x->[x], X, dims=2)[:]
  l1med = geometric_median(xsvec)
  return(l1med = l1med)
end

function kstepLTS(X; maxit=5, args...)
  """
  Computes the K-step LTS estimator of location
  args:
      X: data matrix
      maxit: maximum number of iterations
      # tol: convergence tolerance
      args flag to be able to pass on n - has no effect
    returns:
      m2: location estimate
  """
  tol = 1e-10
  n  = size(X, 1)
  m1 = l1median(X) # initial estimate
  m2 = m1
  iter = 0
  converged = false
  while ((!converged) && (iter < maxit))  
  dists     = mapslices(x -> sum((x .- m2).^2), X, dims=2)
    cutdist   = sort(dists, dims=1)[Int(floor((n + 1) / 2))]
    hsubset   = findall(dists[:,1] .<= cutdist)
    m2        = mapslices(x -> Statistics.mean(x), X[hsubset, :], dims=1)'
    converged   = maximum(abs.(m1 .- m2)) < tol
    iter += 1
    m1 = m2
  end  
  return(m2')
end


function hampel(x,cutoffs)
    """
    Computes the Hampel redescending function

    args:
        x: input as Vector
        cutoffs, onsisting of: probct, hampelb, hampelr, cutoff values for (reweighting,
        harsher reweighting, rejection). If x~Dist than good values for these
        constants can be based on the quantiles of Dist.

    values:
        wx: reweighted x
    """

    probct,hampelb,hampelr = cutoffs
    wx = deepcopy(x)
    wx[findall(x .<= probct)] .= 1
    wx[findall((x .> probct) .& (x .<= hampelb))] .= probct./abs.(x[
            findall((x .> probct) .& (x .<= hampelb))])
    wx[findall((x .> hampelb) .& (x .<= hampelr))] .=
            probct * (hampelr .- (x[findall((x .> hampelb)
            .& (x .<= hampelr))])) ./
            ((hampelr - hampelb) * abs.(x[findall((x .> hampelb) .&
            (x .<= hampelr))]))
    wx[findall(x .> hampelr)] .= 0

    return(wx)

end

function mad(x,c=1.4826)
    """
    Consitency corrected median absolute deviation estimator.
    """
    return(c*median(abs.(x .- median(x))))
end


function ss(dd, p, args...)
  """
  Computes the spatial sign radial function
  args:
    dd: vector of distances
    p: dimension of original data
    args flag to be able to pass on n - has no effect
  returns:
    xi: radial function
  """
  prec = 1e-10
  dd = max.(dd,prec)
  xi = 1 ./ dd
  return(xi=xi)
end

function shell(dd, p,n,d_hmed=nothing,cutoff=nothing,h=Int(floor(n*0.75)))
  """
  # Computes the Shell radial function
  args:
    dd: vector of distances
    p: number of variables in original data
    n: number of rows in original data
    h: Int(floor(n*0.75)) provides a 25% BdP
    ...
  returns:
    xi: radial function
  note:
    Cutoffs are based on hmed/hmad as in the paper, designed to avoid implosion
    beakdown. When n <= (n+p+1)/2, med/mad are used instead of hmed/hmad, with
    the notion that implosion breakdown is possible in such a case.
  """
  if ((d_hmed == nothing) | (cutoff==nothing))
      dWH = dd.^(2/3)
      dWH_hmed = sort(dWH,dims=1)[h]
      dWH_hmad = sort(abs.(dWH .- dWH_hmed),dims=1)[h]
      # d_hmed = dWH_hmed^(3/2)
      cutoff1 = (maximum([0, dWH_hmed - dWH_hmad])) ^ (3 / 2)
      cutoff2 = (dWH_hmed + dWH_hmad)^(3/2)
  end
  idxlow = findall(dd .< cutoff1)
  idxhigh = findall(dd .> cutoff2)
  xi = ones(n)# xi = ones((n,1))
  xi[idxlow] .= 0
  xi[idxhigh] .= 0
  return(xi,dWH_hmed,cutoff)
end


function LR(dd, p,n,d_hmed=nothing,cutoff=nothing,h=Int(floor(n*0.75)))
  """
  # Computes the Linear redescending radial function
  args:
    dd: vector of distances
    p: number of variables in original data
    n: number of rows in original data
    h: Int(floor(n*0.75)) provides a 25% BdP
  returns:
    xi: radial function
  note:
    Cutoffs are based on hmed/hmad as in the paper, designed to avoid implosion
    beakdown. When n <= (n+p+1)/2, med/mad are used instead of hmed/hmad, with
    the notion that implosion breakdown is possible in such a case.
  """
  if ((d_hmed == nothing) | (cutoff==nothing))
      dWH = dd.^(2/3)
      dWH_hmed = sort(dWH,dims=1)[h]
      dWH_hmad = sort(abs.(dWH .- dWH_hmed),dims=1)[h]
      d_hmed = dWH_hmed^(3/2)
      cutoff = (dWH_hmed + 1.4826 * dWH_hmad)^(3/2)
  end
  idxmid = findall((dd .> d_hmed) .& (dd .<= cutoff))
  idxhigh = findall(dd .> cutoff)
  xi = ones(n)# xi = ones((n,1))
  xi[idxmid] = 1 .- (dd[idxmid,:] .- d_hmed) ./ (cutoff .- d_hmed)
  xi[idxhigh] .= 0
  return(xi,d_hmed,cutoff)
end


function winsor(dd,p,n,d_hmed=nothing,cutoff=nothing,h=Int(floor(n*0.75)))
  """
  # Computes the Winsor radial function
  args:
    dd: vector of distances
    p: number of variables in original data
    n: number of rows in original data
    h: Int(floor(n*0.75)) provides a 25% BdP
    ...
  returns:
    xi: radial function
  """
  d_hmed = sort(dd,dims=1)[h]
  idx = findall(dd .> d_hmed)
  xi = ones(n)
  xi[idx] =  1 ./ dd[idx] * d_hmed
  return(xi,d_hmed,nothing)
end


function quad(dd,p,n,d_hmed=nothing,cutoff=nothing,h=Int(floor(n*0.75)))
  """
  # Computes the Quadratic Winsor radial function
  args:
    dd: vector of distances
    p: number of variables in original data
    n: number of rows in original data
    h: Int(floor(n*0.75)) provides a 25% BdP
    ...
  returns:
    xi: radial function
  """
  d_hmed = sort(dd,dims=1)[h]
  idx = findall(dd .> d_hmed)
  xi = ones(n)
  xi[idx] =  1 ./ dd[idx].^2 * d_hmed.^2
  return(xi,d_hmed,nothing)
end


function ball(dd,p,n,d_hmed=nothing,cutoff=nothing,h=Int(floor(n*0.75)))
  """
  # Computes the Ball radial function
  args:
    dd: vector of distances
    p: number of variables in original data
    n: number of rows in original data
    h: Int(floor(n*0.75)) provides a 25% BdP
    ...
  returns:
    xi: radial function
  """
  dWH = dd.^(2/3)
  dWH_hmed = sort(dWH,dims=1)[h]
  d_hmed = dWH_hmed^(3/2)
  idx = findall(dd .> d_hmed)
  xi = ones(n)
  xi[idx] .= 0
  return(xi,d_hmed,nothing)
end


function wrap_univ(y, b=1.5, c = 4, q1 = 1.540793, q2 = 0.8622731, locest="median", scalest=mad)
  """
  # Univariate warapping transformation
  # (see: https://rdrr.io/cran/cellWise/src/R/Wrap.R)
  args:
    y: input vector
  returns:
    xi: output vector with wrapped data
  note:
    b = 1.5 and c = 4 to achieve a BdP of approx 15% for normal data
  """
  xi,locX,scaleX = autoscale(y,locest,scalest)
  indMid = findall((abs.(xi) .< c) .& ((abs.(xi) .>= b)))
  indHigh = findall(abs.(xi) .>= c)
  xi[indMid] .= q1 .* tanh.(q2.*(c.-abs.(xi[indMid]))) .* abs.(xi[indMid])./xi[indMid]
  xi[indHigh] .= 0
  xi .= (xi.*scaleX .+ locX)
  return(wrapX=xi,locX=locX,scaleX=scaleX,indMid=indMid,indHigh=indHigh)
end

function wrap(Y, b=1.5, c=4, q1=1.540793, q2=0.8622731, locest="median", scalest=mad)
  """
  # Univariate warapping transformation
  # (see: https://rdrr.io/cran/cellWise/src/R/Wrap.R)
  args:
    y: input vector
  returns:
    xi: output vector with wrapped data
  note:
    b = 1.5 and c = 4 to achieve a BdP of approx 15% for normal data
  """
  p = size(Y)[2]
  xi = Array{Vector{Float64}}(undef, p, 1) 
  locX = Array{Float64}(undef, p, 1) 
  scaleX = Array{Float64}(undef, p, 1) 
  indMid = Array{Vector{Int64}}(undef, p, 1) 
  indHigh = Array{Vector{Int64}}(undef, p, 1) 
  for j=1:p
    (xi[j],locX[j],scaleX[j],indMid[j],indHigh[j]) = wrap_univ(Y[:,j], b, c, q1, q2, locest, scalest)
  end

  return(wrapX=xi,locX=locX,scaleX=scaleX,indMid=indMid,indHigh=indHigh)
end
