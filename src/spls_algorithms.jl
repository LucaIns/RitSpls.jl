@doc """
`spls_algorithms` Univariate Sparse Partial Least squares Algorithms

Two algorithms provided: SNIPLS and shelland.

Can be called directly or though the scikit-learn API.

Written by Sven Serneels.

""" ->

function _fit_snipls(n_components,eta,n,p,Xs,ys,nys,Xnames,X_name,wh=[],
    all_components=false,Xsd=nothing,ysd=nothing,distance=false,verbose=false)

    T = zeros(n,n_components)
    W = zeros(p,n_components)
    P = zeros(p,n_components)
    C = zeros(n_components,1)
    Xev = zeros(n_components,1)
    yev = zeros(n_components,1)
    b = zeros(p,1)
    if all_components
        B = zeros(p,n_components)
    else
        B = nothing
    end
    oldgoodies = []
    Xi = Xs #deepcopy ?
    yi = ys

    if length(wh) == 0
        if distance
            wh = dcd(reshape(Xs'*ys,(p,1)))
            wh = eigen(wh*wh')
            wh = wh.vectors[:,end]
        end
        wh = Xs'*ys
    end

    for i in 1:n_components
        if i > 1
            if distance
                # Xid = dcd(Xi)
                # yid = dcd(reshape(yi,(n,1)))
                # wh = eigen(Xid * yid).vectors[:,end]
                wh = Xi' * yi
                wh = dcd(reshape(wh,(p,1)))
                wh = eigen(wh * wh').vectors[:,end]
            else
                wh =  Xi' * yi
            end
        end
        (wh, goodies) =  _find_sparse(wh,eta)
        global goodies = union(oldgoodies,goodies)
        oldgoodies = goodies
        if length(goodies)==0
            print("No variables retained at" * string(i) * "latent variables" *
                  "and lambda = " * string(eta) * ", try lower lambda")
            colret = []
            break
        end
        elimvars = setdiff(1:p,goodies)
        wh[elimvars] .= 0
        th = Xi * wh
        nth2 = sum(th.^2)
        ch = (yi'*th)
        ph = (Xi' * Xi * wh)
        yi -= th*ch
        Xi -= th * ph'/ nth2
        ph[elimvars] .= 0
        W[:,i] = wh
        P[:,i] = ph / nth2
        C[i] = ch / nth2
        T[:,i] = th
        Xev[i] = 100 - sum((Xs - T[:,1:i]*P[:,1:i]').^2) / sum(Xs.^2)*100
        yev[i] = 100 - sum((ys - T[:,1:i]*C[1:i]).^2) / nys *100
        if Xnames
            global colret = X_name[setdiff((1:p),elimvars)]
        else
            global colret = goodies
        end
        if verbose
            print("Variables retained for " * string(i) *
                    " latent variable(s):" *  "\n" * string(colret) * ".\n")
        end
        if (all_components & (i < n_components))
            if length(goodies) > 0
                if i==1
                    b = (wh * wh' * Xs' * ys)./nth2
                else
                    b = W[:,1:i] * inv(W[:,1:i]'*Xs'*Xs*W[:,1:i]) * W[:,1:i]' * Xs' * ys
                end
                B[:,i] = b
            else
                b = zeros(p,1)
            end
        end
    end

    if length(goodies) > 0

        b = W * inv(W'*Xs'*Xs*W) * W' * Xs' * ys
        if all_components
            B[:,n_components] = b
        end
        R = W * inv(P'*W)
        all0 = findall(P.==0)
        R[all0] .= 0

    else
        b = zeros(p,1)
        R = b
        T = zeros(n,n_components)
    end

    return((W,P,R,C,T,b,Xev,yev,colret,B))

end

function _fit_shelland(n_components,eta,n,p,Xs,ys,nys,Xnames,X_name,s=[],S=[],
        return_snipls_entities=true,all_components=false,verbose=false)

    W = zeros(p,n_components)
    Ã = zeros(p,n_components)
    T = zeros(n,n_components)
    if return_snipls_entities
        P = zeros(p,n_components)
        R = zeros(p,n_components)
        C = zeros(n_components,1)
    end
    if all_components
        B = zeros(p,n_components)
    else
        B = nothing
    end
    Iₚ = diagm(ones(p))
    Xev = zeros(n_components,1)
    yev = zeros(n_components,1)
    b = zeros(p,1)
    Hₚ = zeros(p,p)
    oldgoodies = []

    if length(S)==0
        S = Xs'*Xs
    end
    if length(s) == 0
        s = Xs'*ys
    end
    ah = deepcopy(s)

    for i in 1:n_components
        if i > 1
            ah = (Iₚ - S*Hₚ)*s
        end
        (ah, goodies) =  _find_sparse(ah,eta)
        global goodies = union(oldgoodies,goodies)
        oldgoodies = goodies
        if length(goodies)==0
            print("No variables retained at" * string(i) * "latent variables" *
                  "and lambda = " * string(eta) * ", try lower lambda")
            colret = []
            break
        end
        elimvars = setdiff(1:p,goodies)
        ah[elimvars] .= 0
        if i > 1
            ãh = (Iₚ - Hₚ*S)*ah
        else
            ãh = ah
        end
        th = Xs*ãh
        nth2 = sum(th.^2)
        Hₚ += (ãh*ãh')/nth2
        W[:,i] = ah
        Ã[:,i] = ãh
        T[:,i] = th
        if return_snipls_entities
            ph = S*ãh / nth2
            ph[elimvars] .= 0
            ch = (ys'*th) / nth2
            P[:,i] = ph
            C[i] = ch
            Xev[i] = 100 - sum((Xs - T[:,1:i]*P[:,1:i]').^2) / sum(Xs.^2)*100
            yev[i] = 100 - sum((ys - T[:,1:i]*C[1:i]).^2) / nys *100
        end
        if Xnames
            global colret = X_name[setdiff((1:p),elimvars)]
        else
            global colret = goodies
        end
        if verbose
            print("Variables retained for " * string(i) *
                    " latent variable(s):" *  "\n" * string(colret) * ".\n")
        end
        b = Hₚ * s
        if all_components
            B[:,i] = b
        end
    end



    if length(goodies) > 0
        if return_snipls_entities
            R = Ã * inv(P'*Ã)
        end
    else
        b = zeros(p,1)
        R = b
    end

    if return_snipls_entities
        return((W,P,R,C,T,b,Xev,yev,colret,Ã,Hₚ,B))
    else
        return((W,Ã,T,Hₚ,b,colret,B))
    end

end #_fit_shelland
