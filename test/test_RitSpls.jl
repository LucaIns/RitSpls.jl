module TestRitSpls

    using RitSpls
    using CSV
    using ScikitLearn
    using ScikitLearnBase
    using DataFrames
    using Test

    import ScikitLearn.GridSearch:GridSearchCV


    close_enough(x0, x1, tol=1e-14) = abs(x0 - x1) <= tol ? true : false

    Xf = CSV.read("../data/Xfearncal.csv", DataFrame, header=0)
    yf = CSV.read("../data/Yfearncal.csv", DataFrame, header=0)
    n,p = size(Xf)

    gsspp_spls_reg = RitSpls.GSSPP();
    gsspp_spls_sol = ScikitLearn.fit_transform!(gsspp_spls_reg,Xf)
    Xpp = gsspp_spls_sol

    ywrap = RitSpls.wrap(yf) 
    ypp = ywrap.wrapX[1]
    
    snipreg = RitSpls.SPLS();
    RitSpls.set_params_dict!(snipreg, Dict(:fit_algorithm=>"snipls",
        :verbose => false)); 
    gridsearch = GridSearchCV(snipreg, cv=2, 
        Dict(:eta => 0.5, :n_components => 3));
    solfit = ScikitLearn.fit!(gridsearch,Xpp,ypp);
    fit!(solfit,Xpp,ypp)

    @testset "regression coeffs and prediction" begin
        @test close_enough(length(solfit.best_estimator_.coef_),p)
        gsspp_spls_sol2 = ScikitLearn.fit_transform!(gsspp_spls_reg,Xf)
        yp = ScikitLearn.predict(solfit.best_estimator_, gsspp_spls_sol2)

        @test all(map(close_enough, yp, solfit.best_estimator_.fitted_))
    end
end