using LinearAlgebra

let
	rFTvals = [
		rFFT(nᵢ=(7,7),pᵢ=(1.0,2.0)),
		rFFT(nᵢ=(7,8),pᵢ=(1.0,2.0)),
		rFFT(nᵢ=(8,7),pᵢ=(1.0,2.0)),
		rFFT(nᵢ=(8,8),pᵢ=(1.0,2.0)),
		rFFT(nᵢ=(128,),pᵢ=(2.0,)),  
		rFFT(nᵢ=(127,),pᵢ=(2.0,)),  
	]
	cFTvals = [
		cFFT(nᵢ=(7,7),pᵢ=(1.0,2.0)),
		cFFT(nᵢ=(7,8),pᵢ=(1.0,2.0)),
		cFFT(nᵢ=(8,7),pᵢ=(1.0,2.0)),
		cFFT(nᵢ=(8,8),pᵢ=(1.0,2.0)),
		cFFT(nᵢ=(128,),pᵢ=(2.0,)),  
		cFFT(nᵢ=(127,),pᵢ=(2.0,)),  
	]

	for (rFT,cFT) ∈ zip(rFTvals, cFTvals)
		rFTp, rg = rFT |> x->(plan(x),Grid(x))
		cFTp, cg = cFT |> x->(plan(x),Grid(x))
		
		rFTpᴴ = adjoint(rFTp) 
		cF = zeros(Complex{Float64},prod(rg.nxi),prod(rg.nxi))
		for i = 1:prod(rg.nxi)
		    imls = zeros(Complex{Float64}, prod(rg.nxi))
		    imls[i] = 1
		    cF[:,i] = vec(cFTp * reshape(imls,rg.nxi))
		end
		cFᴴ  = adjoint(cF)
		cFᵀ  = transpose(cF)
		cF⁻¹ = inv(cF)
		cF⁻ᴴ = inv(cFᴴ)

		##
		# Need to check on subset of full fourier matrix
		# These are the tuples (i,j) of the real fft frequencies
		rCi = CartesianIndices(Base.OneTo.(rg.nki))
		# This is the matrix of linear indices into complex fft frequencies
		cLi = LinearIndices(Base.OneTo.(cg.nki)) 
		# This gets the linear indices of real fft frequencies in the full matrix 
		rsub_cLi = cLi[rCi]

		##
		# First covariance modeling in pixel space
		nrm = zeros(prod(rg.nxi), prod(rg.nxi))
		for xi ∈ pixels(rFT)
		    nrm .+= (vec(xi) .- vec(xi)').^2
		end
		nrm = sqrt.(nrm)
		Σ1   = @. exp(-nrm*2)
		M   = Diagonal(vec(1 .+ sin.(pixels(rFT)[1])))
		Σ1  = M * Σ1 * M
		cΓ1 = cF * Σ1 * cFᴴ 
		cC1 = cF * Σ1 * cFᵀ
		rΓ1 = cΓ1[vec(rsub_cLi), vec(rsub_cLi)]
		rC1 = cC1[vec(rsub_cLi), vec(rsub_cLi)]

		##
		# Second covariance model in pixel space
		kw = wavenumber(cFT)
		Ck = 1 ./ (1 .+ abs.(kw) ./ 10)
		# wnx = randn(rg.nxi...) ./ sqrt(rg.Ωx) # white noise
		# wnk = rFTp * wnx
		# fk = sqrt.(Ck) .* wnk
		δk₀ = 1/cg.Ωk
		Σ2  = real.(cF⁻¹ * Diagonal(vec(Ck * δk₀)) * cF⁻ᴴ)
		cΓ2 = cF * Σ2 * cFᴴ 
		cC2 = cF * Σ2 * cFᵀ
		rΓ2 = cΓ2[vec(rsub_cLi), vec(rsub_cLi)]
		rC2 = cC2[vec(rsub_cLi), vec(rsub_cLi)]

		##
		#now test
		rFFTimpulses, CI, LI, get_dual_ci = get_rFFTimpulses(rFT)
		ve = f -> vec(f)
		rex = vf -> reshape(vf, rg.nxi)

		EZXᵀ1test  = fill(0.0im, prod(rg.nki), prod(rg.nki))
		EZYᵀ1test  = fill(0.0im, prod(rg.nki), prod(rg.nki))
		EZXᵀ2test  = fill(0.0im, prod(rg.nki), prod(rg.nki))
		EZYᵀ2test  = fill(0.0im, prod(rg.nki), prod(rg.nki))
		for (li,ci) ∈ zip(LI, CI)
		    φ, iφ = rFFTimpulses(ci)
		    EZXᵀ1test[:,li] = ve(rFTp * rex(Σ1 * ve(rFTpᴴ * φ)))
		    EZYᵀ1test[:,li] = ve(rFTp * rex(Σ1 * ve(rFTpᴴ * iφ)))
		    EZXᵀ2test[:,li] = ve(rFTp * rex(Σ2 * ve(rFTpᴴ * φ)))
		    EZYᵀ2test[:,li] = ve(rFTp * rex(Σ2 * ve(rFTpᴴ * iφ)))
		end

		Σ1XX = real.(EZXᵀ1test)
		Σ1YY = imag.(EZYᵀ1test)
		Σ1YX = imag.(EZXᵀ1test)
		Σ1XY = real.(EZYᵀ1test)
		
		Σ2XX = real.(EZXᵀ2test)
		Σ2YY = imag.(EZYᵀ2test)
		Σ2YX = imag.(EZXᵀ2test)
		Σ2XY = real.(EZYᵀ2test)

		rΓ1test2 = complex.(Σ1XX .+ Σ1YY, Σ1YX .- Σ1XY)
		rC1test2 = complex.(Σ1XX .- Σ1YY, Σ1YX .+ Σ1XY)
		rΓ2test2 = complex.(Σ2XX .+ Σ2YY, Σ2YX .- Σ2XY)
		rC2test2 = complex.(Σ2XX .- Σ2YY, Σ2YX .+ Σ2XY)

		for (A,B) ∈ zip((rΓ1test2, rC1test2, rΓ2test2, rC2test2), (rΓ1, rC1, rΓ2, rC2))
	    	@test sum(abs2, A.- B) ≈ 0.0 atol=1e-15
		end
	end # for
end # let 
