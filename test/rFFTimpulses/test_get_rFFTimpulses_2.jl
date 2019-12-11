
# using XFields
# using Test
using LinearAlgebra


#%%
#%% =========================================

Σₒ, Γₒ, Cₒ, rFT, rFTplan, gr, fk_sim = let

    # 1-d vector of length N
    N = 12

    # set up the FFT matrices
    rFT = rFFT(nᵢ=(N,),pᵢ=(1.0,))
    gr  = Grid(rFT)    
    𝒲 = [exp(-im * 2π * (k-1)*(n-1) / N) for k=1:N, n=1:N]
    ℱ = (gr.Ωx / sqrt(2π)) .* 𝒲

    # 
    M = fill(0.0,N) |> x-> (x[4:end-1].= 1; x) |> Diagonal

    # cov pixel model before masking
    Σpre   = Symmetric([exp(-abs(i-j)) for i=1:N, j=1:N]) 
    # Σpre  = [exp(-abs(i-j)) for i=1:d, j=1:d]
    # for i = 2:size(Σpre,2)
    # 	Σpre[:,i] .= circshift(Σpre[:,i-1], 1)
    # end
    # Σ = Symmetric((Σpre .+ Σpre') ./ 2) # --> zero C on unique frequencies
    Lpre = cholesky(Σpre).L

    # cov model in pixel space
    Σ = M*Σpre*M

    # 
    fx_sim = M * Lpre * randn(N)
    fk_sim = ℱ * fx_sim |> x -> x[1:N÷2+1] # only the complex unique coeffs

    ## 
    Γ = ℱ * Σ * adjoint(ℱ)   |> x -> x[1:N÷2+1,1:N÷2+1]
    C = ℱ * Σ * transpose(ℱ) |> x -> x[1:N÷2+1,1:N÷2+1]
    #=
    Izu = 2:N÷2
    Γk[Izu, Izu] .|> abs |> matshow; colorbar()
    Ck[Izu, Izu] .|> abs |> matshow; colorbar()
    =#

    Σ, Γ, C, rFT, plan(rFT), gr, fk_sim

end

#%%
#%% =========================================

rΓ, rC, ΣXX, ΣYY, ΣYX, ΣXY = let

    rFFTimpulses, CI, LI, get_dual_ci = get_rFFTimpulses(rFT)
    EZXᵀ = fill(0.0im, length(fk_sim), length(fk_sim))
    EZYᵀ = fill(0.0im, length(fk_sim), length(fk_sim))
    for (li,ci) ∈ zip(LI, CI)
        φ, iφ = rFFTimpulses(ci)
        EZXᵀ[:,li] = rFTplan * (Σₒ * (adjoint(rFTplan) * φ))
        EZYᵀ[:,li] = rFTplan * (Σₒ * (adjoint(rFTplan) * iφ))
    end

    ΣXX = real.(EZXᵀ)
    ΣYY = imag.(EZYᵀ)
    ΣYX = imag.(EZXᵀ)
    ΣXY = real.(EZYᵀ)

    rΓ = complex.(ΣXX .+ ΣYY, ΣYX .- ΣXY)
    rC = complex.(ΣXX .- ΣYY, ΣYX .+ ΣXY)

    rΓ, rC, ΣXX, ΣYY, ΣYX, ΣXY

end


#%%
#%% =========================================
let 
	function logPk1(z; Σxx,Σxy,Σyx,Σyy)
	    # Σ    = [
	    #     Σxx  Σxy
	    #     Σyx  Σyy
	    # ] |> Symmetric
	    Σ    = hvcat((2,2), Σxx, Σxy, Σyx, Σyy) |> Symmetric
	    xy   = vcat(real.(z), imag.(z))
	    rtn  = - dot(xy, Σ \ xy) / 2
	    rtn -= logabsdet(Σ)[1] / 2
	    rtn -= length(z)*log(2)
	    return rtn
	end

	function logPk2(z; Γ, C)
	    # ΓC = [
	    #     Γ         C
	    #     conj.(C)  conj.(Γ)
	    # ] |> Hermitian
	    ΓC   = hvcat((2,2), Γ, C, conj.(C), conj.(Γ)) |> Hermitian
	    zcz = vcat(z, conj.(z))
	    rtn  = real.(- dot(zcz, ΓC \ zcz) / 2)
	    rtn -= logabsdet(ΓC)[1] / 2
	    return rtn
	end

	subInd = 2:5

	val1 = logPk1(fk_sim[subInd]; Σxx = ΣXX[subInd,subInd],
	                    Σxy = ΣXY[subInd,subInd],
	                    Σyx = ΣYX[subInd,subInd],
	                    Σyy = ΣYY[subInd,subInd]
	)
	val2 = logPk2(fk_sim[subInd]; 
	    Γ=rΓ[subInd,subInd], 
	    C=rC[subInd,subInd]
	)

	@test val1 ≈ val2 atol=1e-10


end




# ## check this has the correct std

# function sim1()
# 	dmlw = ℳ * L * randn(N)
# 	dk   = 𝒲 * dmlw
# 	return logPk1(dk[ksub]; Γ=Γksub, C=Cksub)
# end

# function sim2()
# 	dmlw = ℳ * L * randn(N)
# 	dk   = 𝒲 * dmlw
# 	return logPk2(dk[ksub]; Γ=Γksub, C=Cksub)
# end

# logPk1s = [sim1() for i=1:5000]
# logPk2s = [sim2() for i=1:5000]
# std(logPk1s) # should be about sqrt(2*length(ksub)) / sqrt(2)
# std(logPk2s) # should be about sqrt(2*length(ksub)) / sqrt(2)



