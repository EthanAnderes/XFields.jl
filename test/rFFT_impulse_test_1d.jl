# impulse test for reconstructing 

##============
# modules

# using XFields
# using FFTW 
using LinearAlgebra
# using PyPlot
# using Statistics


##============ 
# impulse method definition
function ei(i,n)
    eiv = fill(false,n)
    eiv[i] = true
    return eiv
end

## =======================================
# Construct fourier transform operator FT, and the corresponding matrices

dim, nᵢ, pᵢ = 1, (128,), (1.25,)
rFT, rFTp, rG  = rFFT{nᵢ,pᵢ,dim} |> x->(x,plan(x),Grid(x))
cFT, cFTp, cG  = cFFT{nᵢ,pᵢ,dim} |> x->(x,plan(x),Grid(x))
ntot = prod(nᵢ)

rFTpᴴ  = adjoint(rFTp) 
rFTpᵀ  = transpose(rFTp)
rFTp⁻¹ = inv(rFTp)
rFTp⁻ᴴ = inv(rFTpᴴ)
rFTp⁻ᵀ = inv(rFTpᵀ)

cFTpᴴ  = adjoint(cFTp)
cFTpᵀ  = transpose(cFTp)
cFTp⁻¹ = inv(cFTp)
cFTp⁻ᴴ = inv(cFTpᴴ)
cFTp⁻ᵀ = inv(cFTpᵀ)

cW   = [exp(- im * 2π * (i-1)*(j-1) / ntot) for i=1:ntot, j=1:ntot]
cF   = cFTp.scale .* cW
cFᴴ  = adjoint(cF)
cFᵀ  = transpose(cF)
cF⁻¹ = inv(cF)
cF⁻ᴴ = inv(cFᴴ)
cF⁻ᵀ = inv(cFᵀ)

## =======================================
# Covariance model 

##
# First covariance modeling in pixel space
Σ1   = Symmetric([exp(-abs(i-j)) for i=1:ntot, j=1:ntot]) 
# Σpre  = [exp(-abs(i-j)) for i=1:d, j=1:d]
# for i = 2:size(Σpre,2)
# 	Σpre[:,i] .= circshift(Σpre[:,i-1], 1)
# end
# Σ = Symmetric((Σpre .+ Σpre') ./ 2) # --> zero C on unique frequencies

# covariance modeling in frequency space
cΓ1 = cF * Σ1 * cFᴴ 
cC1 = cF * Σ1 * cFᵀ
rΓ1 = cΓ1[1:prod(rG.nki), 1:prod(rG.nki)]
rC1 = cC1[1:prod(rG.nki), 1:prod(rG.nki)]

##
# Second covariance model in pixel space
kw = wavenumber(cFT)
Ck = 1 ./ (1 .+ abs.(kw) ./ 10)
# wnx = randn(nᵢ...) ./ sqrt(rG.Ωx) # white noise
# wnk = rFTp * wnx
# fk = sqrt.(Ck) .* wnk
δk₀ = 1/cG.Ωk
Σ2 = cF⁻¹ * Diagonal(Ck * δk₀) * cF⁻ᴴ
cΓ2 = cF * Σ2 * cFᴴ 
cC2 = cF * Σ2 * cFᵀ
rΓ2 = cΓ2[1:prod(rG.nki), 1:prod(rG.nki)]
rC2 = cC2[1:prod(rG.nki), 1:prod(rG.nki)]



## =======================================
# test we can recover rΓ and rC using full complex impulse responses

rΓ1test, rC1test, rΓ2test, rC2test = let cn=prod(cG.nki), rn=prod(rG.nki), Σ1=Σ1, Σ2=Σ2
    rΓ1test  = fill(0.0im, rn, rn)
    rC1test  = fill(0.0im, rn, rn)
    rΓ2test  = fill(0.0im, rn, rn)
    rC2test  = fill(0.0im, rn, rn)
    for i=1:rn
        eit = ei(i,cn)
        rΓ1test[:,i] = (cFTp * (Σ1 * (cFTpᴴ * eit)))[1:rn]
        rC1test[:,i] = (cFTp * (Σ1 * (cFTpᵀ * eit)))[1:rn]
        rΓ2test[:,i] = (cFTp * (Σ2 * (cFTpᴴ * eit)))[1:rn]
        rC2test[:,i] = (cFTp * (Σ2 * (cFTpᵀ * eit)))[1:rn]
    end
    rΓ1test, rC1test, rΓ2test, rC2test
end

#=
rΓ1 .|> abs |> matshow 
rC1 .|> abs |> matshow 
rΓ2 .|> abs |> matshow 
rC2 .|> abs |> matshow 
=#



for (A,B) ∈ zip((rΓ1test, rC1test, rΓ2test, rC2test), (rΓ1, rC1, rΓ2, rC2))
    @test sum(abs2, A .- B) <= eps(Float64)
end




## =======================================
# recover rΓ, rC but this time using real FFT impulses

EZXᵀ1test, EZYᵀ1test, EZXᵀ2test, EZYᵀ2test = let cn=prod(cG.nki), rn=prod(rG.nki), Σ1=real.(Σ1), Σ2=real.(Σ2)
    EZXᵀ1test  = fill(0.0im, rn, rn)
    EZYᵀ1test  = fill(0.0im, rn, rn)
    EZXᵀ2test  = fill(0.0im, rn, rn)
    EZYᵀ2test  = fill(0.0im, rn, rn)
    for i=1:rn
    	if i==1 || i==rn
    	 	φ  = ei(i, rn)
    	 	iφ = 0 .* φ
    	else
    	     φ = ei(i, rn) ./ 2
    		iφ = im .* ei(i, rn) ./ 2
 		end
        EZXᵀ1test[:,i] = rFTp * (Σ1 * (rFTpᴴ * φ))
        EZYᵀ1test[:,i] = rFTp * (Σ1 * (rFTpᴴ * iφ))
        EZXᵀ2test[:,i] = rFTp * (Σ2 * (rFTpᴴ * φ))
        EZYᵀ2test[:,i] = rFTp * (Σ2 * (rFTpᴴ * iφ))
    end
    EZXᵀ1test, EZYᵀ1test, EZXᵀ2test, EZYᵀ2test
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

#=
rΓ1test2 .- rΓ1 .|> abs |> matshow 
rC1test2 .- rC1 .|> abs |> matshow 
rΓ2test2 .- rΓ2 .|> abs |> matshow 
rC2test2 .- rC2 .|> abs |> matshow 
=#

for (A,B) ∈ zip((rΓ1test2, rC1test2, rΓ2test2, rC2test2), (rΓ1, rC1, rΓ2, rC2))
    @test sum(abs2, A .- B) <= eps(Float64)
end
