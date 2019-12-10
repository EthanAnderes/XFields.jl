
#%% An extension of XFields to S0,S2 spin fields in Rᵈ where d=2
#%% which defines the fields as separate in the type
#%% ==============================================================

module S02Fields1

using XFields
import LinearAlgebra: dot
import Base: getindex, promote_rule, convert

export S02map, S02fourier, S02field

const module_dir  = joinpath(@__DIR__, "..") |> normpath
const F64   = Float64
const CF64  = Complex{Float64}



#%% Types S02map{FT} and S02fourier{FT} where FT<:FourierTransform 
#%% ---------------------------------------------------------------

rFT2  = rFourierTransform{nᵢ,pᵢ,2} where {nᵢ,pᵢ} 

struct S02map{FT<:rFourierTransform} <: XField{FT}
    I::Array{F64,2}
    Q::Array{F64,2}
    U::Array{F64,2}
    S02map{FT}(x) where {FT<:rFT2} = new{FT}(copy(x),copy(x),copy(x))
    S02map{FT}(x::Array{F64,3}) where {FT<:rFT2} = new{FT}(x[:,:,1],x[:,:,2],x[:,:,3])
    S02map{FT}(x1::T, x2::T, x3::T) where {T<:Array{F64,2},FT<:rFT2} = new{FT}(x1,x2,x3)
    function S02map{FT}(x1::AbstractArray{T,2}, x2::AbstractArray{T,2}, x3::AbstractArray{T,2}) where {T,FT<:rFT2} 
        return new{FT}(F64.(x1), F64.(x2), F64.(x3))
    end
    function S02map{FT}() where {FT<:rFT2} 
        return new{FT}(
            zeros(F64, Grid(FT).nxi),
            zeros(F64, Grid(FT).nxi),
            zeros(F64, Grid(FT).nxi),
        )
    end
end

struct S02fourier{FT<:rFourierTransform} <: XField{FT}
    I::Array{CF64,2}
    E::Array{CF64,2}
    B::Array{CF64,2}
    S02fourier{FT}(k) where {FT<:rFT2} = new{FT}(copy(k),copy(k),copy(k))
    S02fourier{FT}(k::Array{CF64,3}) where {FT<:rFT2} = new{FT}(k[:,:,1],k[:,:,2],k[:,:,3])
    S02fourier{FT}(k1::T, k2::T, k3::T) where {T<:Array{CF64,2},FT<:rFT2} = new{FT}(k1, k2, k3)
    function S02fourier{FT}(k1::AbstractArray{T,2},k2::AbstractArray{T,2},k3::AbstractArray{T,2}) where {T,FT<:rFT2}
        return new{FT}(CF64.(k1), CF64.(k2), CF64.(k3))
    end
    function S02fourier{FT}() where {FT<:rFT2} 
        return new{FT}(
            zeros(CF64, Grid(FT).nki),
            zeros(CF64, Grid(FT).nki),
            zeros(CF64, Grid(FT).nki),
        )
    end
end


#%% S02field union
#%% ---------------------------------------------------------------

const S02field{FT} = Union{S02fourier{FT}, S02map{FT}}



#%% convert
#%% ---------------------------------------------------------------

@generated function sin2ϕ_cos2ϕ(::Type{S02fourier{FT}}) where {nᵢ, pᵢ,FT<:rFourierTransform{nᵢ,pᵢ,2}}
	l1, l2 = frequencies(FT)
	ϕl     = atan.(l2,l1)
    sin2ϕ, cos2ϕ = sin.(2 .* ϕl), cos.(2 .* ϕl)
    if iseven(nᵢ[2]) # force the real hermitian symmitry for sin2ϕl
        sin2ϕ[1, end:-1:(nᵢ[2]÷2+2)]   .= sin2ϕ[1, 2:nᵢ[2]÷2]
        sin2ϕ[end, end:-1:(nᵢ[2]÷2+2)] .= sin2ϕ[end, 2:nᵢ[2]÷2]
    end
	return :( ($sin2ϕ, $cos2ϕ) )
end

function convert(::Type{S02map{FT}}, f::S02fourier{FT}) where {FT<:rFT2} 
	sin2ϕ, cos2ϕ = sin2ϕ_cos2ϕ(S02fourier{FT})
    # Ql = similar(f.E)
    # Ul = similar(f.E)
    # @inbounds @simd for i in eachindex(f.I)
    #     Ql[i] =   f.E[i] * cos2ϕ[i] - f.B[i] * sin2ϕ[i]
    #     Ul[i] =   f.E[i] * sin2ϕ[i] + f.B[i] * cos2ϕ[i]
    # end    
    Ql = @inbounds @.  f.E * cos2ϕ - f.B * sin2ϕ
    Ul = @inbounds @.  f.E * sin2ϕ + f.B * cos2ϕ
	return S02map{FT}(FT \ f.I, FT \ Ql, FT \ Ul)
end
function convert(::Type{S02fourier{FT}}, f::S02map{FT}) where {FT<:rFT2} 
	sin2ϕ, cos2ϕ = sin2ϕ_cos2ϕ(S02fourier{FT})
	Ql = FT*f.Q 
	Ul = FT*f.U
	# El = similar(Ql)
    # Bl = similar(Ul)
    # @inbounds @simd for i in eachindex(f.I)
    #     El[i] =    Ql[i] * cos2ϕ[i] + Ul[i] * sin2ϕ[i]
    #     Bl[i] =  - Ql[i] * sin2ϕ[i] + Ul[i] * cos2ϕ[i]
    # end
    El = @inbounds @.   Ql * cos2ϕ + Ul * sin2ϕ
    Bl = @inbounds @. - Ql * sin2ϕ + Ul * cos2ϕ
	return S02fourier{FT}(FT * f.I, El, Bl)
end

# make constructors fall back to convert
S02fourier{FT}(f::S02field{FT}) where {FT<:rFT2} = convert(S02fourier{FT}, f)
S02map{FT}(f::S02field{FT}) where {FT<:rFT2}     = convert(S02map{FT}, f)



#%% promotion
#%% ---------------------------------------------------------------

promote_rule(::Type{S02map{FT}}, ::Type{S02fourier{FT}}) where {FT<:rFT2} = S02fourier{FT}


#%% methods
#%% ---------------------------------------------------------------

dot(f::S02field{FT}, g::S02field{FT}) where FT = dot(f[:], g[:]) * Grid(FT).Ωx

getindex(f::S02field{FT}, ::typeof(!)) where FT = cat(fielddata(S02fourier{FT}(f))...,dims=3)
getindex(f::S02field{FT}, ::Colon)     where FT = cat(fielddata(S02map{FT}(f))...,dims=3)
function getindex(f::S02field{FT}, sym::Symbol) where FT
    (sym == :Ix) ? S02map{FT}(f).I :
    (sym == :Il) ? S02fourier{FT}(f).I :
    (sym == :Ex) ? FT \ S02fourier{FT}(f).E :
    (sym == :El) ? S02fourier{FT}(f).E :
    (sym == :Bx) ? FT \ S02fourier{FT}(f).B :
    (sym == :Bl) ? S02fourier{FT}(f).B :
    (sym == :Qx) ? S02map{FT}(f).Q :
    (sym == :Ql) ? FT * S02map{FT}(f).Q :
    (sym == :Ux) ? S02map{FT}(f).U :
    (sym == :Ul) ? FT * S02map{FT}(f).U :
    error("index is not defined")
end



end # module