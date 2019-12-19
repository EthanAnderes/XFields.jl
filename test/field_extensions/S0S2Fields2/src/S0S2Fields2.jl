
#%% An extension of XFields to S0,S2 spin fields in Rᵈ where d=2
#%% which defines the fields as slices of an Array{Float64,3}
#%% ==============================================================

module S0S2Fields2

using XFields
import LinearAlgebra: dot
import Base: getindex, promote_rule, convert

export S0S2map, S0S2fourier, S0S2field

const module_dir  = joinpath(@__DIR__, "..") |> normpath
const F64   = Float64
const CF64  = Complex{Float64}
const rFT2  = rFourierTransform{nᵢ,pᵢ,2} where {nᵢ,pᵢ} 
const AA{d} = AbstractArray{T,d} where T


#%% Types S0S2map{FT,d} and S0S2fourier{FT,d} where FT<:rFT2
#%% ---------------------------------------------------------------


struct S0S2map{FT<:rFourierTransform} <: XField{FT}
    IQU::Array{F64,3}
    S0S2map{FT}(x::AA{3}) where {FT<:rFT2} = new{FT}(x)
    S0S2map{FT}(x::AA{2}) where {FT<:rFT2} = new{FT}(cat(x,x,x,dims=3))
    S0S2map{FT}(x1::AA{2}, x2::AA{2}, x3::AA{2}) where {FT<:rFT2} = new{FT}(cat(x1,x2,x3,dims=3))
    S0S2map{FT}() where {FT<:rFT2}          = new{FT}(zeros(F64, Grid(FT).nxi...,3))
    function S0S2map{FT}(n1::Number, n2::Number, n3::Number) where {FT<:rFT2} 
        mn1 = fill(F64(n1), Grid(FT).nxi...)
        mn2 = fill(F64(n2), Grid(FT).nxi...)
        mn3 = fill(F64(n3), Grid(FT).nxi...)
        return new{FT}(cat(mn1,mn2,mn3,dims=3))
    end
end

struct S0S2fourier{FT<:rFourierTransform} <: XField{FT}
    IEB::Array{CF64,3}
    S0S2fourier{FT}(x::AA{3}) where {FT<:rFT2} = new{FT}(x)
    S0S2fourier{FT}(x::AA{2}) where {FT<:rFT2} = new{FT}(cat(x,x,x,dims=3))
    S0S2fourier{FT}(x1::AA{2}, x2::AA{2}, x3::AA{2}) where {FT<:rFT2} = new{FT}(cat(x1,x2,x3,dims=3))
    S0S2fourier{FT}() where {FT<:rFT2}          = new{FT}(zeros(CF64, Grid(FT).nki...,3))
    function S0S2fourier{FT}(n1::Number, n2::Number, n3::Number) where {FT<:rFT2} 
        mn1 = fill(CF64(n1), Grid(FT).nki...)
        mn2 = fill(CF64(n2), Grid(FT).nki...)
        mn3 = fill(CF64(n3), Grid(FT).nki...)
        return new{FT}(cat(mn1,mn2,mn3,dims=3))
    end
end


#%% S0S2field union
#%% ---------------------------------------------------------------

const S0S2field{FT} = Union{S0S2fourier{FT}, S0S2map{FT}}


#%% convert
#%% ---------------------------------------------------------------

@generated function sin2ϕ_cos2ϕ(::Type{S0S2fourier{FT}}) where {nᵢ,pᵢ,FT<:rFourierTransform{nᵢ,pᵢ}}
	l1, l2 = frequencies(FT)
	ϕl     = atan.(l2,l1)
    sin2ϕ, cos2ϕ = sin.(2 .* ϕl), cos.(2 .* ϕl)
    if iseven(nᵢ[2]) # force the real hermitian symmitry for sin2ϕl
        sin2ϕ[1, end:-1:(nᵢ[2]÷2+2)]   .= sin2ϕ[1, 2:nᵢ[2]÷2]
        sin2ϕ[end, end:-1:(nᵢ[2]÷2+2)] .= sin2ϕ[end, 2:nᵢ[2]÷2]
    end
	return :( ($sin2ϕ, $cos2ϕ) )
end

function convert(::Type{S0S2map{FT}}, f::S0S2fourier{FT}) where {FT<:rFT2} 
	sin2ϕ, cos2ϕ = sin2ϕ_cos2ϕ(S0S2fourier{FT})
    IQU_l = deepcopy(f.IEB)
    @inbounds IQU_l[:,:,2] .= f.IEB[:,:,2] .* cos2ϕ .- f.IEB[:,:,3] .* sin2ϕ
    @inbounds IQU_l[:,:,3] .= f.IEB[:,:,2] .* sin2ϕ .+ f.IEB[:,:,3] .* cos2ϕ
	return S0S2map{FT}(plan(FT,XFields.LastDimSize{3}) \ IQU_l)
end
function convert(::Type{S0S2fourier{FT}}, f::S0S2map{FT}) where {FT<:rFT2} 
	sin2ϕ, cos2ϕ = sin2ϕ_cos2ϕ(S0S2fourier{FT})
    IQU_l = plan(FT,XFields.LastDimSize{3})*f.IQU  
	IEB_l = deepcopy(IQU_l)
    @inbounds IEB_l[:,:,2] .=    IQU_l[:,:,2] .* cos2ϕ .+ IQU_l[:,:,3] .* sin2ϕ
    @inbounds IEB_l[:,:,3] .= .- IQU_l[:,:,2] .* sin2ϕ .+ IQU_l[:,:,3] .* cos2ϕ
	return S0S2fourier{FT}(IEB_l)
end

# make constructors fall back to convert
S0S2fourier{FT}(f::S0S2field{FT}) where {FT<:rFT2} = convert(S0S2fourier{FT}, f)
S0S2map{FT}(f::S0S2field{FT}) where {FT<:rFT2}     = convert(S0S2map{FT}, f)


#%% promotion
#%% ---------------------------------------------------------------

promote_rule(::Type{S0S2map{FT}}, ::Type{S0S2fourier{FT}}) where {FT<:rFT2} = S0S2fourier{FT}


#%% methods
#%% ---------------------------------------------------------------

(*)(::Type{FT}, f::S0S2fourier{FT}) where {FT<:rFT2} = f
(*)(::Type{FT}, f::S0S2map{FT})     where {FT<:rFT2} = convert(S0S2fourier{FT}, f)
(\)(::Type{FT}, f::S0S2fourier{FT}) where {FT<:rFT2} = convert(S0S2map{FT}, f)
(\)(::Type{FT}, f::S0S2map{FT})     where {FT<:rFT2} = f

dot(f::S0S2field{FT}, g::S0S2field{FT}) where FT = dot(f[:], g[:]) * Grid(FT).Ωx

getindex(f::S0S2field{FT}, ::typeof(!)) where FT = S0S2fourier{FT}(f).IEB
getindex(f::S0S2field{FT}, ::Colon)     where FT = S0S2map{FT}(f).IQU
function getindex(f::S0S2field{FT}, sym::Symbol) where FT
    (sym == :Ix) ?      S0S2map{FT}(f).IQU[:,:,1] :
    (sym == :Il) ?      S0S2fourier{FT}(f).IEB[:,:,1] :
    (sym == :Ex) ? FT \ S0S2fourier{FT}(f).IEB[:,:,2] :
    (sym == :El) ?      S0S2fourier{FT}(f).IEB[:,:,2] :
    (sym == :Bx) ? FT \ S0S2fourier{FT}(f).IEB[:,:,3] :
    (sym == :Bl) ?      S0S2fourier{FT}(f).IEB[:,:,3] :
    (sym == :Qx) ?      S0S2map{FT}(f).IQU[:,:,2] :
    (sym == :Ql) ? FT * S0S2map{FT}(f).IQU[:,:,2] :
    (sym == :Ux) ?      S0S2map{FT}(f).IQU[:,:,3] :
    (sym == :Ul) ? FT * S0S2map{FT}(f).IQU[:,:,3] :
    error("index is not defined")
end



end # module