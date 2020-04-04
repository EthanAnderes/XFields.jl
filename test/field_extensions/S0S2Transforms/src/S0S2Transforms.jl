module S0S2Transforms

using  XFields

export S0S2transform 

const module_dir  = joinpath(@__DIR__, "..") |> normpath
const F64   = Float64
const CF64  = Complex{Float64}
const AA{d} = AbstractArray{T,d} where T



#%% Construct a new transform 
#%% ---------------------------------------------------------------

struct S0S2transform{nᵢ,pᵢ} <: rFourierTransform{nᵢ,pᵢ,3}  end
# works on Array{T,3}, XFields.LastDimSize{3}

# easy constructor
function S0S2transform(;nᵢ, pᵢ=nothing, Δxᵢ=nothing) 
    nᵢ,pᵢ,d = XFields._get_npd(;nᵢ=nᵢ, pᵢ=pᵢ, Δxᵢ=Δxᵢ)
    @assert d == 2
    return S0S2transform{nᵢ,pᵢ}
end



#%% Need to teach S0S2transform{nᵢ,pᵢ} how to lmultiply or ldivide by arrays
#%% ---------------------------------------------------------------

function XFields.rFFT(::Type{ST}) where {nᵢ,pᵢ,ST<:S0S2transform{nᵢ,pᵢ}}
    return rFFT{nᵢ,pᵢ,2}
end

@generated function sin2ϕ_cos2ϕ(::Type{ST}) where {nᵢ,pᵢ,ST<:S0S2transform{nᵢ,pᵢ}}
	FT = rFFT(ST)
    l1, l2 = frequencies(FT)
	ϕl     = atan.(l2,l1)
    sin2ϕ, cos2ϕ = sin.(2 .* ϕl), cos.(2 .* ϕl)
    if iseven(nᵢ[2]) # force the real hermitian symmitry for sin2ϕl
        sin2ϕ[1, end:-1:(nᵢ[2]÷2+2)]   .= sin2ϕ[1, 2:nᵢ[2]÷2]
        sin2ϕ[end, end:-1:(nᵢ[2]÷2+2)] .= sin2ϕ[end, 2:nᵢ[2]÷2]
    end
	return :( ($sin2ϕ, $cos2ϕ) )
end

function Base.:*(::Type{ST}, IQU::Array{T,3}) where {nᵢ,pᵢ, T, ST<:S0S2transform{nᵢ,pᵢ}}
	sin2ϕ, cos2ϕ = sin2ϕ_cos2ϕ(ST)
	IQU_l = plan(rFFT(ST), XFields.LastDimSize{3}) * IQU
	IEB_l = deepcopy(IQU_l)
    @inbounds IEB_l[:,:,2] .=    IQU_l[:,:,2] .* cos2ϕ .+ IQU_l[:,:,3] .* sin2ϕ
    @inbounds IEB_l[:,:,3] .= .- IQU_l[:,:,2] .* sin2ϕ .+ IQU_l[:,:,3] .* cos2ϕ
	
	return IEB_l
end

function Base.:\(::Type{ST}, IEB::Array{T,3}) where {nᵢ,pᵢ, T, ST<:S0S2transform{nᵢ,pᵢ}}
	sin2ϕ, cos2ϕ = sin2ϕ_cos2ϕ(ST)
    IQU_l = deepcopy(IEB)
    @inbounds IQU_l[:,:,2] .= IEB[:,:,2] .* cos2ϕ .- IEB[:,:,3] .* sin2ϕ
    @inbounds IQU_l[:,:,3] .= IEB[:,:,2] .* sin2ϕ .+ IEB[:,:,3] .* cos2ϕ

	return plan(rFFT(ST), XFields.LastDimSize{3}) \ IQU_l
end


#%% Grid geometry associated with ST
#%% ---------------------------------------------------------------

XFields.Grid(::Type{ST}) where {nᵢ,pᵢ,ST<:S0S2transform{nᵢ,pᵢ}} = XFields.Grid(rFFT(ST))


#%% Specialized constructors for Rmap{ST} and Sfourier{ST}
#%% ---------------------------------------------------------------

Rmap{ST}(x::AA{2}) where {ST<:S0S2transform} = Rmap{ST,3}(cat(x,x,x,dims=3))
Rmap{ST}(x1::AA{2}, x2::AA{2}, x3::AA{2}) where {ST<:S0S2transform} = Rmap{ST,3}(cat(x1,x2,x3,dims=3))
Rmap{ST}() where {ST<:S0S2transform}  = Rmap{ST,3}(zeros(F64, Grid(ST).nxi...,3))
function Rmap{ST}(n1, n2, n3) where {ST<:S0S2transform} 
    ons = fill(F64(1), Grid(ST).nxi...)
    mn1 = n1 .* ons
    mn2 = n2 .* ons
    mn3 = n3 .* ons
    return Rmap{ST,3}(cat(mn1,mn2,mn3,dims=3))
end

Sfourier{ST}(x::AA{2}) where {ST<:S0S2transform} = Sfourier{ST,3}(cat(x,x,x,dims=3))
Sfourier{ST}(x1::AA{2}, x2::AA{2}, x3::AA{2}) where {ST<:S0S2transform} = Sfourier{ST,3}(cat(x1,x2,x3,dims=3))
Sfourier{ST}() where {ST<:S0S2transform}          = Sfourier{ST,3}(zeros(CF64, Grid(ST).nki...,3))
function Sfourier{ST}(n1, n2, n3) where {ST<:S0S2transform} 
    ons = fill(CF64(1), Grid(ST).nki...)
    mn1 = n1 .* ons
    mn2 = n2 .* ons
    mn3 = n3 .* ons
    return Sfourier{ST,3}(cat(mn1,mn2,mn3,dims=3))
end




#%% add custom getindex. 
#%% ---------------------------------------------------------------

function Base.getindex(f::Sfield{ST}, sym::Symbol) where {nᵢ,pᵢ,ST<:S0S2transform{nᵢ,pᵢ}}
    (sym == :Ix) ?      Rmap{ST}(f).x[:,:,1] :
    (sym == :Il) ?      Sfourier{ST}(f).k[:,:,1] :
    (sym == :Ex) ? rFFT{nᵢ,pᵢ,2} \ Sfourier{ST}(f).k[:,:,2] :
    (sym == :El) ?      Sfourier{ST}(f).k[:,:,2] :
    (sym == :Bx) ? rFFT{nᵢ,pᵢ,2} \ Sfourier{ST}(f).k[:,:,3] :
    (sym == :Bl) ?      Sfourier{ST}(f).k[:,:,3] :
    (sym == :Qx) ?      Rmap{ST}(f).x[:,:,2] :
    (sym == :Ql) ? rFFT{nᵢ,pᵢ,2} * Rmap{ST}(f).x[:,:,2] :
    (sym == :Ux) ?      Rmap{ST}(f).x[:,:,3] :
    (sym == :Ul) ? rFFT{nᵢ,pᵢ,2} * Rmap{ST}(f).x[:,:,3] :
    error("index is not defined")
end






end # end module
