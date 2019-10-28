## ============== Smap{FT} and Sfourier{FT} where FT<:FourierTransform ====================

F64   = Float64
CF64  = Complex{Float64}
rFT{d} = rFourierTransform{nᵢ,pᵢ,d} where {nᵢ,pᵢ} 

struct Smap{FT<:rFourierTransform,d} <: XField{FT}
    x::Array{F64,d}
end

struct Sfourier{FT<:rFourierTransform,d} <: XField{FT}
    k::Array{CF64,d}
end

# outer-constructors
Smap{FT}(x::Array{F64,d}) where {d,FT<:rFT{d}} = Smap{FT,d}(x)
Smap{FT}(x::AbstractArray{T,d}) where {T,d,FT<:rFT{d}} = Smap{FT,d}(F64.(x))
Smap{FT}() where {d,FT<:rFT{d}} = Smap{FT,d}(zeros(F64, Grid(FT).nxi))

Sfourier{FT}(k::Array{CF64,d}) where {d,FT<:rFT{d}} = Sfourier{FT,d}(k)
Sfourier{FT}(k::AbstractArray{T,d}) where {T,d,FT<:rFT{d}} = Sfourier{FT,d}(CF64.(k))
Sfourier{FT}() where {d,FT<:rFT{d}} = Sfourier{FT,d}(zeros(CF64, Grid(FT).nki))

#  union type
const Sfield{FT,d} = Union{Sfourier{FT,d}, Smap{FT,d}}

#  convert and promote
promote_rule(::Type{Smap{FT,d}}, ::Type{Sfourier{FT,d}}) where {d,FT<:rFT{d}} = Sfourier{FT,d}
convert(::Type{Smap{FT,d}}, f::Sfourier{FT,d}) where {d,FT<:rFT{d}} = Smap{FT,d}(FT \ f.k)
convert(::Type{Sfourier{FT,d}}, f::Smap{FT,d}) where {d,FT<:rFT{d}} = Sfourier{FT,d}(FT * f.x)
Sfourier{FT,d}(f::Sfield{FT,d}) where {d,FT<:rFT{d}} = convert(Sfourier{FT,d}, f)
Sfourier{FT}(f::Sfield{FT,d}) where {d,FT<:rFT{d}}   = convert(Sfourier{FT,d}, f)
Smap{FT,d}(f::Sfield{FT,d}) where {d,FT<:rFT{d}}     = convert(Smap{FT,d}, f)
Smap{FT}(f::Sfield{FT,d}) where {d,FT<:rFT{d}}       = convert(Smap{FT,d}, f)

#  getindex
function getindex(f::Sfield{FT}, sym::Symbol) where FT
    (sym == :k)  ? Sfourier{FT}(f).k :
    (sym == :x)  ? Smap{FT}(f).x :
    error("index is not defined")
end



# TODO: 
# ## dot for Pix <: Flat
# dot(a::X, b::X) where X<:XField{P} where P<:Flat = _dot(a, b, is_map(X))

# ## dot(map, map)
# function _dot(a::X, b::X, ::Type{IsMap{true}}) where X<:XField{P} where P<:Flat
#     FT = harmonic_transform(X)
#     return sum(map(_realdot, fielddata(a),fielddata(b))) * FT.Ωx
# end

# # dot(fourier, fourier)
# function _dot(a::X, b::X, ::Type{IsMap{false}}) where X<:XField{P} where P<:Flat
#     FT = harmonic_transform(X)
#     sum(map(_complexdot, fielddata(a),fielddata(b))) * FT.Ωk
# end

# # these work better for ArrayFire
# _realdot(a,b) = sum(a.*b)

# function _complexdot(a,b)
#     n,m = size(a)
#     @assert size(a)==size(b) && n==m÷2+1
#     # ----- this kills the repeated frequencies
#     multip_ri = fill(true, n, m)   # both real and imaginary
#     multip_ri[1,(n+1):m] .= false  # repeats
#     if iseven(m)
#         multip_ri[end,(n+1):m] .= false  # repeats
#     end
#     # ----- Now to a direct (x 2) dot product of the real and imag parts
#     ra, ia = real(a), imag(a)
#     rb, ib = real(b), imag(b)
#     rtn  = 2*dot(ra, multip_ri .* rb) + 2*dot(ia, multip_ri .* ib)
#     # ----- but we don't mult by 2 for the real terms... so we take away 1
#     rtn -= ra[1,1]*rb[1,1]  
#     rtn -= ra[1,n]*rb[1,n] 
#     if iseven(m) 
#         rtn -= ra[n,1]*rb[n,1] 
#         rtn -= ra[n,n]*rb[n,n] 
#     end 
#     return rtn
# end
