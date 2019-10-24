## ============== Smap{FFT} and Sfourier{FFT} where FFT<:FourierTransform ====================

F64   = Float64
CF64  = Complex{Float64}
FT{d} = FourierTransform{nᵢ,pᵢ,d} where {nᵢ,pᵢ} 

struct Smap{FFT<:FourierTransform,d} <: XField{FFT}
    x::Array{F64,d}
end

struct Sfourier{FFT<:FourierTransform,d} <: XField{FFT}
    k::Array{CF64,d}
end

# outer-constructors
Smap{FFT}(x::Array{F64,d}) where {d,FFT<:FT{d}} = Smap{FFT,d}(x)
Smap{FFT}(x::Array{T,d}) where {T,d,FFT<:FT{d}} = Smap{FFT,d}(F64.(x))
Smap{FFT}() where {d,FFT<:FT{d}} = Smap{FFT,d}(zeros(F64, dims_xi(FFT)))
Sfourier{FFT}(k::Matrix{CF64}) where {d,FFT<:FT{d}} = Sfourier{FFT,d}(k)
Sfourier{FFT}(k::Array{T,d}) where {T,d,FFT<:FT{d}} = Sfourier{FFT,d}(CF64.(k))
Sfourier{FFT}() where {d,FFT<:FT{d}} = Sfourier{FFT,d}(zeros(CF64, dims_ki(FFT)))

#  union type
const Sfield{FFT,d} = Union{Sfourier{FFT,d}, Smap{FFT,d}}

#  convert and promote
promote_rule(::Type{Smap{FFT,d}}, ::Type{Sfourier{FFT,d}}) where {d,FFT<:FT{d}} = Sfourier{FFT,d}
convert(::Type{Smap{FFT,d}}, f::Sfourier{FFT,d}) where {d,FFT<:FT{d}} = Smap{FFT,d}(FFT \ f.k)
convert(::Type{Sfourier{FFT,d}}, f::Smap{FFT,d}) where {d,FFT<:FT{d}} = Sfourier{FFT,d}(FFT * f.x)
Sfourier{FFT,d}(f::Sfield{FFT,d}) where {d,FFT<:FT{d}} = convert(Sfourier{FFT,d}, f)
Sfourier{FFT}(f::Sfield{FFT,d}) where {d,FFT<:FT{d}}   = convert(Sfourier{FFT,d}, f)
Smap{FFT,d}(f::Sfield{FFT,d}) where {d,FFT<:FT{d}}     = convert(Smap{FFT,d}, f)
Smap{FFT}(f::Sfield{FFT,d}) where {d,FFT<:FT{d}}       = convert(Smap{FFT,d}, f)

#  getindex
function getindex(f::Sfield{FFT}, sym::Symbol) where FFT
    (sym == :k)  ? Sfourier{FFT}(f).k :
    (sym == :x)  ? Smap{FFT}(f).x :
    error("index is not defined")
end



# TODO: 
# ## dot for Pix <: Flat
# dot(a::X, b::X) where X<:XField{P} where P<:Flat = _dot(a, b, is_map(X))

# ## dot(map, map)
# function _dot(a::X, b::X, ::Type{IsMap{true}}) where X<:XField{P} where P<:Flat
#     FFT = harmonic_transform(X)
#     return sum(map(_realdot, fielddata(a),fielddata(b))) * FFT.Ωx
# end

# # dot(fourier, fourier)
# function _dot(a::X, b::X, ::Type{IsMap{false}}) where X<:XField{P} where P<:Flat
#     FFT = harmonic_transform(X)
#     sum(map(_complexdot, fielddata(a),fielddata(b))) * FFT.Ωk
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
