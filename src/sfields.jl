## =====================================================
# Smap{FT} and Sfourier{FT} where FT<:FourierTransform 

struct Smap{FT<:rFourierTransform,d} <: XField{FT}
    x::Array{F64,d}
    Smap{FT,d}(x) where {d,FT<:rFT{d}} = new{FT,d}(x)
    Smap{FT}(x::Array{F64,d}) where {d,FT<:rFT{d}} = new{FT,d}(x)
	Smap{FT}(x::AbstractArray{T,d}) where {T,d,FT<:rFT{d}} = new{FT,d}(F64.(x))
	Smap{FT}() where {d,FT<:rFT{d}} = new{FT,d}(zeros(F64, Grid(FT).nxi))
	Smap{FT}(n::Number) where {d,FT<:rFT{d}} = new{FT,d}(fill(F64(n), Grid(FT).nxi))
end

struct Sfourier{FT<:rFourierTransform,d} <: XField{FT}
    k::Array{CF64,d}
    Sfourier{FT,d}(k) where {d,FT<:rFT{d}} = new{FT,d}(k)
    Sfourier{FT}(k::Array{CF64,d}) where {d,FT<:rFT{d}} = new{FT,d}(k)
	Sfourier{FT}(k::AbstractArray{T,d}) where {T,d,FT<:rFT{d}} = new{FT,d}(CF64.(k))
	Sfourier{FT}() where {d,FT<:rFT{d}} = new{FT,d}(zeros(CF64, Grid(FT).nki))
	Sfourier{FT}(n::Number) where {d,FT<:rFT{d}} = new{FT,d}(fill(CF64(n), Grid(FT).nki))
end

#  union type
const Sfield{FT,d} = Union{Sfourier{FT,d}, Smap{FT,d}}

#  convert and promote
promote_rule(::Type{Smap{FT,d}}, ::Type{Sfourier{FT,d}}) where {d,FT<:rFT{d}} = Sfourier{FT,d}
convert(::Type{Smap{FT,d}}, f::Sfourier{FT,d}) where {d,FT<:rFT{d}} = Smap{FT}(FT \ f.k)
convert(::Type{Sfourier{FT,d}}, f::Smap{FT,d}) where {d,FT<:rFT{d}} = Sfourier{FT}(FT * f.x)

# make constructors fall back to convert
Sfourier{FT,d}(f::Sfield{FT,d}) where {d,FT<:rFT{d}} = convert(Sfourier{FT,d}, f)
Sfourier{FT}(f::Sfield{FT,d}) where {d,FT<:rFT{d}}   = convert(Sfourier{FT,d}, f)
Smap{FT,d}(f::Sfield{FT,d}) where {d,FT<:rFT{d}}     = convert(Smap{FT,d}, f)
Smap{FT}(f::Sfield{FT,d}) where {d,FT<:rFT{d}}       = convert(Smap{FT,d}, f)

# Use FT to convert between Smap vrs Sfourier
(*)(::Type{FT}, f::Sfourier{FT,d}) where {d,FT<:rFT{d}} = f
(*)(::Type{FT}, f::Smap{FT,d})     where {d,FT<:rFT{d}} = convert(Sfourier{FT,d}, f)
(\)(::Type{FT}, f::Sfourier{FT,d}) where {d,FT<:rFT{d}} = convert(Smap{FT,d}, f)
(\)(::Type{FT}, f::Smap{FT,d})     where {d,FT<:rFT{d}} = f

#  getindex
getindex(f::Sfield{FT}, ::typeof(!)) where FT = Sfourier{FT}(f).k
getindex(f::Sfield{FT}, ::Colon)     where FT = Smap{FT}(f).x

function getindex(f::Sfield{FT}, sym::Symbol) where FT
    (sym == :k) ? Sfourier{FT}(f).k :
    (sym == :x) ? Smap{FT}(f).x :
    error("index is not defined")
end

dot(f::Sfield{FT}, g::Sfield{FT}) where FT = dot(f[:], g[:]) * Grid(FT).Ωx
