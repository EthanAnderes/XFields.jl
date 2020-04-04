
#%% Smap{FT} and Sfourier{FT} where FT<:FourierTransform 
#%% ============================================================


struct Smap{F<:rFourierTransform,T<:Real,d} <: XField{F}
    x::Array{T,d}
    Smap{F,T,d}(x) where {T,d,F<:rFT{T,d}} = new{F,T,d}(x)
    Smap{F}(x::AbstractArray{A,d}) where {A,T,d,F<:rFT{T,d}} = new{F,T,d}(T.(x))
    Smap{F}() where {T,d,ni,F<:rFT{T,d,ni}} = new{F,T,d}(zeros(T, ni))
    Smap{F}(n::Number) where {T,d,ni,F<:rFT{T,d,ni}} = new{F,T,d}(fill(T(n), ni))
end

struct Sfourier{F<:rFourierTransform,T<:Real,d} <: XField{F}
    k::Array{Complex{T},d}
    Sfourier{F,T,d}(k) where {T,d,F<:rFT{T,d}} = new{F,T,d}(k)
    Sfourier{F}(k::AbstractArray{A,d}) where {A,T,d,F<:rFT{T,d}} = new{F,T,d}(Complex{T}.(k))
    Sfourier{F}() where {T,d,F<:rFT{T,d}} = new{F,T,d}(zeros(Complex{T}, Grid(F).nki))
    Sfourier{F}(n::Number) where {T,d,F<:rFT{T,d}} = new{F,T,d}(fill(Complex{T}(n), Grid(F).nki))
end

#  union type
const Sfield{F,T,d} = Union{Sfourier{F,T,d}, Smap{F,T,d}}

#  convert and promote
promote_rule(::Type{Smap{F,T,d}}, ::Type{Sfourier{F,T,d}}) where {T,d,F<:rFT{T,d}} = Sfourier{F,T,d}
convert(::Type{Smap{F,T,d}}, f::Sfourier{F,T,d}) where {T,d,F<:rFT{T,d}} = Smap{F}(F \ f.k)
convert(::Type{Sfourier{F,T,d}}, f::Smap{F,T,d}) where {T,d,F<:rFT{T,d}} = Sfourier{F}(F * f.x)

# make constructors fall back to convert
Sfourier{F,T,d}(f::Sfield{F,T,d}) where {T,d,F<:rFT{T,d}} = convert(Sfourier{F,T,d}, f)
Sfourier{F}(f::Sfield{F,T,d})     where {T,d,F<:rFT{T,d}} = convert(Sfourier{F,T,d}, f)
Smap{F,T,d}(f::Sfield{F,T,d})     where {T,d,F<:rFT{T,d}} = convert(Smap{F,T,d}, f)
Smap{F}(f::Sfield{F,T,d})         where {T,d,F<:rFT{T,d}} = convert(Smap{F,T,d}, f)

# Use FT to convert between Smap vrs Sfourier
(*)(::Type{F}, f::Sfourier{F,T,d}) where {T,d,F<:rFT{T,d}} = f
(*)(::Type{F}, f::Smap{F,T,d})     where {T,d,F<:rFT{T,d}} = convert(Sfourier{F,T,d}, f)
(\)(::Type{F}, f::Sfourier{F,T,d}) where {T,d,F<:rFT{T,d}} = convert(Smap{F,T,d}, f)
(\)(::Type{F}, f::Smap{F,T,d})     where {T,d,F<:rFT{T,d}} = f

#  getindex
getindex(f::Sfield{F}, ::typeof(!)) where F = Sfourier{F}(f).k
getindex(f::Sfield{F}, ::Colon)     where F = Smap{F}(f).x

function getindex(f::Sfield{F}, sym::Symbol) where F
    (sym == :k) ? Sfourier{F}(f).k :
    (sym == :x) ? Smap{F}(f).x :
    error("index is not defined")
end

dot(f::Sfield{F}, g::Sfield{F}) where F = dot(f[:], g[:]) * Grid(F).Ωx
