
#%% Cmap{F} and Cfourier{F} where F<:c2cTransform{T,d,ni}
#%% ============================================================

struct Cmap{F<:c2cTransform,T<:Real,d} <: XField{F}
    x::Array{Complex{T},d}
    Cmap{F,T,d}(x) where {T,d,F<:c2cTransform{T,d}} = new{F,T,d}(x)
    Cmap{F}(x::AbstractArray{A,d}) where {A,T,d,F<:c2cTransform{T,d}} = new{F,T,d}(Complex{T}.(x))
    Cmap{F}() where {T,d,ni,F<:c2cTransform{T,d,ni}} = new{F,T,d}(zeros(Complex{T}, ni))
    Cmap{F}(n::Number) where {T,d,ni,F<:c2cTransform{T,d,ni}} = new{F,T,d}(fill(Complex{T}(n), ni))
end

struct Cfourier{F<:c2cTransform,T<:Real,d} <: XField{F}
    k::Array{Complex{T},d}
    Cfourier{F,T,d}(k) where {T,d,F<:c2cTransform{T,d}} = new{F,T,d}(k)
    Cfourier{F}(k::AbstractArray{A,d}) where {A,T,d,F<:c2cTransform{T,d}} = new{F,T,d}(Complex{T}.(k))
    Cfourier{F}() where {T,d,F<:c2cTransform{T,d}} = new{F,T,d}(zeros(Complex{T}, ni))
    Cfourier{F}(n::Number) where {T,d,F<:c2cTransform{T,d}} = new{F,T,d}(fill(Complex{T}(n), ni))
end

#  union type
const Cfield{F,T,d} = Union{Cfourier{F,T,d}, Cmap{F,T,d}}

#  convert and promote
promote_rule(::Type{Cmap{F,T,d}}, ::Type{Cfourier{F,T,d}}) where {T,d,F<:c2cTransform{T,d}} = Cfourier{F,T,d}
convert(::Type{Cmap{F,T,d}}, f::Cfourier{F,T,d}) where {T,d,F<:c2cTransform{T,d}} = Cmap{F}(F \ f.k)
convert(::Type{Cfourier{F,T,d}}, f::Cmap{F,T,d}) where {T,d,F<:c2cTransform{T,d}} = Cfourier{F}(F * f.x)

# make constructors fall back to convert
Cfourier{F,T,d}(f::Cfield{F,T,d}) where {T,d,F<:c2cTransform{T,d}} = convert(Cfourier{F,T,d}, f)
Cfourier{F}(f::Cfield{F,T,d})     where {T,d,F<:c2cTransform{T,d}} = convert(Cfourier{F,T,d}, f)
Cmap{F,T,d}(f::Cfield{F,T,d})     where {T,d,F<:c2cTransform{T,d}} = convert(Cmap{F,T,d}, f)
Cmap{F}(f::Cfield{F,T,d})         where {T,d,F<:c2cTransform{T,d}} = convert(Cmap{F,T,d}, f)

# Use FT to convert between Cmap vrs Cfourier
(*)(::Type{F}, f::Cfourier{F,T,d}) where {T,d,F<:c2cTransform{T,d}} = f
(*)(::Type{F}, f::Cmap{F,T,d})     where {T,d,F<:c2cTransform{T,d}} = convert(Cfourier{F,T,d}, f)
(\)(::Type{F}, f::Cfourier{F,T,d}) where {T,d,F<:c2cTransform{T,d}} = convert(Cmap{F,T,d}, f)
(\)(::Type{F}, f::Cmap{F,T,d})     where {T,d,F<:c2cTransform{T,d}} = f

#  getindex
getindex(f::Cfield{F}, ::typeof(!)) where F = Cfourier{F}(f).k
getindex(f::Cfield{F}, ::Colon)     where F = Cmap{F}(f).x

function getindex(f::Cfield{F}, sym::Symbol) where F
    (sym == :k) ? Cfourier{F}(f).k :
    (sym == :x) ? Cmap{F}(f).x :
    error("index is not defined")
end

dot(f::Cfield{F}, g::Cfield{F}) where F = dot(f[:], g[:]) * Grid(F).Ωx
