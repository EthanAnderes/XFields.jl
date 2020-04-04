
#%% Rmap{F} and Rfourier{F} where F<:r2cTransform{T,d,ni}
#%% ============================================================


struct Rmap{F<:r2cTransform,T<:Real,d} <: XField{F}
    x::Array{T,d}
    Rmap{F,T,d}(x) where {T,d,F<:r2cTransform{T,d}} = new{F,T,d}(x)
    Rmap{F}(x::AbstractArray{A,d}) where {A,T,d,F<:r2cTransform{T,d}} = new{F,T,d}(T.(x))
    Rmap{F}() where {T,d,ni,F<:r2cTransform{T,d,ni}} = new{F,T,d}(zeros(T, ni))
    Rmap{F}(n::Number) where {T,d,ni,F<:r2cTransform{T,d,ni}} = new{F,T,d}(fill(T(n), ni))
end

struct Rfourier{F<:r2cTransform,T<:Real,d} <: XField{F}
    k::Array{Complex{T},d}
    Rfourier{F,T,d}(k) where {T,d,F<:r2cTransform{T,d}} = new{F,T,d}(k)
    Rfourier{F}(k::AbstractArray{A,d}) where {A,T,d,F<:r2cTransform{T,d}} = new{F,T,d}(Complex{T}.(k))
    Rfourier{F}() where {T,d,F<:r2cTransform{T,d}} = new{F,T,d}(zeros(Complex{T}, Grid(F).nki))
    Rfourier{F}(n::Number) where {T,d,F<:r2cTransform{T,d}} = new{F,T,d}(fill(Complex{T}(n), Grid(F).nki))
end

#  union type
const Rfield{F,T,d} = Union{Rfourier{F,T,d}, Rmap{F,T,d}}

#  convert and promote
promote_rule(::Type{Rmap{F,T,d}}, ::Type{Rfourier{F,T,d}}) where {T,d,F<:r2cTransform{T,d}} = Rfourier{F,T,d}
convert(::Type{Rmap{F,T,d}}, f::Rfourier{F,T,d}) where {T,d,F<:r2cTransform{T,d}} = Rmap{F}(F \ f.k)
convert(::Type{Rfourier{F,T,d}}, f::Rmap{F,T,d}) where {T,d,F<:r2cTransform{T,d}} = Rfourier{F}(F * f.x)

# make constructors fall back to convert
Rfourier{F,T,d}(f::Rfield{F,T,d}) where {T,d,F<:r2cTransform{T,d}} = convert(Rfourier{F,T,d}, f)
Rfourier{F}(f::Rfield{F,T,d})     where {T,d,F<:r2cTransform{T,d}} = convert(Rfourier{F,T,d}, f)
Rmap{F,T,d}(f::Rfield{F,T,d})     where {T,d,F<:r2cTransform{T,d}} = convert(Rmap{F,T,d}, f)
Rmap{F}(f::Rfield{F,T,d})         where {T,d,F<:r2cTransform{T,d}} = convert(Rmap{F,T,d}, f)

# Use FT to convert between Rmap vrs Rfourier
(*)(::Type{F}, f::Rfourier{F,T,d}) where {T,d,F<:r2cTransform{T,d}} = f
(*)(::Type{F}, f::Rmap{F,T,d})     where {T,d,F<:r2cTransform{T,d}} = convert(Rfourier{F,T,d}, f)
(\)(::Type{F}, f::Rfourier{F,T,d}) where {T,d,F<:r2cTransform{T,d}} = convert(Rmap{F,T,d}, f)
(\)(::Type{F}, f::Rmap{F,T,d})     where {T,d,F<:r2cTransform{T,d}} = f

#  getindex
getindex(f::Rfield{F}, ::typeof(!)) where F = Rfourier{F}(f).k
getindex(f::Rfield{F}, ::Colon)     where F = Rmap{F}(f).x

function getindex(f::Rfield{F}, sym::Symbol) where F
    (sym == :k) ? Rfourier{F}(f).k :
    (sym == :x) ? Rmap{F}(f).x :
    error("index is not defined")
end

dot(f::Rfield{F}, g::Rfield{F}) where F = dot(f[:], g[:]) * Grid(F).Ωx
