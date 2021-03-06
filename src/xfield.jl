# Xmap
# =================================

struct Xmap{F<:Transform, Tf<:Number, Ti<:Number, d} <: MapField{F,Tf,Ti,d}
    ft::F
    fd::Array{Tf,d}
    function Xmap{F,Tf,Ti,d}(ft::F, fd::AbstractArray{T,d})  where {Tf,Ti,T,d,F<:Transform{Tf,d}}
        @assert Ti == eltype_out(ft)
        @assert size(fd) == size_in(ft)
        new{F,Tf,Ti,d}(ft, fd)
    end
end

# Xfourier
# =================================

struct Xfourier{F<:Transform, Tf<:Number, Ti<:Number, d}  <: FourierField{F,Tf,Ti,d}
    ft::F
    fd::Array{Ti,d}
    function Xfourier{F,Tf,Ti,d}(ft::F, fd::AbstractArray{T,d})  where {Tf,Ti,T,d,F<:Transform{Tf,d}}
        @assert Ti == eltype_out(ft)
        @assert size(fd) == size_out(ft)
        new{F,Tf,Ti,d}(ft, fd)
    end
end

# Interface definitions required to hook into Field definitions
# =================================

FourierField(::Type{Xfourier{F,Tf,Ti,d}}) where {Tf,Ti,d, F<:Transform{Tf,d}} = Xfourier{F,Tf,Ti,d}
FourierField(::Type{Xmap{F,Tf,Ti,d}}) where {Tf,Ti,d, F<:Transform{Tf,d}}     = Xfourier{F,Tf,Ti,d}
MapField(::Type{Xmap{F,Tf,Ti,d}})     where {Tf,Ti,d, F<:Transform{Tf,d}}     = Xmap{F,Tf,Ti,d}
MapField(::Type{Xfourier{F,Tf,Ti,d}}) where {Tf,Ti,d, F<:Transform{Tf,d}}     = Xmap{F,Tf,Ti,d}

Xfield{F,Tf,Ti,d} = Union{Xfourier{F,Tf,Ti,d}, Xmap{F,Tf,Ti,d}}
@inline fieldtransform(f::Xfield{F,Tf,Ti,d})  where {Tf,Ti,d, F<:Transform{Tf,d}} = f.ft
@inline fielddata(f::Xfield{F,Tf,Ti,d}) where {Tf,Ti,d, F<:Transform{Tf,d}} = f.fd


# Extras
# =================================

function Xfourier(ft::F, fd::AbstractArray{T,d})  where {T,Tf,d,F<:Transform{Tf,d}}
    @assert size(fd) == size_out(ft)
    Ti = eltype_out(ft)
    Xfourier{F,Tf,Ti,d}(ft, fd)
end

function Xmap(ft::F, fd::AbstractArray{T,d})  where {T,Tf,d,F<:Transform{Tf,d}}
    @assert size(fd) == size_in(ft)
    Ti = eltype_out(ft)
    Xmap{F,Tf,Ti,d}(ft, fd)
end

Xmap(f::Xfield{F,Tf,Ti,d})     where {Tf,Ti,d, F<:Transform{Tf,d}} = MapField(f)
Xfourier(f::Xfield{F,Tf,Ti,d}) where {Tf,Ti,d, F<:Transform{Tf,d}} = FourierField(f)

# zero and constant constructors
Xmap(ft::F)                 where {Tf,d,F<:Transform{Tf,d}} = Xmap(ft, zeros(Tf, size_in(ft)))
Xmap(ft::F, n::Number)      where {Tf,d,F<:Transform{Tf,d}} = Xmap(ft, fill(Tf(n), size_in(ft)))
function Xfourier(ft::F) where {Tf,d,F<:Transform{Tf,d}}
    Ti = eltype_out(ft) 
    Xfourier(ft, zeros(Ti, size_out(ft)))
end
function Xfourier(ft::F, n::Number) where {Tf,d,F<:Transform{Tf,d}} 
    Ti = eltype_out(ft) 
    Xfourier(ft, fill(Ti(n), size_out(ft)))
end
