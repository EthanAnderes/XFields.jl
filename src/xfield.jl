# Xmap
# =================================

struct Xmap{F<:Transform, Tf<:Number, Ti<:Number, d} <: MapField{F,Tf,Ti,d}
    ft::F
    fd::Array{Tf,d}
    function Xmap{F,Tf,Ti,d}(ft::F, fd::AbstractArray{T,d})  where {F,Tf,Ti,d,T}
        @assert Tf == eltype_in(ft)
        @assert Ti == eltype_out(ft)
        @assert size(fd) == size_in(ft)
        new{F,Tf,Ti,d}(ft, Tf.(fd)) 
    end
    # Note: it is important that the constructor Xmap{F,Tf,Ti,d}
    # check that the type parameters Tf, Ti, d match the 
    # input, output, dimension of the array storage that 
    # the transform ft::F expects. This is the only place that this 
    # check happens.
end


# Xfourier
# =================================

struct Xfourier{F<:Transform, Tf<:Number, Ti<:Number, d}  <: FourierField{F,Tf,Ti,d}
    ft::F
    fd::Array{Ti,d}
    function Xfourier{F,Tf,Ti,d}(ft::F, fd::AbstractArray{T,d}) where {F,Tf,Ti,d,T}
        @assert Tf == eltype_in(ft)
        @assert Ti == eltype_out(ft)
        @assert size(fd) == size_out(ft)
        new{F,Tf,Ti,d}(ft, Ti.(fd))
    end
    # Note: it is important that the constructor Xfield{F,Tf,Ti,d}
    # check that the type parameters Tf, Ti, d match the 
    # input, output, dimension of the array storage that 
    # the transform ft::F expects. This is the only place that this 
    # check happens.
end

# Interface definitions required to hook into Field definitions
# =================================

FourierField(::Type{Xfourier{F,Tf,Ti,d}}) where {F,Tf,Ti,d} = Xfourier{F,Tf,Ti,d}
FourierField(::Type{Xmap{F,Tf,Ti,d}}) where {F,Tf,Ti,d}     = Xfourier{F,Tf,Ti,d}
MapField(::Type{Xmap{F,Tf,Ti,d}})     where {F,Tf,Ti,d}     = Xmap{F,Tf,Ti,d}
MapField(::Type{Xfourier{F,Tf,Ti,d}}) where {F,Tf,Ti,d}     = Xmap{F,Tf,Ti,d}

Xfield{F,Tf,Ti,d} = Union{Xfourier{F,Tf,Ti,d}, Xmap{F,Tf,Ti,d}}
@inline fieldtransform(f::Xfield{F,Tf,Ti,d})  where {F,Tf,Ti,d} = f.ft
@inline fielddata(f::Xfield{F,Tf,Ti,d}) where {F,Tf,Ti,d} = f.fd


# Extras
# =================================

function Xfourier(ft::F, fd::AbstractArray{T,d}) where {F<:Transform,T,d}
    @assert size(fd) == size_out(ft)
    Tf = eltype_in(ft)
    Ti = eltype_out(ft)
    Xfourier{F,Tf,Ti,d}(ft, fd)
end

function Xmap(ft::F, fd::AbstractArray{T,d}) where {F<:Transform,T,d}
    @assert size(fd) == size_in(ft)
    Tf = eltype_in(ft)
    Ti = eltype_out(ft)
    Xmap{F,Tf,Ti,d}(ft, fd)
end

Xmap(f::Xfield{F,Tf,Ti,d})     where {F,Tf,Ti,d} = MapField(f)
Xfourier(f::Xfield{F,Tf,Ti,d}) where {F,Tf,Ti,d} = FourierField(f)

# zero and constant constructors
Xmap(ft::Transform) = Xmap(ft, zeros(eltype_in(ft), size_in(ft)))
function Xmap(ft::Transform, n::Number)  
    Tf = eltype_in(ft)
    Xmap(ft, fill(Tf(n), size_in(ft)))
end

Xfourier(ft::Transform) = Xfourier(ft, zeros(eltype_out(ft), size_out(ft)))
function Xfourier(ft::Transform, n::Number)
    Ti = eltype_out(ft) 
    Xfourier(ft, fill(Ti(n), size_out(ft)))
end
