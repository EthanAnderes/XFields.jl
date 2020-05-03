# Xmap
# =================================

struct Xmap{F<:Transform, Tf<:Number, Ti<:Number, d} <: MapField{F}
    ft::F
    f::Array{Tf,d}
    function Xmap{F,Tf,Ti,d}(ft::F, f::Array{Tf,d})  where {Tf,Ti,d,F<:Transform{Tf,d}}
        @assert Ti == eltype_out(ft)
        @assert size(f) == size_in(ft)
        new{F,Tf,Ti,d}(ft, f)
    end
    function Xmap(ft::F, f::Array{T,d})  where {T,Tf,d,F<:Transform{Tf,d}}
        @assert size(f) == size_in(ft)
        Ti = eltype_out(ft)
        new{F,Tf,Ti,d}(ft, f)
    end

end

# some extra constructors
Xmap(ft::F)  where {Tf,d,F<:Transform{Tf,d}}            = Xmap(ft, zeros(Tf, size_in(ft)))
Xmap(ft::F, n::Number)  where {Tf,d,F<:Transform{Tf,d}} = Xmap(ft, fill(Tf(n), size_in(ft)))

# Xfourier
# =================================

struct Xfourier{F<:Transform, Tf<:Number, Ti<:Number, d}  <: FourierField{F}
    ft::F
    f::Array{Ti,d}
    function Xfourier{F,Tf,Ti,d}(ft::F, f::Array{Ti,d})  where {Tf,Ti,d,F<:Transform{Tf,d}}
        @assert Ti == eltype_out(ft)
        @assert size(f) == size_out(ft)
        new{F,Tf,Ti,d}(ft, f)
    end
    function Xfourier(ft::F, f::Array{T,d})  where {T,Tf,d,F<:Transform{Tf,d}}
        @assert size(f) == size_out(ft)
        Ti = eltype_out(ft)
        new{F,Tf,Ti,d}(ft, f)
    end
end

# some extra constructors
Xfourier(ft::F)  where {Tf,d,F<:Transform{Tf,d}}            = Xfourier(ft, zeros(Tf, size_in(ft)))
Xfourier(ft::F, n::Number)  where {Tf,d,F<:Transform{Tf,d}} = Xfourier(ft, fill(Tf(n), size_in(ft)))



# Union and getindex
# =================================

# union 
const Xfield{F,Tf,Ti,d} = Union{Xfourier{F,Tf,Ti,d}, Xmap{F,Tf,Ti,d}}


#  getindex
Base.getindex(f::Xfield, ::typeof(!)) = Xfourier(f).f
Base.getindex(f::Xfield, ::Colon)     = Xmap(f).f


