


# Xmap
# =================================
# where {Tf,d,F<:Transform{Tf,d},Ti}


# function Xmap(t::𝕎{Tf,d}, f::AA{R,d}) where {Tf,d} 
#     Xmap(t, Tf.(f))
# end

# function Xmap(t::𝕎{Tf,d}, n::Number) where {Tf,d} 
#     f = fill(Tf(n), size_in(t))
#     Xmap(t,f)
# end

# function Xmap(w::𝕎{Tf,d}) where {Tf,d} 
#     f = zeros(Tf, w.sz)
#     Xmap(w,f)
# end


struct Xmap{F<:Transform, Tf<:Number, Ti<:Number, d} <: MapField{F}
    ft::F
    f::Array{Tf,d}
    function Xmap{F,Tf,Ti,d}(ft::F, f::Array{Tf,d})  where {Tf,d,F<:Transform{Tf,d},Ti}
        @assert Ti == eltype_out(ft)
        @assert size(f) == size_in(ft)
        new{F,Tf,Ti,d}(ft, f)
    end
    function Xmap(ft::F, f::Array{Tf,d})  where {Tf,d,F<:Transform{Tf,d}}
        Ti = eltype_out(ft)
        @assert size(f) == size_in(ft)
        new{F,Tf,Ti,d}(ft, f)
    end
end

# Xfourier
# =================================
# TODO: ensure constraints between F,Tf, Ti and d with the inner constructor
# Xmap(ft,f) ...

struct Xfourier{F<:Transform, Tf<:Number, Ti<:Number, d}  <: FourierField{F}
    ft::F
    f::Array{Ti,d}
    # I think I need the full constructor to allow F<:Xfourier to construct the method
    function Xfourier{F,Tf,Ti,d}(ft::F, f::Array{Ti,d})  where {Tf,d,F<:Transform{Tf,d},Ti}
        @assert Ti == eltype_out(ft)
        @assert size(f) == size_out(ft)
        new{F,Tf,Ti,d}(ft, f)
    end
    function Xfourier(ft::F, f::Array{Ti,d})  where {Tf,d,F<:Transform{Tf,d},Ti}
        @assert Ti == eltype_out(ft)
        @assert size(f) == size_out(ft)
        new{F,Tf,Ti,d}(ft, f)
    end
end

# function Xfourier(w::𝕎{Tf,d}, f::AA{R,d}) where {Tf,d}
#     @assert size(f) == size_out(w) 
#     Ti = eltype_out(w)
#     Xfourier(w, Ti.(f))
# end

# function Xfourier(w::𝕎{Tf,d}, n::Number) where {Tf,d}
#     Ti = eltype_out(w)  
#     f = fill(Ti(n), size_out(w))
#     Xfourier(w, f)
# end

# function Xfourier(w::𝕎{Tf,d}) where {Tf,d} 
#     Ti = eltype_out(w)  
#     f = zeros(Ti, size_out(w))
#     Xfourier(w, f)
# end

# 
# =================================

#  union type
const Xfield{F,Tf,Ti,d} = Union{Xfourier{F,Tf,Ti,d}, Xmap{F,Tf,Ti,d}}


#  getindex
getindex(f::Xfield, ::typeof(!)) = Xfourier(f).f
getindex(f::Xfield, ::Colon)     = Xmap(f).f

# function getindex(f::Xfield, sym::Symbol)
#     (sym == :k) ? Xfourier(f).f :
#     (sym == :l) ? Xfourier(f).f :
#     (sym == :x) ? Xmap(f).f :
#     error("index is not defined")
# end

dot(f::Xfield{F}, g::Xfield{F}) where F = dot(f[:], g[:]) .* Ωx(F)



