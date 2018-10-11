module CirculantMatrices
using FFTW
using FFTW: Plan
using LinearAlgebra

import Base: convert, eltype, *, \, /, getindex, print_matrix, size, Matrix, +, copy, -,
             real, imag, conj

#import Base: copyto!
import LinearAlgebra: mul!, ldiv!, inv, norm, eigvals, eigvecs, eigen, adjoint, cond,
                     issymmetric, ishermitian, rank, axpy!, adjoint!, transpose!

import LinearAlgebra: rmul!, lmul!

# import Base: setindex!,
#               Nothing, copyto!,
#              , ^, conj!, promote_rule, similar, fill!,

export Circulant

mutable struct Circulant{T <: Number, S <: Number} <: AbstractMatrix{T}
    vc::Vector{T}
    vc_dft::Vector{S}
    fftplan::Plan
    ifftplan::Plan
end

function Circulant{T}(vc::AbstractVector{T}) where T
    tmp = plan_fft(vc)
    return Circulant(copy(convert(Array{T,1},vc)), tmp*vc, tmp, plan_ifft(vc))
end

Circulant(vc::AbstractVector) = Circulant{ eltype(vc) <: Integer ?
                                            promote_type(eltype(vc), Float64) :
                                            promote_type(eltype(vc), Float32)}(vc)
Circulant{T}(A::AbstractMatrix) where T = Circulant{T}(A[:,1])
Circulant(A::AbstractMatrix) = Circulant(A[:,1])


## From Base
size(A::Circulant) = (size(A, 1), size(A, 2))

size(A::Circulant, _) = length(A.vc)

eltype(A::Circulant) = eltype(A.vc)

function getindex(C::Circulant, i::Integer, j::Integer)
    n = size(C, 1)
    if i > n || j > n
        error("BoundsError()")
    end
    return C.vc[mod(i - j, length(C.vc)) + 1]
end

function copy(A::Circulant)
    #println("We are copying it")
    return Circulant(copy(A.vc))
end

#function (=)(A::Circulant)
# Convert an Circulant matrix to a full matrix
function Matrix(A::Circulant{T}) where T
    m, n = size(A)
    #println("Generating a matrix of dimension: " * string(m) * "x" * string(n))
    Af = Matrix{T}(undef, m, n)
    for j = 1:n
        for i = 1:m
            Af[i,j] = A[i,j]
        end
    end
    return Af
end

function real(A::Circulant)
    return Circulant(real(A.vc))
end

function imag(A::Circulant)
    return Circulant(imag(A.vc))
end

function isreal(A::Circulant)
    if all(imag(A.vc) .!= 0)
        return false
    else
        return true
    end
end

function conj(A::Circulant)
    return Circulant(conj(A.vc))
end

function mul!(Y::Circulant, A::Circulant, B::Circulant)
    if (size(A, 2) != size(B, 1)) || (size(A, 1) != size(Y, 1))
        error("DimensionsMismatch()")
    end
    #println("Here we are 4")
    Y.vc_dft[:] = A.vc_dft.*B.vc_dft
    Y.vc[:] = convert(Vector{eltype(Y)},
              eltype(Y) <: Real ? real(Y.ifftplan*Y.vc_dft) : Y.ifftplan*Y.vc_dft )
end

function (*)(A::Circulant, B::Circulant)
    Y = Circulant(Vector{promote_type(eltype(A), eltype(B))}(undef, size(A,1)))
    mul!(Y, A, B)
    return Y
end

function (+)(A::Circulant, B::Circulant)
    if (size(A, 1) != size(B, 1))
        error("DimensionsMismatch()")
    end
    return Circulant(
            Vector{promote_type(eltype(A), eltype(B))}(A.vc .+ B.vc),
            A.vc_dft .+ B.vc_dft,
            A.fftplan,
            A.ifftplan)
end

function (-)(A::Circulant, B::Circulant)
    if (size(A, 1) != size(B, 1))
        error("DimensionsMismatch()")
    end
    return Circulant(
            Vector{promote_type(eltype(A), eltype(B))}(A.vc .- B.vc),
            A.vc_dft .- B.vc_dft,
            A.fftplan,
            A.ifftplan)
end

function (-)(A::Circulant)
    return Circulant(
            -A.vc,
            -A.vc_dft,
            A.fftplan,
            A.ifftplan)
end


function mul!(Y::AbstractMatrix, A::Circulant, B::AbstractMatrix)
    if (size(A, 2) != size(B, 1)) || (size(A, 1) != size(Y, 1)) ||
        (size(B, 2) != size(Y, 2))
        error("DimensionsMismatch()")
    end

    rettype = eltype(Y)
    if rettype <: Real && ( eltype(A) <: Complex || eltype(B) <:  Complex )
        println("Output matrix is real, while input(s) are complex")
    end
    #println("Here we are 5")

    for i=1:size(B, 2)
        Y[:, i] = convert(Vector{eltype(Y)},
                eltype(Y) <: Real ?
                real(A.ifftplan*(A.vc_dft.*(A.fftplan*B[:, i]))) :
                A.ifftplan*(A.vc_dft.*(A.fftplan*B[:, i])))
    end
end

function mul!(Y::AbstractMatrix, A::Circulant, B::Adjoint{<:Any,<:AbstractMatrix})
    #println("Adjoint mul R")
    mul!(Y, A, copy(B))
end

function mul!(Y::AbstractMatrix, A::Circulant, B::Transpose{<:Any,<:AbstractMatrix})
    #println("Transpose mul R")
    mul!(Y, A, copy(B))
end

function (*)(A::Circulant, B::AbstractMatrix)
    Y = Matrix{promote_type(eltype(A), eltype(B))}(undef, size(A, 1), size(B, 2))
    mul!(Y, A, B)
    return Y
end

function mul!(y::AbstractVector, A::Circulant, x::AbstractVector)
    if length(y) != length(x) || size(A, 2) != length(x)
        throw(DimensionMismatch(""))
    end

    rettype = eltype(y)
    if rettype <: Real && ( eltype(A) <: Complex || eltype(x) <:  Complex )
        println("Output matrix is real, while input(s) are complex")
    end

    #println("Here we are 222")
    y[:] = convert(Vector{eltype(y)},
            eltype(y) <: Real ?
            real(A.ifftplan*(A.vc_dft.*(A.fftplan*x))) :
            A.ifftplan*(A.vc_dft.*(A.fftplan*x)))
end

function (*)(A::Circulant, x::AbstractVector)
    y = Vector{promote_type(eltype(A), eltype(x))}(undef, size(A, 1))
    #println("Here we are, what about 222")
    mul!(y, A, x)
    return y
end

function mul!(Y::AbstractMatrix, A::AbstractMatrix, B::Circulant)
    if (size(A, 2) != size(B, 1)) || (size(A, 1) != size(Y, 1)) ||
        (size(B, 2) != size(Y, 2))
        error("DimensionsMismatch()")
    end

    rettype = eltype(Y)
    if rettype <: Real && ( eltype(A) <: Complex || eltype(B) <:  Complex )
        println("Output matrix is real, while input(s) are complex")
    end
    #println("Here we are 6")

    for i=1:size(Y, 1)
        Y[i, :] = convert(Vector{eltype(Y)},
                eltype(Y) <: Real ?
                real(B.fftplan*(B.vc_dft.*(B.ifftplan*A[i, :]))) :
                B.fftplan*(B.vc_dft.*(B.ifftplan*A[i, :])))
    end
end

function mul!(Y::AbstractMatrix, A::Adjoint{<:Any,<:AbstractMatrix}, B::Circulant)
    #println("Adjoint mul L")
    mul!(Y, copy(A), B)
end

function mul!(Y::AbstractMatrix, A::Transpose{<:Any,<:AbstractMatrix}, B::Circulant)
    #println("Transpose mul 1")
    mul!(Y, copy(A), B)
end

function (*)(A::AbstractMatrix, B::Circulant)
    Y = Matrix{promote_type(eltype(A), eltype(B))}(undef, size(A, 1), size(B, 2))
    mul!(Y, A, B)
    return Y
end

function mul!(Y::Circulant, A::Circulant, b::Number)
    if (size(Y, 1) != size(A, 1)) || (size(Y, 2) != size(A, 2))
        error("DimensionsMismatch()")
    end

    rettype = eltype(Y)
    if rettype <: Real && ( eltype(A) <: Complex)
        error("Output matrix is real, while input(s) are complex")
    end
    #println("Here we are 7")

    Y.vc = A.vc .* b
    Y.vc_dft = A.vc_dft .* b
    Y.fftplan = A.fftplan
    Y.ifftplan = A.ifftplan
end

function mul!(Y::Circulant, b::Number, A::Circulant)
    mul!(Y, A, b)
end

function (*)(A::Circulant, b::Number)
    Y = Circulant(Vector{promote_type(eltype(A), eltype(b))}(undef, size(A,1)))
    mul!(Y, A, b)
    return Y
end

function (/)(A::Circulant, b::Number)
    Y = Circulant(Vector{promote_type(eltype(A), eltype(b))}(undef, size(A,1)))
    mul!(Y, A, 1/b)
    return Y
end

# function .(*)(A::Circulant,b::Number)
#     return A*b
# end

function (*)(b::Number, A::Circulant)
    Y = Circulant(Vector{promote_type(eltype(A), eltype(b))}(undef, size(A,1)))
    mul!(Y, A, b)
    return Y
end


function (/)(b::Number, A::Circulant)
    error("ERROR #84616: Not implemented yet")
end

function lmul!(a::Number, B::Circulant)
    error("ERROR #78913: Not implemented yet")
end

function rmul!(A::Circulant, b::Number)
    error("ERROR #123564: Not implemented yet")
end

function axpy!(a, X::Circulant, Y::Circulant)
    Y.vc[:] = Y.vc[:] + X.vc[:]*a
    Y.vc_dft[:] = Y.vc_dft[:] + X.vc_dft[:]*a
end

#
# function .(*)(b::Number, A::Circulant)
#     return A*b
# end

function ldiv!(y::AbstractVector, A::Circulant, x::AbstractVector)
    @assert length(x) == length(y)
    if T <: Real
        y[:] = real(A.ifftplan*((1 ./ A.vc_dft).*(A.fftplan*x)))
    else
        y[:] = A.ifftplan*((1 ./ A.vc_dft).*(A.fftplan*x))
    end
end

function ldiv!(Y::AbstractMatrix, A::Circulant, X::AbstractMatrix)
    @assert size(Y) == size(X)
    for i=1:size(X, 2)
        if T <: Real
            Y[:,i] = real(A.ifftplan*((1 ./ A.vc_dft).*(A.fftplan*X[:,i])))
        else
            Y[:,i] = A.ifftplan*((1 ./ A.vc_dft).*(A.fftplan*X[:,i]))
        end
    end
end

# function copy(A::Adjoint{Circulant})
#     return(\))
# end

function adjoint!(A::Circulant)
    error("Error #78617")
end

function transpose!(A::Circulant)
    error("Error #123489")
end

function adjoint(A::Circulant)
    n = size(A,1)
    Adj = zeros(eltype(A),n)
    Adj[1] = conj(A.vc[1])

    for i = 2:n
        Adj[i] = conj(A.vc[n-i+2])
    end

    return Circulant(Adj)
end

function transpose(A::Circulant)
    n = size(A,1)
    Trsp = zeros(eltype(A), n)
    Trsp[1] = A.vc[1]

    for i = 2:n
        Trsp[i] = A.vc[n-i+2]
    end

    return Circulant(Trsp)
end


function inv(A::Circulant)
    if eltype(A) <: Real
        return Circulant(convert(Vector{eltype(A)}, real(A.ifftplan*(1 ./ A.vc_dft))), 1 ./ A.vc_dft, A.fftplan, A.ifftplan)
    else
        return Circulant(convert(Vector{eltype(A)}, A.ifftplan*(1 ./ A.vc_dft)), 1 ./ A.vc_dft, A.fftplan, A.ifftplan)
    end
end

function norm(A::Circulant)
    return maximum(abs.(A.vc_dft))
end

function eigvals(A::Circulant, vl=1, vr=size(A,1))
    return A.vc_dft[vl:vr]
end

function eigvecs(A::Circulant, vl=1, vr=size(A,1))
    n = size(A, 1)
    E = zeros(Complex{Float64}, n, vr-vl+1)
    for i = 1:size(A,1)
        for j = vl:vr
            E[i, j-vl+1] = (exp(im*2*π*(j-1)/n))^(i-1)/sqrt(n)
        end
    end
    return E
end

function eigen(A::Circulant, vl=1, vr=size(A,1))
    return (eigvals(A, vl, vr), eigvecs(A, vl, vr))
end

function cond(A::Circulant)
    σ = abs.(eigvals(A))
    return maximum(σ)/minimum(σ)
end

function (\)(A::Circulant, b::AbstractVector)
    return inv(A)*b
end
#
# function ishermitian(A::Circulant)
#     if all((imag(A.vc_dft) ./ real(A.vc_dft)) .< eps()*100)
#         println("Matrix is Hermitian")
#         return true
#     else
#         println("Matrix is not Hermitian")
#         return false
#     end
# end

function ishermitian(A::Circulant)
    diffvec = abs.(A.vc[2:end] - conj(reverse(A.vc[2:end])))
    if all(diffvec .< eps()*10)
        #println("Matrix is Hermitian")
        return true
    else
        #println("Matrix is not Hermitian")
        return false
    end
end

function issymmetric(A::Circulant)
    diffvec = abs.(A.vc[2:end] - reverse(A.vc[2:end]))
    if all(diffvec .< eps()*10)
        #println("Matrix is symmetric")
        return true
    else
        #println("Matrix is not symmetric")
        return false
    end
end

function rank(A::Circulant; rtol=eps())
    myrank = 0
    atol = norm(A)*rtol
    σ  = abs.(A.vc_dft)
    for i=1:size(A,1)
        if σ[i] > atol
            myrank = myrank + 1
        end
    end
    return myrank
end
#function
end
