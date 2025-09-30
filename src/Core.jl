# Basics for exact diagonalization method
"""
    const edtimer = TimerOutput()

Default shared timer for all exact diagonalization methods.
"""
const edtimer = TimerOutput()

"""
    EDKind{K}

Kind of the exact diagonalization method applied to a quantum lattice system.
"""
struct EDKind{K} end
@inline EDKind(K::Symbol) = EDKind{K}()
@inline EDKind(object::Union{Hilbert, Internal, OperatorIndex}) = EDKind(typeof(object))
@inline EDKind(::Type{H}) where {H<:Hilbert} = EDKind(valtype(H))
@inline EDKind(::Type{I}) where {I<:Internal} = EDKind(eltype(I))
@inline EDKind(::Type{I}) where {I<:CompositeIndex} = EDKind(indextype(I))
@inline EDKind(::Type{I}) where {I<:Index} = EDKind(internalindextype(I))

"""
    Sector

A sector of the Hilbert space which forms the bases of an irreducible representation of the Hamiltonian of a quantum lattice system.
"""
abstract type Sector end
@inline Base.hash(sector::Sector, h::UInt) = hash(id(sector), h)
@inline Base.:(==)(sector₁::Sector, sector₂::Sector) = isequal(id(sector₁), id(sector₂))
@inline Base.isequal(sector₁::Sector, sector₂::Sector) = isequal(id(sector₁), id(sector₂))

"""
    match(sector₁::Sector, sector₂::Sector) -> Bool

Judge whether two sectors match each other, that is, whether they can be used together as the bra and ket spaces.
"""
@inline Base.match(sector₁::Sector, sector₂::Sector) = false

"""
    sumable(sector₁::Sector, sector₂::Sector) -> Bool

Judge whether two sectors could be direct summed.
"""
@inline sumable(::Sector, ::Sector) = false

"""
    productable(sector₁::Sector, sector₂::Sector) -> Bool

Judge whether two sectors could be direct producted.
"""
@inline productable(::Sector, ::Sector) = false

"""
    broadcast(::Type{Sector}, quantumnumbers::OneAtLeast{Abelian}, hilbert::Hilbert, args...; table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert))) -> NTuple{fieldcount(typeof(quantumnumbers)), Sector}

Construct a set of sectors based on the quantum numbers and a Hilbert space.
"""
@inline function Base.broadcast(::Type{Sector}, quantumnumbers::OneAtLeast{Abelian}, hilbert::Hilbert, args...; table::AbstractDict=Table(hilbert, Metric(EDKind(hilbert), hilbert)))
    return map(quantumnumbers) do quantumnumber
        Sector(quantumnumber, hilbert, args...; table=table)
    end
end

"""
    matrix(op::Operator{V, Tuple{}}, braket::NTuple{2, Sector}, table::AbstractDict, dtype=V) where V -> SparseMatrixCSC{V, Int}

Get the CSC-formed sparse matrix representation of a scalar operator.
"""
@inline matrix(op::Operator{V, Tuple{}}, braket::NTuple{2, Sector}, ::AbstractDict, dtype=V) where V = sparse(one(dtype)*op.value*I, dimension(braket[1]), dimension(braket[2]))

"""
    matrix(ops::Operators, braket::NTuple{2, Sector}, table::AbstractDict, dtype=scalartype(ops)) -> SparseMatrixCSC{dtype, Int}

Get the CSC-formed sparse matrix representation of a set of operators.

Here, `table` specifies the order of the operator indexes.
"""
function matrix(ops::Operators, braket::NTuple{2, Sector}, table::AbstractDict, dtype=scalartype(ops))
    length(ops)==0 && return spzeros(dtype, length(braket[1]), length(braket[2]))
    length(ops)==1 && return matrix(ops[1], braket, table, dtype)
    return matrix(ops[1:length(ops)÷2], braket, table, dtype) + matrix(ops[length(ops)÷2+1:length(ops)], braket, table, dtype)
end

"""
    EDMatrix{M<:SparseMatrixCSC, S<:Sector} <: OperatorPack{M, Tuple{S, S}}

Matrix representation of quantum operators between a ket Hilbert space and a bra Hilbert space.
"""
struct EDMatrix{M<:SparseMatrixCSC, S<:Sector} <: OperatorPack{M, Tuple{S, S}}
    matrix::M
    bra::S
    ket::S
end
@inline parameternames(::Type{<:EDMatrix}) = (:value, :sector)
@inline getcontent(m::EDMatrix, ::Val{:value}) = m.matrix
@inline getcontent(m::EDMatrix, ::Val{:id}) = (m.bra, m.ket)
@inline Base.promote_rule(M::Type{<:EDMatrix}, N::Type{<:Number}) = reparameter(M, :value, reparameter(valtype(M), 1, promote_type(scalartype(M), N)))

"""
    EDMatrix(m::SparseMatrixCSC, sector::Sector)
    EDMatrix(m::SparseMatrixCSC, braket::NTuple{2, Sector})
    EDMatrix(m::SparseMatrixCSC, bra::Sector, ket::Sector)

Construct a matrix representation when
1) the bra and ket Hilbert spaces share the same bases;
2) the bra and ket Hilbert spaces may be different;
3) the bra and ket Hilbert spaces may or may not be the same.
"""
@inline EDMatrix(m::SparseMatrixCSC, sector::Sector) = EDMatrix(m, sector, sector)
@inline EDMatrix(m::SparseMatrixCSC, braket::NTuple{2, Sector}) = EDMatrix(m, braket[1], braket[2])

"""
    EDEigenData{V<:Number, T<:Number, S<:Sector} <: Data

Eigen decomposition in exact diagonalization method.

Compared to the usual eigen decomposition `Eigen`, `EDEigenData` contains a `:sectors` attribute to store the sectors of Hilbert space in which the eigen values and eigen vectors are computed.
Furthermore, given that in different sectors the dimensions of the sub-Hilbert spaces can also be different, the `:vectors` attribute of `EDEigenData` is a vector of vector instead of a matrix.
"""
struct EDEigenData{V<:Number, T<:Number, S<:Sector} <: Data
    values::Vector{V}
    vectors::Vector{Vector{T}}
    sectors::Vector{S}
    function EDEigenData(values::AbstractVector{<:Number}, vectors::AbstractVector{<:AbstractVector{<:Number}}, sectors::AbstractVector{<:Sector})
        @assert length(values)==length(vectors)==length(sectors) "EDEigenData error: mismatched length of values, vectors and sectors."
        new{eltype(values), eltype(eltype(vectors)), eltype(sectors)}(values, vectors, sectors)
    end
end
@inline Base.iterate(data::EDEigenData) = (data.values, Val(:vectors))
@inline Base.iterate(data::EDEigenData, ::Val{:vectors}) = (data.vectors, Val(:sectors))
@inline Base.iterate(data::EDEigenData, ::Val{:sectors}) = (data.sectors, Val(:done))
@inline Base.iterate(data::EDEigenData, ::Val{:done}) = nothing

"""
    count(data::EDEigenData) -> Int

Count the number of eigen value-vector-sector groups contained in an `EDEigenData`.
"""
@inline Base.count(data::EDEigenData) = length(data.values)

"""
    eigen(m::EDMatrix; nev::Int=1, which::Symbol=:SR, tol::Real=1e-12, maxiter::Int=300, v₀::Union{AbstractVector{<:Number}, Int}=dimension(m.bra), krylovdim::Int=max(20, 2*nev+1), verbosity::Int=0) -> EDEigenData

Solve the eigen problem by the restarted Lanczos method provided by the [KrylovKit](https://github.com/Jutho/KrylovKit.jl) package.
"""
@inline function eigen(m::EDMatrix; nev::Int=1, which::Symbol=:SR, tol::Real=1e-12, maxiter::Int=300, v₀::Union{AbstractVector{<:Number}, Int}=dimension(m.bra), krylovdim::Int=max(20, 2*nev+1), verbosity::Int=0)
    @assert m.bra==m.ket "eigen error: eigen decomposition of an `EDMatrix` are only available for those with the same bra and ket Hilbert spaces."
    eigvals, eigvecs = eigsolve(m.matrix, v₀, nev, which, scalartype(m); krylovdim=krylovdim, tol=tol, maxiter=maxiter, ishermitian=true, verbosity=verbosity)
    if length(eigvals) > nev
        eigvals = eigvals[1:nev]
        eigvecs = eigvecs[1:nev]
    end
    return EDEigenData(eigvals, eigvecs, fill(m.bra, length(eigvals)))
end

"""
    eigen(
        ms::OperatorSum{<:EDMatrix};
        nev::Int=1,
        which::Symbol=:SR,
        tol::Real=1e-12,
        maxiter::Int=300,
        v₀::Union{Dict{<:Abelian, <:Union{AbstractVector{<:Number}, Int}}, Dict{<:Sector, <:Union{AbstractVector{<:Number}, Int}}}=Dict(Abelian(m.ket)=>dimension(m.ket) for m in ms),
        krylovdim::Int=max(20, 2*nev+1),
        verbosity::Int=0,
        timer::TimerOutput=edtimer
    ) -> EDEigenData

Solve the eigen problem by the restarted Lanczos method provided by the Arpack package.
"""
@inline function eigen(
    ms::OperatorSum{<:EDMatrix};
    nev::Int=1,
    which::Symbol=:SR,
    tol::Real=1e-12,
    maxiter::Int=300,
    v₀::Union{Dict{<:Abelian, <:Union{AbstractVector{<:Number}, Int}}, Dict{<:Sector, <:Union{AbstractVector{<:Number}, Int}}}=Dict(m.ket=>dimension(m.ket) for m in ms),
    krylovdim::Int=max(20, 2*nev+1),
    verbosity::Int=0,
    timer::TimerOutput=edtimer
)
    @timeit timer "eigen" begin
        values, vectors, sectors = real(scalartype(ms))[], Vector{scalartype(ms)}[], eltype(idtype(eltype(ms)))[]
        isa(v₀, Dict{<:Abelian, <:Union{AbstractVector{<:Number}, Int}}) && (v₀ = Dict(m.ket=>get(v₀, Abelian(m.ket), dimension(m.ket)) for m in ms))
        for m in ms
            @timeit timer string(Abelian(m.ket)) begin
                k = min(dimension(m.ket), nev)
                eigensystem = eigen(m; nev=k, which=which, tol=tol, maxiter=maxiter, v₀=get(v₀, Abelian(m.ket), dimension(m.ket)), krylovdim=krylovdim, verbosity=verbosity)
                for i = 1:k
                    push!(values, eigensystem.values[i])
                    push!(vectors, eigensystem.vectors[i])
                    push!(sectors, eigensystem.sectors[i])
                end
            end
        end
        nev>length(values) && @warn("Requested number ($nev) of eigen values exceeds the maximum available ($(length(values))).")
        perm = sortperm(values)[1:min(nev, length(values))]
        result = EDEigenData(values[perm], vectors[perm], sectors[perm])
    end
    return result
end

"""
    EDMatrixization{D<:Number, T<:AbstractDict, S<:Sector} <: Matrixization

Matrixization of a quantum lattice system on a target Hilbert space.
"""
struct EDMatrixization{D<:Number, T<:AbstractDict, S<:Sector} <: Matrixization
    table::T
    brakets::Vector{Tuple{S, S}}
    EDMatrixization{D}(table::AbstractDict, brakets::Vector{Tuple{S, S}}) where {D<:Number, S<:Sector} = new{D, typeof(table), S}(table, brakets)
end
@inline function Base.valtype(::Type{<:EDMatrixization{D, <:AbstractDict, S}}, ::Type{M}) where {D<:Number, S<:Sector, M<:Operator}
    @assert promote_type(D, valtype(M))==D "valtype error: convert $(valtype(M)) to $D is inexact."
    E = EDMatrix{SparseMatrixCSC{D, Int}, S}
    return OperatorSum{E, idtype(E)}
end
@inline Base.valtype(::Type{R}, ::Type{M}) where {R<:EDMatrixization, M<:Operators} = valtype(R, eltype(M))
function (matrixization::EDMatrixization)(m::Union{Operator, Operators}; kwargs...)
    result = zero(valtype(matrixization, m))
    if isa(m, Operator) || length(m)>0
        for braket in matrixization.brakets
            add!(result, EDMatrix(matrix(m, braket, matrixization.table, scalartype(result)), braket))
        end
    end
    return result
end

"""
    EDMatrixization{D}(table::AbstractDict, sector::S, sectors::S...) where {D<:Number, S<:Sector}
    EDMatrixization{D}(table::AbstractDict, brakets::Vector{Tuple{S, S}}) where {D<:Number, S<:Sector}

Construct a matrixization.
"""
@inline EDMatrixization{D}(table::AbstractDict, sector::S, sectors::S...) where {D<:Number, S<:Sector} = EDMatrixization{D}(table, [(target, target) for target in (sector, sectors...)])

"""
    SectorFilter{S} <: LinearTransformation

Filter the target bra and ket Hilbert spaces.
"""
struct SectorFilter{S} <: LinearTransformation
    brakets::Set{Tuple{S, S}}
end
@inline Base.valtype(::Type{<:SectorFilter}, ::Type{M}) where {M<:OperatorSum{<:EDMatrix}} = M
@inline (sectorfileter::SectorFilter)(m::EDMatrix) = id(m)∈sectorfileter.brakets ? m : EDMatrix(spzeros(scalartype(m), size(value(m))...), id(m))
@inline SectorFilter(sector::Sector, sectors::Sector...) = SectorFilter(map(target->(target, target), (sector, sectors...))...)
@inline SectorFilter(braket::NTuple{2, Sector}, brakets::NTuple{2, Sector}...) = SectorFilter(push!(Set{typeof(braket)}(), braket, brakets...))

"""
    ED{K<:EDKind, L<:Union{AbstractLattice, Nothing}, S<:Generator{<:Operators}, M<:EDMatrixization, H<:CategorizedGenerator{<:OperatorSum{<:EDMatrix}}} <: Frontend

Exact diagonalization method of a quantum lattice system.
"""
struct ED{K<:EDKind, L<:Union{AbstractLattice, Nothing}, S<:Generator{<:Operators}, M<:EDMatrixization, H<:CategorizedGenerator{<:OperatorSum{<:EDMatrix}}} <: Frontend
    lattice::L
    system::S
    matrixization::M
    H::H
    function ED{K}(lattice::Union{AbstractLattice, Nothing}, system::Generator{<:Operators}, matrixization::EDMatrixization) where {K<:EDKind}
        H = matrixization(empty(system))
        new{K, typeof(lattice), typeof(system), typeof(matrixization), typeof(H)}(lattice, system, matrixization, H)
    end
end
@inline kind(ed::ED) = kind(typeof(ed))
@inline kind(::Type{<:ED{K}}) where K = K()
@inline scalartype(::Type{<:ED{<:EDKind, <:AbstractLattice, S}}) where {S<:Generator{<:Operators}} = scalartype(S)
@inline scalartype(::Type{<:Algorithm{F}}) where {F<:ED} = scalartype(F)
@inline function update!(ed::ED; kwargs...)
    if length(kwargs)>0
        update!(ed.system; kwargs...)
        update!(ed.H, ed.matrixization, ed.system; kwargs...)
    end
    return ed
end
@inline Parameters(ed::ED) = Parameters(ed.system)

"""
    prepare!(ed::ED; timer::TimerOutput=edtimer) -> ED
    prepare!(ed::Algorithm{<:ED}) -> Algorithm{<:ED}

Prepare the matrix representation.
"""
@inline function prepare!(ed::ED; timer::TimerOutput=edtimer)
    isempty(ed.H) && @timeit timer "prepare!" begin
        reset!(ed.H, ed.matrixization, ed.system)
        @info "ED prepare complete."
    end
    return ed
end
@inline prepare!(ed::Algorithm{<:ED}) = (prepare!(ed.frontend; timer=ed.timer); ed)

"""
    release!(ed::ED; gc::Bool=true) -> ED
    release!(ed::Algorithm{<:ED}; gc::Bool=true) -> Algorithm{<:ED}

Release the memory source used in preparing the matrix representation. If `gc` is `true`, call the garbage collection immediately.
"""
@inline function release!(ed::ED; gc::Bool=true)
    empty!(ed.H)
    gc && GC.gc()
    return ed
end
@inline release!(ed::Algorithm{<:ED}; gc::Bool=true) = release!(ed.frontend; gc=gc)

"""
    matrix(ed::ED, sectors::Union{Abelian, Sector}...; timer::TimerOutput=edtimer, release::Bool=false) -> OperatorSum{<:EDMatrix}
    matrix(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; release::Bool=false) -> OperatorSum{<:EDMatrix}

Get the sparse matrix representation of a quantum lattice system in the target space.
"""
function matrix(ed::ED; timer::TimerOutput=edtimer, release::Bool=false)
    prepare!(ed; timer=timer)
    @timeit timer "matrix" begin
        result = expand(ed.H)
    end
    release && release!(ed; gc=true)
    return result
end
function matrix(ed::ED, sector::Sector, sectors::Sector...; timer::TimerOutput=edtimer, release::Bool=false)
    prepare!(ed; timer=timer)
    @timeit timer "matrix" begin
        result = expand(SectorFilter(sector, sectors...)(ed.H))
    end
    release && release!(ed; gc=true)
    return result
end
function matrix(ed::ED, quantumnumber::Abelian, quantumnumbers::Abelian...; timer::TimerOutput=edtimer, release::Bool=false)
    sectors = eltype(eltype(ed.matrixization.brakets))[]
    for (bra, ket) in ed.matrixization.brakets
        @assert bra==ket "matrix error: unequal bra and ket Hilbert spaces found."
        Abelian(bra)∈(quantumnumber, quantumnumbers...) && push!(sectors, bra)
    end
    return matrix(ed, sectors...; timer=timer, release=release)
end
function matrix(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; release::Bool=false)
    return matrix(ed.frontend, sectors...; timer=ed.timer, release=release)
end

"""
    eigen(ed::ED, sectors::Union{Abelian, Sector}...; timer::TimerOutput=edtimer, release::Bool=false, kwargs...) -> EDEigenData
    eigen(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; release::Bool=false, kwargs...) -> EDEigenData

Solve the eigen problem by the restarted Lanczos method provided by the Arpack package.
"""
@inline eigen(ed::ED, sectors::Union{Abelian, Sector}...; timer::TimerOutput=edtimer, release::Bool=false, kwargs...) = eigen(matrix(ed, sectors...; timer=timer, release=release); timer=timer, kwargs...)
@inline eigen(ed::Algorithm{<:ED}, sectors::Union{Abelian, Sector}...; release::Bool=false, kwargs...) = eigen(ed.frontend, sectors...; timer=ed.timer, release=release, kwargs...)

"""
    ED(system::Generator{<:Operators}, table::AbstractDict, sectors::OneOrMore{Sector}, dtype::Type{<:Number}=scalartype(system))
    ED(lattice::Union{AbstractLattice, Nothing}, system::Generator{<:Operators}, table::AbstractDict, sectors::OneOrMore{Sector}, dtype::Type{<:Number}=scalartype(system))

Construct the exact diagonalization method for a quantum lattice system.
"""
@inline ED(system::Generator{<:Operators}, table::AbstractDict, sectors::OneOrMore{Sector}, dtype::Type{<:Number}=scalartype(system)) = ED(nothing, system, table, sectors, dtype)
@inline function ED(lattice::Union{AbstractLattice, Nothing}, system::Generator{<:Operators}, table::AbstractDict, sectors::OneOrMore{Sector}, dtype::Type{<:Number}=scalartype(system))
    kind = typeof(EDKind(eltype(eltype(system))))
    matrixization = EDMatrixization{dtype}(table, sectors)
    return ED{kind}(lattice, system, matrixization)
end

"""
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms)
    )
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, table::AbstractDict, sectors::OneOrMore{Sector}=Sector(hilbert; table=table), boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms)
    )

Construct the exact diagonalization method for a quantum lattice system.
"""
@inline function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms)
)
    return ED(lattice, hilbert, terms, Table(hilbert, Metric(EDKind(hilbert), hilbert)), Sector(hilbert), boundary, dtype; neighbors=neighbors)
end
function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, table::AbstractDict, sectors::OneOrMore{Sector}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms)
)
    system = Generator(bonds(lattice, neighbors), hilbert, OneOrMore(terms), boundary, eager; half=false)
    matrixization = EDMatrixization{dtype}(table, OneOrMore(sectors)...)
    return ED{typeof(EDKind(hilbert))}(lattice, system, matrixization)
end

"""
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms)
    )
    ED(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, table::AbstractDict, quantumnumbers::OneOrMore{Abelian}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms)
    )

Construct the exact diagonalization method for a quantum lattice system.
"""
@inline function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms)
)
    return ED(lattice, hilbert, terms, Table(hilbert, Metric(EDKind(hilbert), hilbert)), quantumnumbers, boundary, dtype; neighbors=neighbors)
end
@inline function ED(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, table::AbstractDict, quantumnumbers::OneOrMore{Abelian}, boundary::Boundary=plain, dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms)
)
    return ED(lattice, hilbert, terms, table, broadcast(Sector, OneOrMore(quantumnumbers), hilbert; table=table), boundary, dtype; neighbors=neighbors)
end
