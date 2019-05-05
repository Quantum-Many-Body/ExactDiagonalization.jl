using Test
using ExactDiagonalization.FED
using QuantumLattices
import QuantumLattices.Mathematics.AlgebraOverFields: idtype
using Arpack: eigs

@testset "GBasis" begin
    basis=GBasis{UInt}(2)
    @test basis|>kind==basis|>typeof|>kind==:g
    @test basis|>idtype==basis|>typeof|>idtype==Int
    @test dimension(basis)==2^2
    @test id(basis)==2
    @test basis[4]==3
    @test searchsortedfirst(basis,3)==4
    @test string(basis)=="GBasis(2):\n  0\n  1\n  10\n  11\n"
    @test repr(basis)=="GBasis(2)"
end

@testset "matrix" begin
    lattice=Lattice("L2P",[Point(PID(1),(0.0,)),Point(PID(2),(1.0,))])
    config=IDFConfig{Fock}(pid->Fock(atom=pid.site%2,norbital=1,nspin=2,nnambu=2),lattice.pids)
    table=Table(config,usualfockindextotuple)
    t=Hopping{'F'}(:t,1.0,1)
    U=Hubbard{'F'}(:U,1.0)
    μ=Onsite{'F'}(:μ,-0.5)
    g=Generator((t,U,μ),Bonds(lattice),config,table,false)
    m=matrix(expand(g),GBasis{UInt}(4))
    @test eigs(m,which=:SR,ritzvec=false)[1][1]≈-2.5615528128088303
end
