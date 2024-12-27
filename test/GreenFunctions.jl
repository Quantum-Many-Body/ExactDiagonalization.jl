# @testset "Partition" begin
#     unitcell = Lattice([0, 0]; vectors=[[1, 0]])
#     cluster = Lattice(unitcell, (4,), ('p',))
#     hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
#     bs = Sector(hilbert, SpinfulParticle(2, 0.0))
#     table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
#     @test length(Partition(:N, table, bs)[1])==length(Partition(:N, table, bs)[2])==2
#     @test Partition(:N, table, bs)[1][1].block[1]==[1,2,3,4]
#     @test Partition(:N, table, bs)[1][2].block[1]==[5,6,7,8]
#     @test [Partition(:N, table, bs)[1][i].block[1] for i in 1:2]==[Partition(:N, table, bs)[2][i].block[1] for i in 1:2]
#     @test Partition(:N, table, bs)[1][1].sector.quantumnumbers[1].N==0
#     @test Partition(:N, table, bs)[1][1].sector.quantumnumbers[2].N==1.0
#     bs = Sector(hilbert, SpinfulParticle(NaN, 0.0))
#     @test [Partition(:A, table, bs)[1][i].block for i in 1:4] == [[[1, 2, 3, 4], [1, 2, 3, 4]],[[5, 6, 7, 8], [5, 6, 7, 8]],[[5, 6, 7, 8], [9, 10, 11, 12]],[[1, 2, 3, 4], [13, 14, 15, 16]]]
#     @test [Partition(:A, table, bs)[2][i].block for i in 1:4] == [[[1, 2, 3, 4], [1, 2, 3, 4]],[[5, 6, 7, 8], [5, 6, 7, 8]],[[9, 10, 11, 12], [5, 6, 7, 8]],[[13, 14, 15, 16], [1, 2, 3, 4]]]
#     @test [Partition(:A, table, bs)[1][i].sector.quantumnumbers[1].Sz for i in 1:4]==[0.5,-0.5,-0.5,0.5]
#     @test [Partition(:A, table, bs)[2][i].sector.quantumnumbers[1].Sz for i in 1:4]==[-0.5,0.5,0.5,-0.5]
# end

# @testset "EDSolver" begin
#     unitcell = Lattice([0, 0]; vectors=[[1, 0]])
#     cluster = Lattice(unitcell, (2,), ('p',))
#     hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
#     bs = BinaryBases(4, 2)
#     t = Hopping(:t, -1.0, 1)
#     U = Hubbard(:U, 0.0)
#     μ = Onsite(:μ, -0.0)
#     referterms = (t, U, μ)
#     neighbors = Neighbors(0=>0.0, 1=>1.0)
#     table = Table(hilbert, Metric(EDKind(hilbert), hilbert)) 
#     referbonds = filter(bond -> isintracell(bond), bonds(cluster, neighbors))
#     refergenerator = OperatorGenerator(referterms, referbonds, hilbert)
#     solver = EDSolver(EDKind(hilbert), Partition(:N, table, bs), refergenerator, bs, table; m=min(150, length(bs)))
#     @test isapprox(solver.gse, -2.0; atol=1e-10)
#     U = Hubbard(:U, 1.0)
#     μ = Onsite(:μ, -0.5)
#     referterms = (t, U, μ)
#     refergenerator = OperatorGenerator(referterms, referbonds, hilbert)
#     solver = EDSolver(EDKind(hilbert), Partition(:N, table, bs), refergenerator, bs, table; m=min(150, length(bs)))
#     @test isapprox(solver.gse, -2.5615528128088285, atol=1e-10)
#     bs = BinaryBases(1:2, 3:4, 0.0)
#     scoupling=Coupling(Index(:, FID(1, 1//2, 1)), Index(:, FID(1, -1//2, 1))) - Coupling(Index(:, FID(1, -1//2, 1)), Index(:, FID(1, 1//2, 1)))
#     s = Pairing(:s, 0.3, 0, scoupling/2)
#     μ = Onsite(:μ, -0.5)
#     referterms = (t, s, μ)
#     refergenerator = OperatorGenerator(referterms, referbonds, hilbert)
#     solver = EDSolver(EDKind(hilbert), Partition(:A, table, bs), refergenerator, bs, table; m=min(150, length(bs)))
#     @test isapprox(solver.gse, -3.112801043562362; atol=1e-10)
# end

# @testset "ClusterNormalGreenFunction" begin
#     unitcell = Lattice([0, 0]; vectors=[[1, 0]])
#     cluster = Lattice(unitcell, (2,), ('p',))
#     hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
#     bs = BinaryBases(4, 2)
#     t = Hopping(:t, -1.0, 1)
#     U = Hubbard(:U, 1.0)
#     μ = Onsite(:μ, -0.5)
#     referterms = (t, U, μ)
#     neighbors = Neighbors(0=>0.0, 1=>1.0)
#     table = Table(hilbert, Metric(EDKind(hilbert), hilbert)) 
#     referbonds = filter(bond -> isintracell(bond), bonds(cluster, neighbors))
#     refergenerator = OperatorGenerator(referterms, referbonds, hilbert)
#     solver = EDSolver(EDKind(hilbert), Partition(:N, table, bs), refergenerator, bs, table; m=min(150, length(bs)))
#     @test isapprox(ClusterGreenFunction(true, :f, solver, 1.0+0.05im),ComplexF64[-4.583831017312281 - 3.921908049638074im 5.054142598734976 + 3.910150260102493im -3.6817080423139325e-15 - 2.8210121881585814e-15im -8.16883166669904e-15 - 6.368055774409478e-15im; 5.0541425987349795 + 3.9101502601025104im -4.583831017312289 - 3.9219080496380645im 8.168831666699048e-15 + 6.36805577440949e-15im 3.681708042313941e-15 + 2.821012188158585e-15im; -3.681708042313931e-15 - 2.8210121881585762e-15im 8.168831666699043e-15 + 6.368055774409467e-15im -4.583831017312292 - 3.921908049638075im 5.054142598734978 + 3.9101502601025016im; -8.168831666699043e-15 - 6.368055774409489e-15im 3.68170804231394e-15 + 2.8210121881585715e-15im 5.054142598734979 + 3.9101502601025033im -4.58383101731228 - 3.921908049638066im], atol=1e-10)
# end

# @testset "ClusterGorkovGreenFunction" begin
#     unitcell = Lattice([0, 0]; vectors=[[1, 0]])
#     cluster = Lattice(unitcell, (2,), ('p',))
#     hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
#     bs = BinaryBases(1:2, 3:4, 0.0)
#     t = Hopping(:t, -1.0, 1)
#     scoupling=Coupling(Index(:, FID(1, 1//2, 1)), Index(:, FID(1, -1//2, 1))) - Coupling(Index(:, FID(1, -1//2, 1)), Index(:, FID(1, 1//2, 1)))
#     s = Pairing(:s, 0.3, 0, scoupling/2)
#     μ = Onsite(:μ, -0.5)
#     referterms = (t, s, μ)
#     neighbors = Neighbors(0=>0.0, 1=>1.0)
#     table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
#     referbonds = filter(bond -> isintracell(bond), bonds(cluster, neighbors))
#     refergenerator = OperatorGenerator(referterms, referbonds, hilbert)
#     solver = EDSolver(EDKind(hilbert), Partition(:A, table, bs), refergenerator, bs, table; m=min(150, length(bs)))
#     @test isapprox(ClusterGreenFunction(false, :f, solver, 1.0+0.05im), ComplexF64[1.3071188002247869 - 0.13712724863974438im -0.9339754642130047 + 0.12767796268159015im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.11186366185388817 - 0.0421899008892926im -0.3340943140960254 + 0.025636407239971216im; -0.9339754642130046 + 0.1276779626815901im 1.3071188002247853 - 0.13712724863974426im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im -0.33409431409602514 + 0.025636407239971154im 0.11186366185388819 - 0.04218990088929258im; 0.0 + 0.0im 0.0 + 0.0im 1.307118800224786 - 0.13712724863974432im -0.9339754642130047 + 0.1276779626815901im -0.1118636618538878 + 0.04218990088929258im 0.33409431409602547 - 0.025636407239971154im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im -0.9339754642130049 + 0.12767796268159015im 1.3071188002247864 - 0.13712724863974435im 0.33409431409602497 - 0.025636407239971196im -0.11186366185388848 + 0.04218990088929258im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im -0.11186366185388798 + 0.04218990088929256im 0.3340943140960251 - 0.025636407239971168im -0.5472977542357593 - 0.10685087000424552im -1.3018654321738403 - 0.06813335244712346im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.33409431409602536 - 0.025636407239971133im -0.11186366185388803 + 0.042189900889292545im -1.3018654321738405 - 0.0681333524471235im -0.5472977542357611 - 0.1068508700042456im 0.0 + 0.0im 0.0 + 0.0im; 0.11186366185388798 - 0.04218990088929255im -0.3340943140960252 + 0.02563640723997113im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im -0.547297754235761 - 0.10685087000424563im -1.30186543217384 - 0.06813335244712344im; -0.3340943140960252 + 0.025636407239971164im 0.1118636618538879 - 0.04218990088929256im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im -1.3018654321738403 - 0.06813335244712349im -0.5472977542357593 - 0.1068508700042455im], atol=1e-10)
# end
