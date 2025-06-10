using TypedPolynomials
using SphericalHarmonicExpansions
using MPIFiles
include("utils/plotMagneticField.jl")
include("utils/computeStatistics.jl")
include("utils/coordinateConvert.jl")
include("utils/tDesRotationOptim.jl")
include("utils/rotationOffsetDef.jl")

using MonteCarloMeasurements
using Random
Random.seed!(1234)

using GLMakie
using Printf

using HDF5
using DelimitedFiles

# Debug parameters
DEBUG = true
PARTICLE_COUNT = DEBUG ? 10^1 : 10^3
SAMPLES        = DEBUG ? 10^2 : 10^5
STEPS          = DEBUG ? 25   : 10^2
FRAMES         = DEBUG ? 120  : 120

# Spherical Harmonics parameters
L = 6
T = 12
R = 0.045#0.04541
center = [0.0, 0.0, 0.0]
TypedPolynomials.@polyvar x y z

# MagSphere parameters
N = 86
tDesign = loadTDesign(T, N, R)
Bᵢᵏ = [0.0008242248385510878 0.003964926641491151 0.004239623554035758 -0.004221702521378727 0.002922673740449898 -0.00442283700569307 0.0025389800099434234 0.002372779178415687 -0.0009875425167300033 0.003593245778107923 0.0004246233222551863 -0.00358136547826023 -0.004018411382764822 0.00023211701354191298 -0.0025998718812014878 -0.0029556989277163236 0.004279822581142647 -0.0029977184040608403 0.002261672827204327 0.0015208902379443032 -0.004466059469644102 0.003974235520800876 -0.003996132081952711 0.004222976437049262 -0.002703918152234829 0.0003585930393082594 -0.0005158697756834833 0.0016517296277109392 0.0034458062963645887 -0.0016324435355875328 -0.0019240527822228432 0.0034653975139929717 -0.0027289449326731077 -0.0025479396106933244 0.0010392557956691944 -0.0014656265035828152 -0.0004374941824165771 -0.004003112704209233 -0.0020533187126085674 3.941283008208801e-5 -0.003832042536410438 -0.0035111068326308154 -0.003932570359311868 -0.004665665945328731 -0.0020376003731689546 0.004406109655418154 0.0017030555709185873 -0.0004000934924814548 -0.0022363101980310413 0.0033005311791879277 -0.0005501794037353092 -0.002762944925544793 0.004519807701598589 7.991278258183186e-5 0.001098924367581863 0.003229758239421761 -0.0038803325723478343 0.0031990728814917263 0.003979267571991667 0.0003154519159004647 -0.000845667530845637 -0.004315143247951074 -0.0029958866625489377 0.0014477549593691057 0.002173302278142039 0.0010437938999512284 0.004370066495990172 0.002518951174065922 -0.004599590386040057 0.0019843370459428043 0.004415936243894271 0.002619944807271752 -0.0011924430278803095 -0.0007303768778318427 -0.0030308339056001714 -0.004143711143024598 -0.0004599453741761999 0.004692097844740604 0.002856689871200028 0.0009362411441672581 0.002751974506155183 -0.004644383524316851 0.004029597440542507 0.0037698353198721266 -0.001130700138271112 -0.00146757133363278; -0.001422505156103913 0.0012909749932881955 -0.0015174028924531844 -0.005046126318407465 -0.00695188102225077 -0.005133168101738214 0.0003898122553420483 0.0013661453227310108 0.00866810065568517 -0.0004928132879288841 -0.010233079103895107 0.008182200376603645 0.0005267870918914806 -0.0017295625073466534 -0.003645682177339757 -0.006823789477960583 0.005170118549904164 0.006643839091189562 0.01083254398809983 0.002797741730698111 0.0029613028719156923 0.003186338427434464 0.0013113713148452976 -0.005667475415702493 -0.0008868369889067696 0.002536857306593447 0.002413102965778229 0.00838215385902526 -0.008896912944653692 0.011238576522162633 0.004238848171293741 -0.002121908965646727 0.0017041652251945278 0.010190356192862493 0.009472013915281739 -0.0028362980381108215 -0.009539883164734335 0.006847635757910089 0.0032219393435802845 -0.011850552677453104 0.003367128597595079 0.004858733433862842 -0.0023503875012530016 0.0010684323467591652 -0.010764947172237522 0.004029222493349249 0.005913496809336942 0.005666089480861844 -0.009993387950873834 -0.005807342782775442 -0.0032861713835598684 -0.004722280924981622 0.0016212520789566818 0.00597431373107916 0.00021026726227654872 0.006290329098500259 -0.008404035770586902 0.0031712549242936478 -0.0033991828393503034 0.012059507557978345 -0.0063450235597990855 0.004599541524982929 -0.007766431928584298 -0.010911761402992923 -0.002903348305905621 -0.0060722102026282785 -0.0054112401338240294 -0.0030396521055055053 -0.0009480235466411624 -0.008038205008259719 0.00027306758024294385 0.004992768146144821 0.0005898432744930739 -0.007364219256071376 -0.00022084007111278169 -0.0031653834986572664 0.01076022110141348 -0.0020234885715311774 -0.009745590283583475 -0.00490944523163295 0.010172150610562447 -0.0023525997078624856 0.008316397709236154 0.007084429910859166 0.0003226412115712756 0.007207431655358371; 0.00548715900708086 -0.0020669933727901456 -0.0013575124091092436 -0.0013519303559638282 0.003159292045750283 0.001216375122722514 -0.003402177091909373 0.004819420176843166 -0.003577437704010522 0.003590975108492883 0.002920648828342123 -0.0009309763980577656 -0.0019906102109523934 -0.004551675733676274 0.004546495181863615 0.0032093404841624616 -0.0012147650801252918 -0.00286896039906897 0.0016181578239549798 -0.004206474048332333 0.0013270324155441751 0.0029949916470360578 0.0030509873633589816 0.001004461824895397 -0.003737150486094087 0.005491881723811408 -0.004366696735096915 0.003850666676414969 -0.0008765891191295241 -0.0014418331332282932 -0.0038733594637194073 -0.002747294958958021 -0.0035317137395110385 0.0013364779907299594 -0.0032558916788101478 -0.004323202396415643 -0.003637039374638451 0.0017532266899562944 0.004897784360454927 0.00039568029812742274 -0.002187513170166626 0.0034367714621784786 -0.0024492089025339794 -0.0002688137601878344 -0.0014950287297271344 0.0009914975026977784 -0.004036704111294658 -0.00430223987382985 0.0018069795262874854 -0.002854356321001932 0.005287932532470934 -0.003551689302695483 -0.0002731599254221529 0.0050197948448964125 -0.004271930244014909 -0.002648396835722293 3.6649695454134284e-5 -0.0031460989683534746 0.002650300096687457 -0.000329287365640715 -0.004351482560575305 -0.0005499148281325008 -0.002810947291934631 -0.0018908418169936373 -0.0040099591060904875 0.0046657489479887205 -0.0010219430472451238 0.004543794510082797 0.0015665228446379092 -0.0035405377164822445 0.0016163091315399615 0.0042733773166132765 -0.004280502374382785 0.004323466541264433 0.004317787463509609 0.0029140683185393443 0.002778751710415505 0.0006391105755393942 0.0014398374621882664 -0.00447493615839154 -0.0017013147938608677 -0.0003817380101945683 9.326615433796656e-5 0.0023876993473669903 0.005376023729917014 0.004100459067230261]

σ²ᵦ = [0.0*x + 0.0*y + 0.0*z for _ in 1:3]

σ₁² = 2^2
σ₂² = (0.0115, (1.1e-3/sqrt(3))^2)
σ₃² = (0.05, [3e-6, 24e-6])
figname = "combinedAngle-$(σ₁²)FieldUncertainty-$(σ₂²)-$(σ₃²)"
if !isdir("figures")
  mkdir("figures")
end

verbose(x) = DEBUG ? map(p -> std(p.particles) / mean(p.particles), x) : x

# Bᵢᵏ = Rᵏ * bᵏ + Oᵏ
bᵏ = ustrip(parseMagSphereFile("data/HeadScannerAxisAlignmentReady2.txt")[:, :, 1])
bᵏ_mc = [Particles{Float64, PARTICLE_COUNT}(0.0) for _ in 1:3, _ in 1:N]
for i in 1:N
  bᵏ_mc[:, i] = bᵏ[:, i] + [Particles(PARTICLE_COUNT, Normal(0, σ₃²[1] * abs(b) + sum(σ₃²[2]))) for b in bᵏ[:, i]]
end
Rᵏ, Oᵏ = calibSensorRotationWithoutZeroMeas(
  asin(σ₂²[1]) / sqrt(3), sqrt(σ₂²[2]);
  fnXPlus = "data/20mTXPlusCalib.txt",
  fnXMinus = "data/20mTXMinusCalib.txt",
  fnYPlus = "data/20mTYPlusCalib.txt",
  fnYMinus = "data/20mTYMinusCalib.txt",
  fnZPlus = "data/20mTZPlusCalib.txt",
  fnZMinus = "data/20mTZMinusCalib.txt",
  N = N,
  PARTICLE_COUNT = PARTICLE_COUNT
)
println("Rᵏ: ", size(Rᵏ), " - ", verbose(Rᵏ[1:2, :, :]))
println("Oᵏ: ", size(Oᵏ), " - ", verbose(Oᵏ[1:2, :]))
println("bᵏ: ", size(bᵏ_mc), " - ", verbose(bᵏ_mc[:, 1:2]))
Bᵢᵏ_mc = [Particles{Float64, PARTICLE_COUNT}(0.0) for _ in 1:3, _ in 1:N]
for i in 1:N
  Bᵢᵏ_mc[:, i] = Rᵏ[i, :, :] * bᵏ_mc[:, i] + Oᵏ[i, :]
end
println("Bᵢᵏ_mc: ", size(Bᵢᵏ_mc), " - ", verbose(Bᵢᵏ_mc[:, 1:2]))

rs = [Rᵏ[i, :, 3] / norm(Rᵏ[i, :, 3]) for i in 1:N]
println("rs: ", size(rs), " - ", verbose(rs[1]))
ps = reduce(hcat, map(cartesian_to_spherical, rs))
println("ps: ", size(ps), " - ", verbose(ps[:, 1:2]))
ps_mc = reduce(hcat, [ps[:, i] + [0, deg2rad(Particles(PARTICLE_COUNT, Normal(0, sqrt(σ₁²)))), deg2rad(Particles(PARTICLE_COUNT, Normal(0, sqrt(σ₁²)))) / sin(ps[2, i] == 0 ? 1e-9 : ps[2, i])] for i in 1:N])
println("ps_mc: ", size(ps_mc), " - ", verbose(ps_mc[:, 1:2]))
rs_mc = reduce(hcat, map(spherical_to_cartesian, eachcol(ps_mc)))
println("rs_mc: ", size(rs_mc), " - ", verbose(rs_mc[:, 1:2]))

rs_opt = [Particles(PARTICLE_COUNT) for _ in 1:3, _ in 1:N]
for i in 1:PARTICLE_COUNT
  positions = map(p -> p.particles[i], rs_mc)
  positions_opt = findtDesRotation(tDesign, positions)

  for j in 1:3
    for k in 1:N
      rs_opt[j, k].particles[i] = positions_opt[j, k]
    end
  end
end
rs_opt = [rs_opt[:, i] for i in 1:N]
println("rs_opt: ", size(rs_opt), " - ", verbose(rs_opt[1]))

for l in 0:L
  for m in -l:l
    Zₗᵐ = SphericalHarmonicExpansions.zlm(l, m, x, y, z)
    fz = (r) -> Zₗᵐ(x => r[1], y => r[2], z => r[3])
    rs_optᵣ = map(fz, rs_opt)

    for i in 1:3
      γⁱₗₘ = (2l + 1) / (R^l * N) * sum(Bᵢᵏ_mc[i, :] .* rs_optᵣ)
      σᵧ = std(γⁱₗₘ.particles)
      σ²ᵦ[i] += σᵧ * Zₗᵐ * Zₗᵐ
    end
  end
end

coeffs = h5read("data/coeffs.h5", "coeffs")
cs = [SphericalHarmonicCoefficients(vec(coeffs[i, 1, :]), R, true) for i in 1:3]
B = [sphericalHarmonicsExpansion(c, x, y, z) for c in cs]
fs = [(ix, iy, iz) -> B[i](x => ix, y => iy, z => iz) for i in 1:3]
fieldᵣ = (ix, iy, iz) -> map(f -> f(ix, iy, iz), fs)

Bₑ = σ²ᵦ
fsₑ = [(ix, iy, iz) -> Bₑ[i](x => ix, y => iy, z => iz) for i in 1:3]
fieldErrorᵣ = (ix, iy, iz) -> map(f -> f(ix, iy, iz), fsₑ)

scales = [1e3, 1e6]

averageFieldErrorStrings = Dict{String, String}()
for (fₑ, dir) in zip(fsₑ, ["x", "y", "z"])
  averageError = meanErrorMonteCarlo(fₑ, R, SAMPLES)
  # averageError = meanError(fₑ, R)

  averageDirErrorStr = @sprintf("%.4g", averageError * scales[2])
  averageFieldErrorStrings[dir] = averageDirErrorStr
  println("Mean field error in $dir-direction: ", averageDirErrorStr)
end

xs, ys, zs = ntuple(_ -> range(-R, R; length=STEPS), 3)
xsₐ, ysₐ, zsₐ = ntuple(_ -> range(-R, R; length=20), 3)

fig = GLMakie.Figure(size=(2200, 500), resolution=(8800, 2000), fontsize=56)

axs = [Axis3(fig[1, i == 1 ? i : i+1]; height=Relative(0.9), aspect=:data, elevation=0.25π, azimuth=-0.75π, xlabel="x / mm", ylabel="y / mm", zlabel="z / mm", xlabeloffset=200, ylabeloffset=200, zlabeloffset=200) for i=1:4]
axs[1].title = "Measured field"
for (i, dir) in enumerate(["x", "y", "z"])
  axs[i+1].title = "Field error in $dir-direction\n(ME: $(averageFieldErrorStrings[dir]) μT)"
end

cranges = [
  extrema(norm.(fieldᵣ.(xs, ys, zs))) .* scales[1],
  extrema(Iterators.flatten(norm.(f.(xs, ys, zs)) for f in fsₑ)) .* scales[2],
]

function plotSlices(ax, f, crange, scale, min_arrow_norm=0.002)
    # Hilfsfunktion für Slice-Plot und Pfeile
    function slice_surface!(plane, ax, f; arrow_scale=0.3)
        if plane == :xy
            X, Y = [repeat(xs', STEPS, 1), repeat(ys, 1, STEPS)]
            Z = fill(0f0, STEPS, STEPS)
            mask = X.^2 .+ Y.^2 .<= R^2
            ps = [Point3f(x, y, 0) for x in xsₐ, y in ysₐ if x^2 + y^2 <= R^2]
        elseif plane == :xz
            X, Z = [repeat(xs', STEPS, 1), repeat(zs, 1, STEPS)]
            Y = fill(0f0, STEPS, STEPS)
            mask = X.^2 .+ Z.^2 .<= R^2
            ps = [Point3f(x, 0, z) for x in xsₐ, z in zsₐ if x^2 + z^2 <= R^2]
        else # :yz
            Y, Z = [repeat(ys', STEPS, 1), repeat(zs, 1, STEPS)]
            X = fill(0f0, STEPS, STEPS)
            mask = Y.^2 .+ Z.^2 .<= R^2
            ps = [Point3f(0, y, z) for y in ysₐ, z in zsₐ if y^2 + z^2 <= R^2]
        end
        # Feldwerte und Maske
        B = norm.(f.(X, Y, Z)) * scale
        B[.!mask] .= NaN
        surface!(ax, X, Y, Z; color=B, colormap=:viridis, transparency=true, alpha=0.7, colorrange=crange)
        # Feldpfeile
        ns = map(p -> begin
            v = arrow_scale * Vec3f(f(p[1], p[2], p[3]))
            nrm = norm(v)
            nrm < min_arrow_norm ? v * (min_arrow_norm / nrm) : v
        end, ps)
        arrows!(ax, ps, ns, fxaa=true, linecolor=:black, arrowcolor=:black, quality=256,
            linewidth=R/90, arrowsize=Vec3f(R/40, R/40, R/40), align=:center, transparency=true, alpha=0.7)
    end
    slice_surface!(:xy, ax, f)
    slice_surface!(:xz, ax, f)
    slice_surface!(:yz, ax, f)
end

plotSlices(axs[1], fieldᵣ, cranges[1], scales[1])
plotSlices(axs[2], fsₑ[1], cranges[2], scales[2])
plotSlices(axs[3], fsₑ[2], cranges[2], scales[2])
plotSlices(axs[4], fsₑ[3], cranges[2], scales[2])

Colorbar(fig[1, 2], limits=cranges[1], label="||B||₂ / mT", height=Relative(0.8), width=20)
Colorbar(fig[1, 6], limits=cranges[2], label="||B||₂ / μT", height=Relative(0.8), width=20)

save("figures/$figname.png", fig)

if false
  for ax in axs
    ax.viewmode = :fit
    ax.tellwidth = false
    ax.tellheight = false
    # ax.width = 400
    ax.height = Relative(0.9)
    ax.protrusions = 0
    attributes = ["ticksvisible", "labelvisible", "ticklabelsvisible", "gridvisible", "spinesvisible"]
    for attr in attributes
      for dir in [:x, :y, :z]
        setproperty!(ax, Symbol(dir, attr), false)
      end
    end
  end

  angles = range(-0.75π, 1.25π; length=FRAMES+1)[1:end-1]

  record(fig, "figures/$figname.gif", 1:FRAMES) do i
    map(ax -> ax.azimuth = angles[i], axs)
  end
end