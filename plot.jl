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
using MathTeXEngine
using LaTeXStrings

using HDF5
using DelimitedFiles

# Debug parameters
DEBUG = true
PARTICLE_COUNT = DEBUG ? 10^1 : 10^5
SAMPLES = DEBUG ? 10^2 : 10^5
STEPS = DEBUG ? 25 : 10^2 * 5
FRAMES = DEBUG ? 120 : 120

# Spherical Harmonics parameters
L = 6
T = 12
R = 0.045#0.04541
center = [0.0, 0.0, 0.0]
TypedPolynomials.@polyvar x y z

# MagSphere parameters
N = 86
tDesign = loadTDesign(T, N, R)

figname = "combinedAngle-$(σ₁²)FieldUncertainty-$(σ₂²)-$(σ₃²)"
if !isdir("figures")
  mkdir("figures")
end

verbose(x) = DEBUG ? map(p -> std(p.particles) / mean(p.particles), x) : x

function variance(σ₁², σ₂², σ₃²)
  σ²ᵦ = [0.0 * x + 0.0 * y + 0.0 * z for _ in 1:3]

  # Bᵢᵏ = Rᵏ * bᵏ + Oᵏ
  bᵏ = ustrip(parseMagSphereFile("data/HeadScannerAxisAlignmentReady2.txt")[:, :, 1])
  bᵏ_mc = [Particles{Float64,PARTICLE_COUNT}(0.0) for _ in 1:3, _ in 1:N]
  for i in 1:N
    bᵏ_mc[:, i] = bᵏ[:, i] + [Particles(PARTICLE_COUNT, Normal(0, sqrt((σ₃²[1] * abs(b))^2 + sum(abs2, σ₃²[2])))) for b in bᵏ[:, i]]
  end
  Rᵏ, Oᵏ = calibSensorRotationWithoutZeroMeas(
    sqrt(σ₂²[1]), sqrt(σ₂²[2]), σ₂²[3];
    fnXPlus="data/20mTXPlusCalib.txt",
    fnXMinus="data/20mTXMinusCalib.txt",
    fnYPlus="data/20mTYPlusCalib.txt",
    fnYMinus="data/20mTYMinusCalib.txt",
    fnZPlus="data/20mTZPlusCalib.txt",
    fnZMinus="data/20mTZMinusCalib.txt",
    N=N,
    PARTICLE_COUNT=PARTICLE_COUNT
  )
  println("Rᵏ: ", size(Rᵏ), " - ", verbose(Rᵏ[1:2, :, :]))
  println("Oᵏ: ", size(Oᵏ), " - ", verbose(Oᵏ[1:2, :]))
  println("bᵏ: ", size(bᵏ_mc), " - ", verbose(bᵏ_mc[:, 1:2]))
  Bᵢᵏ_mc = [Particles{Float64,PARTICLE_COUNT}(0.0) for _ in 1:3, _ in 1:N]
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
        σᵧ² = var(γⁱₗₘ.particles)
        σ²ᵦ[i] += σᵧ² * Zₗᵐ * Zₗᵐ
      end
    end
  end

  return σ²ᵦ
end

# σ₁² = 2^2
# σ₂² = ((asin(0.0115) / sqrt(3))^2, (1.1e-3 / sqrt(3))^2, 50e-6)
# σ₃² = (0.05, [3e-6, 24e-6])

σ²ᵦ = variance(2^2, ((asin(0.0115) / sqrt(3))^2, (1.1e-3 / sqrt(3))^2, 50e-6), (0.05, [3e-6, 24e-6])) # 201.9, 261.6, 234.3
# σ²ᵦ = variance(2^2, (0, 0, 0), (0, [0])) # 73.78, 165.9, 112.3
# σ²ᵦ = variance(0, ((asin(0.0115) / sqrt(3))^2, 0, 0), (0, [0])) # 29.25, 24.6, 41.66
# σ²ᵦ = variance(0, (0, (1.1e-3 / sqrt(3))^2, 0), (0, [0])) # 187.2, 183.7, 167.1
# σ²ᵦ = variance(0, (0, 0, 50e-6), (0, [0])) # 15.86, 14.66, 16.38
# σ²ᵦ = variance(0, (0, 0, 0), (0.05, [0])) # 70.5, 85.4, 70.72
# σ²ᵦ = variance(0, (0, 0, 0), (0, [3e-6, 24e-6])) # 0.1, 0.1027, 0.1106
# σ²ᵦ = variance(0, (0, 0, 0), (0, [0])) # 0.1, 0.1027, 0.1106

coeffs = h5read("data/coeffs.h5", "coeffs")
cs = [SphericalHarmonicCoefficients(vec(coeffs[i, 1, :]), R, true) for i in 1:3]
B = [sphericalHarmonicsExpansion(c, x, y, z) for c in cs]
fs = [(ix, iy, iz) -> B[i](x => ix, y => iy, z => iz) for i in 1:3]
fieldᵣ = (ix, iy, iz) -> map(f -> f(ix, iy, iz), fs)

Bₑ = σ²ᵦ
fsₑ = [(ix, iy, iz) -> sqrt(Bₑ[i](x => ix, y => iy, z => iz)) for i in 1:3]
fieldErrorᵣ = (ix, iy, iz) -> map(f -> sqrt(f(ix, iy, iz)), fsₑ)

scales = [1e3, 1e6]

averageFieldErrorStrings = Dict{String,String}()
for (fₑ, dir) in zip(fsₑ, ["x", "y", "z"])
  if DEBUG
    averageError = meanErrorMonteCarlo(fₑ, R, SAMPLES)
  else
    averageError = meanError(fₑ, R)
  end

  averageDirErrorStr = @sprintf("%.4g", averageError * scales[2])
  averageFieldErrorStrings[dir] = averageDirErrorStr
  println("Mean field error in $dir-direction: ", averageDirErrorStr)
end

xs, ys, zs = ntuple(_ -> range(-R, R; length=STEPS), 3)
xsₐ, ysₐ, zsₐ = ntuple(_ -> range(-R, R; length=15), 3)

Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

fig = GLMakie.Figure(size=(1600, 450), resolution=(6400, 1800), fontsize=64)

axs = [Axis3(fig[1, i]; height=Relative(0.9), aspect=:data, elevation=0.3π, azimuth=-0.75π, xlabel="x / mm", ylabel="y / mm", zlabel="z / mm", xlabeloffset=200, ylabeloffset=200, zlabeloffset=200, title=L"x", titlefont=:bold) for i = 1:4]
axs[1].title = "Measured field"
for (i, dir) in enumerate(["x", "y", "z"])
  axs[i+1].title = L"Field error in %$(dir)-component (mean of %$(averageFieldErrorStrings[dir]) $\mu\mathrm{T}$)"
end

function slice_extrema(f, scale)
    vals = Float64[]
    for (plane, get_grid, get_mask) in zip(
            (:xy, :xz, :yz),
            [() -> (repeat(xs', STEPS, 1), repeat(ys, 1, STEPS), fill(0.0, STEPS, STEPS)),
             () -> (repeat(xs', STEPS, 1), fill(0.0, STEPS, STEPS), repeat(zs, 1, STEPS)),
             () -> (fill(0.0, STEPS, STEPS), repeat(ys', STEPS, 1), repeat(zs, 1, STEPS))],
            [() -> (X, Y, Z) -> X.^2 .+ Y.^2 .<= R^2,
             () -> (X, Y, Z) -> X.^2 .+ Z.^2 .<= R^2,
             () -> (X, Y, Z) -> Y.^2 .+ Z.^2 .<= R^2]
        )
        X, Y, Z = get_grid()
        mask = get_mask()(X, Y, Z)
        B = norm.(f.(X, Y, Z)) * scale
        append!(vals, B[mask])
    end
    return extrema(vals)
end

crange_field = slice_extrema(fieldᵣ, scales[1])
crange_error_x = slice_extrema(fsₑ[1], scales[2])
crange_error_y = slice_extrema(fsₑ[2], scales[2])
crange_error_z = slice_extrema(fsₑ[3], scales[2])
crange_error_xz = (min(crange_error_x[1], crange_error_z[1]), max(crange_error_x[2], crange_error_z[2]))

cranges = [crange_field, crange_error_xz, crange_error_y, crange_error_xz]

function plotSlices(ax, f, crange, scale, min_arrow_norm=0.0005)
  # Hilfsfunktion für Slice-Plot und Pfeile
  function slice_surface!(plane, ax, f; arrow_scale=0.4)
    if plane == :xy
      X, Y = [repeat(xs', STEPS, 1), repeat(ys, 1, STEPS)]
      Z = fill(0f0, STEPS, STEPS)
      mask = X .^ 2 .+ Y .^ 2 .<= R^2
      ps = [Point3f(x, y, 0) for x in xsₐ, y in ysₐ if x^2 + y^2 <= R^2]
    elseif plane == :xz
      X, Z = [repeat(xs', STEPS, 1), repeat(zs, 1, STEPS)]
      Y = fill(0f0, STEPS, STEPS)
      mask = X .^ 2 .+ Z .^ 2 .<= R^2
      ps = [Point3f(x, 0, z) for x in xsₐ, z in zsₐ if x^2 + z^2 <= R^2]
    else # :yz
      Y, Z = [repeat(ys', STEPS, 1), repeat(zs, 1, STEPS)]
      X = fill(0f0, STEPS, STEPS)
      mask = Y .^ 2 .+ Z .^ 2 .<= R^2
      ps = [Point3f(0, y, z) for y in ysₐ, z in zsₐ if y^2 + z^2 <= R^2]
    end
    # Feldwerte und Maske
    B = norm.(f.(X, Y, Z)) * scale
    B[.!mask] .= NaN
    surface!(ax, X, Y, Z; color=B, colormap=:viridis, alpha=0.7, colorrange=crange)

    # Feldpfeile
    if min_arrow_norm === nothing
      return
    end
    ns = map(p -> begin
        v = arrow_scale * Vec3f(f(p[1], p[2], p[3]))
        nrm = norm(v)
        nrm < min_arrow_norm ? v * (min_arrow_norm / nrm) : v
      end, ps)
    arrows!(ax, ps, ns, fxaa=true, linecolor=:white, arrowcolor=:white, quality=256,
      linewidth=R / 90, arrowsize=Vec3f(R / 30, R / 30, R / 30), align=:center, transparency=true, alpha=0.7)
  end
  slice_surface!(:xy, ax, f)
  slice_surface!(:xz, ax, f)
  slice_surface!(:yz, ax, f)
end

plotSlices(axs[1], fieldᵣ, cranges[1], scales[1], 0.0005)
plotSlices(axs[2], fsₑ[1], cranges[2], scales[2], nothing)
plotSlices(axs[3], fsₑ[2], cranges[3], scales[2], nothing)
plotSlices(axs[4], fsₑ[3], cranges[4], scales[2], nothing)

labels = [L"$\left|\left|\mathbf{B}\right|\right|_2$ / $\mathrm{mT}$", L"$\left|\left|\mathbf{B}\right|\right|_2$ / $\mu\mathrm{T}$"]
for i in 1:4
  Colorbar(fig[2, i], limits=cranges[i], label=labels[i == 1 ? 1 : 2], width=Relative(0.8), height=50, vertical=false, flipaxis=false)
end

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

  angles = range(-0.75π, 1.25π; length=FRAMES + 1)[1:end-1]

  record(fig, "figures/$figname.gif", 1:FRAMES) do i
    map(ax -> ax.azimuth = angles[i], axs)
  end
end