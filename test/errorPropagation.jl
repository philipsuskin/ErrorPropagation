#Error propagation from the glueing error of the individual sensor to the error of the corrected sesnor postion by a Monte-Carlo Method.
#FF 7.5.25

using MonteCarloMeasurements, Distributions, Manifolds, ManifoldDiff, LinearAlgebra, Manopt, Random, MPIFiles, DelimitedFiles, Statistics

#Transformation form carthesian coordinates to spherical coordinates
function cartesian_to_spherical(v)
    x=v[1]
    y=v[2]
    z=v[3]

    r = sqrt(x^2 + y^2 + z^2)
    θ = acos(z / r)
    φ = atan(y, x)
    return [r, θ, φ]
end

#Transformation form spherical coordinates to carthesian coordinates
function spherical_to_cartesian(v)
    r=v[1]#ustrip(0.04541u"m")
    θ=v[2]
    φ=v[3]

    x = r * sin(θ) * cos(φ)
    y = r * sin(θ) * sin(φ)
    z = r * cos(θ)
    return [x, y, z]
end

function findtDesRotation(estimatedPosKK)
    # define cost funtion ∑ᵢ d(p*tDesᵢ,estimatedᵢ)², where d is the geodesic distance on the sphere S², tDesᵢ is a point from the tdesign positions in the CAD model, estimatedᵢ is a point from sensor estimation by their rotation matrices and p is a rotation matrix
    #We have to reload the tDesign every time, because the optimization has to start with with a Design in the wrong direction like in the real optimization.

    radius = 0.045
    N = 86 #Number Of measurement points
    T = 12 #Design Order. Degree of Polynomial: 6
    tDes = loadTDesign(T, N, radius)

    estimatedPos = reduce(hcat,map(spherical_to_cartesian, eachcol(estimatedPosKK)))'
    println(size(estimatedPos))
    S2 = Manifolds.Sphere(2) #Unit Sphere
    SO3 = SpecialOrthogonal(3) # rotation group

    G(p) = 1 / (2*N) * mapreduce(i->distance(S2, p*tDes.positions[:,i], estimatedPos[i,:])^2, +, 1:N)
    # cost funtion for Manopt.jl
    g(M, p) = G(p)

    # define gradient using automatic diffentiation https://manoptjl.org/stable/tutorials/AutomaticDifferentiation/
    rb_onb_fwdd = ManifoldDiff.TangentDiffBackend(ManifoldDiff.AutoForwardDiff())
    grad_g(M, p) = Manifolds.gradient(M, G, p, rb_onb_fwdd)
    #@info "" check_gradient(SO3, g, grad_g)

    # initial value
    R0 = project(SO3,I(3).*1.0) #rand(SO3)

    # gradient descent https://redoblue.github.io/techblog/2019/02/22/optimization-riemannian-manifolds/
    Rest = gradient_descent(SO3, g, grad_g, R0;
        debug=[:Iteration, (:Change, "|Δp|: %1.9f |"), (:Cost, " F(x): %1.11f | "), "\n", :Stop, 5], stopping_criterion=StopWhenGradientNormLess(1e-6) | StopAfterIteration(100))

    @info "rotation matrix estimate " Rest
    for i=1:length(tDes)
        tDes.positions[:,i] = Rest*tDes.positions[:,i]
    end
    #Wir müssen am Ende alles wiede rin Kugelkoordinaten bringen, damit wir einen Fehler in theta und phi angeben, weil wir ja annehmen, dass der Sensor auf der Kugeloberfläche ist.
    return reduce(hcat,map(cartesian_to_spherical, eachcol(tDes.positions)))[2:3,:]
end

#######################################################################################################################
#Script:

#Initialization

N = 86 #Number of Sensors
estimatedPos = zeros(N,3) #Estimated Positions
calibrationRotation = reshape(readdlm("../RotationCalibration.txt"),N,3,3)

#estimated Postions
for i=1:N
    estimatedPos[i,:] = calibrationRotation[i,:,3]/norm(calibrationRotation[i,:,3])#.*radius, no radius, because we compare it pon the unit sphere. Estimated direction of the sensor position assuming perfect orientation
end

angleStd = 2*pi/360*2 #Assumed angle error in Rad
estimatedPosKK = reduce(hcat,map(cartesian_to_spherical, eachcol(estimatedPos'))) #Estimated Positions in Spherical coordinates
estimatedPosKKError = estimatedPosKK.± [0,2*pi/360*angleStd,2*pi/360*angleStd] # only φ and θ are assumed to have an error

#Since the Optimizer does not support particles as input, we have the workaround by iterating over the particles

numParticles = size(estimatedPosKKError[1,1].particles)[1] #I'll use spherical coordinates for the particles, since this allows us to more accurately model the fact that there is no error in r.
estimatedPosKKDistribution = zeros(numParticles,3,N) #in spherical coordinates
tDesPositionDistribution = zeros(numParticles,2,N) # middle Argument is two since r is not considered

#1: Filling an array with all particles
for i=1:numParticles
    for j=1:N
        for k=1:3
            estimatedPosKKDistribution[i,k,j] = estimatedPosKKError[k,j].particles[i]
        end
    end
end
println(size(estimatedPosKKDistribution[1, :, :]))

#2: For every set in estimatedPos we have to find the corresponding tDes position
for i=1:numParticles
    tDesPositionDistribution[i,:,:] = findtDesRotation(estimatedPosKKDistribution[i,:,:])
    @info i/numParticles*100," % done"
    exit()
end

#3: Calc mean and std of the tDes positions

meantDesPosition = zeros(2,N)
vartDesPosition = zeros(2,8N)

for i=1:2
    for j=1:86
        meantDesPosition[i,j] = mean(tDesPositionDistribution[:,i,j])
        vartDesPosition[i,j] = var(tDesPositionDistribution[:,i,j])
    end
end

#4: Calc covariance matrix
∑ = cov(reshape(tDesPositionDistribution,numParticles,2*N))