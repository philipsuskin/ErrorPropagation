using MonteCarloMeasurements
include("coordinateConvert.jl")

using DelimitedFiles
using Unitful

function parseMagSphereFileV1(filename::String)
    return reshape(readdlm(filename)[:,1:2:end], :,3,86)
end

function parseMagSphereFileV2(filename::String)
    return reshape(readdlm(filename)[:,1],3,86,:)*u"T"
end

function parseMagSphereFile(filename::String)
    nCols = size(readdlm(filename),2)
    if nCols == 2
        return parseMagSphereFileV2(filename)
    elseif nCols == 3 * 2 * 86
        return parseMagSphereFileV1(filename)
    else
        @error "Can't parse MagSphere file."
    end
end

function calibSensorRotationWithoutZeroMeas(
    σₐ, σᵣ;
    fnXPlus::String, fnXMinus::String,
    fnYPlus::String, fnYMinus::String,
    fnZPlus::String, fnZMinus::String,
    N::Int, PARTICLE_COUNT::Int)

    #To be found:
    # rotationCalibration = fill(Particles{Float64, PARTICLE_COUNT}(0.0), N,3,3)
    # offsetCalibration = fill(Particles{Float64, PARTICLE_COUNT}(0.0),N,3)
    rotationCalibration = [Particles{Float64, PARTICLE_COUNT}(0.0) for _ in 1:N, _ in 1:3, _ in 1:3]
    offsetCalibration = [Particles{Float64, PARTICLE_COUNT}(0.0) for _ in 1:N, _ in 1:3]
    
    #Load data
    calibXPlus  = parseMagSphereFile(fnXPlus)
    calibXMinus = parseMagSphereFile(fnXMinus)
    calibYPlus  = parseMagSphereFile(fnYPlus)
    calibYMinus = parseMagSphereFile(fnYMinus)
    calibZPlus  = parseMagSphereFile(fnZPlus)
    calibZMinus = parseMagSphereFile(fnZMinus)
    numOfMeas   = size(calibXPlus[:,1,1])
    @info "Number Of measurements: " numOfMeas

    #1. Calc all mean values for the measurements:
    # Measurement = fill(Particles{Float64, PARTICLE_COUNT}(0.0), 6, N, 3)
    Measurement = [Particles{Float64, PARTICLE_COUNT}(0.0) for _ in 1:6, _ in 1:N, _ in 1:3]
    MeasurementStd = zeros(6,N,3)

    for component=1:3
        for i = 1:N
            #Error of the mean value
            MeasurementStd[1,i,component] = std(calibXMinus[:,component,i])/sqrt(numOfMeas[1])
            MeasurementStd[2,i,component] = std(calibXPlus[:,component,i])/sqrt(numOfMeas[1])
            MeasurementStd[3,i,component] = std(calibYMinus[:,component,i])/sqrt(numOfMeas[1])
            MeasurementStd[4,i,component] = std(calibYPlus[:,component,i])/sqrt(numOfMeas[1])
            MeasurementStd[5,i,component] = std(calibZMinus[:,component,i])/sqrt(numOfMeas[1])
            MeasurementStd[6,i,component] = std(calibZPlus[:,component,i])/sqrt(numOfMeas[1])
            
            #Mean Values
            Measurement[1,i,component] = mean(calibXMinus[:,component,i]) + Particles(PARTICLE_COUNT, Normal(0, MeasurementStd[1,i,component]))
            Measurement[2,i,component] = mean(calibXPlus[:,component,i]) + Particles(PARTICLE_COUNT, Normal(0, MeasurementStd[2,i,component]))
            Measurement[3,i,component] = mean(calibYMinus[:,component,i]) + Particles(PARTICLE_COUNT, Normal(0, MeasurementStd[3,i,component]))
            Measurement[4,i,component] = mean(calibYPlus[:,component,i]) + Particles(PARTICLE_COUNT, Normal(0, MeasurementStd[4,i,component]))
            Measurement[5,i,component] = mean(calibZMinus[:,component,i]) + Particles(PARTICLE_COUNT, Normal(0, MeasurementStd[5,i,component]))
            Measurement[6,i,component] = mean(calibZPlus[:,component,i]) + Particles(PARTICLE_COUNT, Normal(0, MeasurementStd[6,i,component]))
        end
    end

    #2: Filling up A at the correct positions with measurement values
    #A = zeros(N,6*3,12) #6*3: we have 6 Measurements of 3 field components
    # A = fill(Particles{Float64, PARTICLE_COUNT}(0.0), N, 6*3, 12)
    A = [Particles{Float64, PARTICLE_COUNT}(0.0) for _ in 1:N, _ in 1:6*3, _ in 1:12]
    
    for i= 1:N
        global Messung = 1
        for Meas=1:3:18
            for zaehler=0:2
                A[i,Meas+zaehler,zaehler*3+1:zaehler*3+3] = Measurement[Messung,i,:]
            end
            global Messung = Messung + 1
        end
        #ones for Offset Values
        for meas=0:5
        A[i,1+3*meas:3+3*meas,10:12] = -Matrix(1.0I, 3, 3)
        end
    end

    #Kalibmessung Sekels LakeShore: [0.244 -0.025 -20.08]
    calibLakeShore = norm([0.244 -0.025 -20.08].*10^-3)
    
    #3: Fill b with actual Values
    b = [-calibLakeShore 0 0 calibLakeShore 0 0 0 -calibLakeShore 0 0 calibLakeShore 0 0 0 -calibLakeShore 0 0 calibLakeShore]'
    # b_mc = fill(Particles{Float64, PARTICLE_COUNT}(0.0), length(b))
    b_mc = [Particles{Float64, PARTICLE_COUNT}(0.0) for _ in 1:length(b)]
    for i in 1:3:length(b)
        r = b[i:i+2]
        p = cartesian_to_spherical(r)
        p_mc = p + [Particles(PARTICLE_COUNT, Normal(0, σᵣ)), Particles(PARTICLE_COUNT, Normal(0, σₐ / sqrt(2))), Particles(PARTICLE_COUNT, Normal(0, σₐ / sqrt(2))) / sin(p[2] == 0 ? 1e-9 : p[2])]
        r_mc = spherical_to_cartesian(p_mc)
        b_mc[i:i+2] = copy(r_mc)
    end

    #4: Fill up rotation calibration matrix
    for i=1:N
        @unsafe sol = \(A[i,:,:],b_mc)
        rotationCalibration[i,:,:] = reshape(sol[1:9],3,3)'
        offsetCalibration[i,:] = sol[10:12]
    end

    return rotationCalibration, offsetCalibration
end