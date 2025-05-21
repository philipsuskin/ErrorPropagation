using PyPlot

"""
    fignumber = plotMagneticField(func::Vector{Function}, R::T;
			   intersection::Vector{T}=[0.0,0.0,0.0], 
			   center::Vector{T}=[0.0,0.0,0.0],
			   fignumber::Int=1, plane::Int=0, fontsize::Real=14,
			   plotSphere::Bool=true, vmin::T=0.0, vmax::T=0.0,
			   useMilli::Bool=false, yMainAxis::Bool=false) where T <: Real
*Description:*  Plot of the magnetic field\\
 \\
*Input:*
- `func`         - Function of the expansion of the field (Array of length 3)
- `R`            - radius of the measured sphere
**kwargs:**
- `intersection` - intersection of the plotted planes (default: [0.0,0.0,0.0])
- `center`       - center of the measured sphere (default: [0.0,0.0,0.0])
- `fignumber`    - Number of the figure (Integer, default: 1)
- `plane`:\\
        plane = 0 - Plot all 3 planes (default)\\
        plane = 1 - Plot yz-plane\\
        plane = 2 - Plot xz-plane\\
        plane = 3 - Plot xy-plane (or yx-plane)
- `fontsize`   - Fontsize (default: 14)
- `plotSphere` - Boolean (default: true)\\
                 true -> plot boundary of convergence area
- `vmin, vmax` - Minimum and maximum for imshow (default: 0)\\
                 (if vmin == vmax values are not set manually)
- `useMilli`   - Boolean (default: false)\\
                 true -> convert meter and tesla to millimeter and millitesla
- `yMainAxis`  - Boolean (default: false)\\
		 true -> plot yx-plane instead of xy-plane
*Output:*
- `fignumber` = fignumber+1
"""
function plotMagneticField(func::Vector{F}, R::T;
			   intersection::Vector{T}=[0.0,0.0,0.0], 
			   center::Vector{T}=[0.0,0.0,0.0],
			   fignumber::Int=1, plane::Int=0, fontsize::Real=14,
			   plotSphere::Bool=true, vmin::Float64=0.0, vmax::T=0.0,
			   useMilli::Bool=false, yMainAxis::Bool=false) where {T <: Real, F <: Function}

    # convert coordinates of the plot to mm if useMilli = true
    convMilli = useMilli ? 1000 : 1;

    # Cartesian coordinates of the grid
    Nx = range(-R-center[1],stop=R-center[1],length=301);
    Ny = range(-R-center[2],stop=R-center[2],length=301);
    Nz = range(-R-center[3],stop=R-center[3],length=301);

    # calculate norm of the field at each grid position
    Norm = zeros(length(Nx),length(Nx),3);
    xx = zeros(length(Nx),length(Nx),3)
    yy = zeros(length(Nx),length(Nx),3)
    zz = zeros(length(Nx),length(Nx),3)
    for i = 1:length(Nx)
        for j = 1:length(Nx)
            xx[i,j,1] = func[1](intersection[1],Ny[i],Nz[j])
            yy[i,j,1] = func[2](intersection[1],Ny[i],Nz[j])
            zz[i,j,1] = func[3](intersection[1],Ny[i],Nz[j])
            Norm[i,302-j,1] = sqrt(xx[i,j,1]^2+yy[i,j,1]^2+zz[i,j,1]^2).*convMilli
            xx[i,j,2] = func[1](Nx[i],intersection[2],Nz[j])
            yy[i,j,2] = func[2](Nx[i],intersection[2],Nz[j])
            zz[i,j,2] = func[3](Nx[i],intersection[2],Nz[j])
            Norm[i,302-j,2] = sqrt(xx[i,j,2]^2+yy[i,j,2]^2+zz[i,j,2]^2).*convMilli
            xx[i,j,3] = func[1](Nx[i],Ny[j],intersection[3])
            yy[i,j,3] = func[2](Nx[i],Ny[j],intersection[3])
            zz[i,j,3] = func[3](Nx[i],Ny[j],intersection[3])
            Norm[i,302-j,3] = sqrt(xx[i,j,3]^2+yy[i,j,3]^2+zz[i,j,3]^2).*convMilli
        end
    end
    if yMainAxis
      # change x- and y-axis of third plot
      xx[:,:,3] = permutedims(xx[:,:,3],[2,1])
      yy[:,:,3] = permutedims(yy[:,:,3],[2,1])
      zz[:,:,3] = permutedims(zz[:,:,3],[2,1])
      Norm[:,:,3] = reverse(reverse(permutedims(Norm[:,:,3],[2,1]),dims=1),dims=2)
    end

    # nodes for quiver
    xx = xx[1:30:end,1:30:end,:].*convMilli
    yy = yy[1:30:end,1:30:end,:].*convMilli
    zz = zz[1:30:end,1:30:end,:].*convMilli

    # for quiver:
    NNx = Nx[1:30:end].*convMilli
    NNy = Ny[1:30:end].*convMilli
    NNz = Nz[1:30:end].*convMilli

    # boundary of convergence area
    ϕ=range(0,stop=2*pi,length=100);
    rr = zeros(100,2);
    for i=1:100
       rr[i,1] = R*sin(ϕ[i])*convMilli;
       rr[i,2] = R*cos(ϕ[i])*convMilli;
    end

    center *= convMilli
    R *= convMilli

    # calculate maximum for colorbar
    if vmin == vmax
        vmax = maximum(abs.(Norm))
    end

    ##########
    ## Plot ##
    ##########
    if plane == 0
	# 3 plots
        figure(fignumber,figsize=(18,5))
    else
	# single plot
        figure(fignumber,figsize=(9,6))
    end

    # conversion into millimeter
    M = useMilli ? "mm" : "m"

    # Plot yz-plane
    if plane == 0
        subplot(1,3,1)
    end
    if plane == 0 || plane == 1
	# norm
        imshow(convert(Array{Float64,2},Norm[:,:,1]'),cmap="viridis",vmin = vmin, vmax = vmax,extent= (-R-center[2], R-center[2], -R-center[3], R-center[3]));
	# colorbar
        cb = colorbar()
        cb.ax.tick_params(labelsize=fontsize)
        useMilli ? cb.set_label("||B|| / mT",fontsize=fontsize+2) : cb.set_label("||B|| / T",fontsize=fontsize+2)
	# quiver
        PyPlot.quiver(NNy,NNz,convert(Array{Float64,2},yy[:,:,1]'),convert(Array{Float64,2},zz[:,:,1]'))
	# convergence area
        plotSphere && PyPlot.plot(rr[:,1] .- center[2],rr[:,2] .- center[3],"w");
	# label
        xlabel("y / $M",fontsize=fontsize+2)
        ylabel("z / $M",fontsize=fontsize+2)
        xticks(fontsize=fontsize)
        yticks(fontsize=fontsize)
    end

    # Plot xz-plane
    if plane == 0
        subplot(1,3,2)
    end
    if plane == 0 || plane == 2
	# norm
        imshow(convert(Array{Float64,2},Norm[:,:,2]'),cmap="viridis",vmin = vmin, vmax = vmax,extent= (-R-center[1], R-center[1], -R-center[3], R-center[3]));
	# colorbar
        cb = colorbar()
        cb.ax.tick_params(labelsize=fontsize)
        useMilli ? cb.set_label("||B|| / mT",fontsize=fontsize+2) : cb.set_label("||B|| / T",fontsize=fontsize+2)
	# quiver
        PyPlot.quiver(NNx,NNz,convert(Array{Float64,2},xx[:,:,2]'),convert(Array{Float64,2},zz[:,:,2]'))
	# convergence area
        plotSphere && PyPlot.plot(rr[:,1] .- center[1],rr[:,2] .- center[3],"w");
	# label
        xlabel("x / $M",fontsize=fontsize+2)
        ylabel("z / $M",fontsize=fontsize+2)
        xticks(fontsize=fontsize)
        yticks(fontsize=fontsize)
    end

    # Plot xy-plane
    if plane == 0
        subplot(1,3,3)
    end
    if plane == 0 || plane == 3
	t = ["x","y"]
	if yMainAxis # plot yx-plane
	  a = [2,1];
	else # plot xy-plane
	  a = [1,2];
	end
	# norm
        imshow(convert(Array{Float64,2},Norm[:,:,3]'),cmap="viridis",vmin = vmin, vmax = vmax,extent= (-R-center[a[1]], R-center[a[1]], -R-center[a[2]], R-center[a[2]]));
	# colorbar
        cb = colorbar()
        cb.ax.tick_params(labelsize=fontsize)
        useMilli ? cb.set_label("||B|| / mT",fontsize=fontsize+2) : cb.set_label("||B|| / T",fontsize=fontsize+2)
	# quiver
        yMainAxis ? PyPlot.quiver(NNy,NNx,convert(Array{Float64,2},yy[:,:,3]'),convert(Array{Float64,2},xx[:,:,3]')) : PyPlot.quiver(NNx,NNy,convert(Array{Float64,2},xx[:,:,3]'),convert(Array{Float64,2},yy[:,:,3]'))
	# convergence area
        plotSphere && PyPlot.plot(rr[:,1] .- center[a[1]],rr[:,2] .- center[a[2]],"w");
	# label
        xlabel("$(t[a[1]]) / $M",fontsize=fontsize+2)
        ylabel("$(t[a[2]]) / $M",fontsize=fontsize+2)
        xticks(fontsize=fontsize)
        yticks(fontsize=fontsize)
    end

    tight_layout()

    return fignumber+1
end