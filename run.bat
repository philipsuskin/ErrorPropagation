@REM julia -t 4 -e "include(\"viewer.jl\")"
@REM julia -e "include(\"variance.jl\")"
julia --project=venv -e "include(\"plot.jl\")"