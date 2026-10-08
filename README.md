# SmashMPM.jl
 
**S**elf-gravitating **M**eshfree **A**nalysis of **S**hock & **H**ypervelocity


![SMASH Logo](assets/logo/smash_logo.svg)

A Julia implementation of the Material Point Method (MPM) for impact and hypervelocity simulations of solid bodies. The solver runs on the CPU (multithreaded) and on GPUs through [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) with a single code base.
 
> **Status:** early development (v0.1.0). The API may change.

---
 
## Installation
 
Julia 1.12 or newer is required.
1. Clone the repo:
```bash
git clone https://github.com/schwennis/SmashMPM.jl
cd SmashMPM.jl
```
2. Install the dependecies:
```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```
3. Optional: Install CUDA.jl for GPU runs:
```bash
julia --project -e 'using Pkg; Pkg.add("CUDA.jl")'
```
 
## Running the Examples
 
The examples live in [`examples/`](examples). Each example consists of a simulation script and a script that checks conservation of energy and angular momentum.
 
| Example | Simulation | Conservation check |
|---|---|---|
| Rotating cube | `examples/rotating_cube/simulation.jl` | `examples/rotating_cube/conservation_check.jl` |
| Colliding spheres | `examples/sphere_collision/colliding_spheres_sim.jl` | `examples/sphere_collision/conservation_check.jl` |
 
### 1. Set up an environment for the examples
 
The examples need a few packages that are not dependencies of SmashMPM itself (`CairoMakie` and `Glob` for the checks, `CUDA` for GPU runs). Install them in a separate environment so the package's `Project.toml` stays untouched:
 
```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.add(["CairoMakie", "Glob", "CUDA"])'
```
 
`CUDA` can be left out if you only run on the CPU.
 
### 2. Choose the backend
 
Both simulation scripts select the backend with a variable at the top of the file. The default is the GPU:
 
```julia
backend_to_use = :cuda    # change to :cpu to run without a GPU
```
 
For CPU runs, start Julia with multiple threads, e.g. `julia -t auto`.
 
### 3. Run the simulation
 
Run all commands from the repository root, since the output is written to `./output`:
 
```bash
julia -t auto --project=examples examples/sphere_collision/colliding_spheres_sim.jl
```
 
The simulation writes one HDF5 file and one XDMF file (`.xmf`) per output step to `output/`, e.g. `colliding_spheres000042.h5` / `.xmf`. Progress and an ETA are printed to the console.
 
Simulation parameters (resolution `DX`, end time `T_MAX`, output interval, materials, impact parameters, ...) are constants at the top of each script.
 
### 4. Visualize
 
Open the `.xmf` files in [ParaView](https://www.paraview.org/) with the XDMF reader. Particles and the grid are loaded together as one time series.
 
### 5. Check conservation
 
After the simulation has finished:
 
```bash
julia --project=examples examples/sphere_collision/conservation_check.jl
```
 
The script reads all matching HDF5 files from `output/`, prints the final relative deviation of kinetic energy and angular momentum (for particles and for the grid), and saves the plot `energy_and_particle_angular_momentum.png` in the current directory.
 
Note that the two examples write to the same output directory under different file prefixes (`rotating_cube*` and `colliding_spheres*`). The conservation checks pick their own files by prefix. The time window evaluated in the sphere collision check is hard-coded in the script and may need adjusting if you change the simulation parameters.
 
## Tests
 
```bash
julia --project tests/runtests.jl
```
 
## License
 
This project is licensed under the [GNU General Public License v3.0](LICENSE).