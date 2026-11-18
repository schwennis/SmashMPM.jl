# Changes: 2nd-order KDK / MUSL G2P2G, heat bookkeeping, tools

## Time integration (src/solver/g2p2g.jl, solve.jl, mpm_model.jl)
- 2-pass G2P2G, staggered leapfrog (kick-drift-kick), 2nd order in time (checked in
  `tests/test_time_order.jl`: order 2.00 for positions and F, constant and variable Δt).
  - Pass 1: kick with Δt̄ = (Δt_prev + Δt)/2 (first step Δt/2), AFLIP/APIC update,
    MUSL P2G of mass and momentum at the predicted midpoint x^{n+1/2} = x^n + Δt/2 ṽ^{n-1/2}.
  - Pass 2: MUSL G2P at x^{n+1/2}, drift x^{n+1} = x^n + Δt ṽ, F^{n+1} = exp₂(Δt L) F^n,
    material, spin-fix, P2G at x^{n+1}.
- `MPMModel.dt_prev`, particle field `v_drift` (ṽ^{n-1/2}), MUSL buffer in the grids.
- External forces act on the nodal force (`apply_external_forces!(force, grid)`, no dt).
- Angular momentum stays exact (C-FLIP + spin-fix, spin-fix centered in the MUSL map).

## Heat bookkeeping switch (src/transfer_schemes.jl, g2p2g.jl, grid.jl)
- `AFLIP(α; heat = true)`, `APIC(heat = true)`: the kinetic energy the transfer
  dissipates is booked into the specific internal energy e. Default: `heat = false`
  (identical to before).
- Only terms that are ≥ 0 by construction are booked:
  - P2G loss per node, L_I = Σ_p N ½ m |v_p + C_p r|² - ½ |p_I|²/m_I (Jensen), shared
    by mass over the particles of the node (like the KE fix-up of ALE remaps);
    new grid node field `kin`, written only with heat = true,
  - G2P residual of the affine fit per particle,
  - AFLIP blend loss ½ m α(1-α) ‖ū - u‖²; the other terms times (1-α).
- Not booked (sign-indefinite): energy error of the FLIP path, kinetic energy change
  of the spin-fix.
- Mass, momentum and angular momentum are not affected.
- Tested (`tests/test_heat.jl`): collision E/E0 = 0.997 (0.959 without).
  Hypervelocity impact (Nakamura & Fujiwara setup, Δx = 2 mm, AFLIP(0.95)): stable to
  101 µs, angular momentum exact, E/E0 = 1.05 (0.43 without).
- APIC + heat books its transfer loss exactly but exposes the mismatch between the
  work of the grid forces and the work booked by the material (E/E0 ≈ 1.07-1.16):
  use AFLIP.
- AFLIP with α → 1 (e.g. 0.99) can become unstable in debris clouds (spurious spin from
  the spin-fix, barely damped); the heat switch does not change that. α = 0.95 is stable.

## Tools / output
- `tools/conservation_check.jl`: CLI for mass, momentum, angular momentum (incl. affine
  spin part) and energy (½m|v|², affine ½m d‖C‖², m e) over the HDF5 output.
- HDF5 export writes the file attributes `dx` and `shapefunction`.
- `format_sim_time`: export log with adaptive time units (ps … d).
