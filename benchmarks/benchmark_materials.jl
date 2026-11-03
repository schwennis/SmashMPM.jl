# File: benchmarks/benchmark_materials.jl

"""
    add_material_benchmarks!(suite, T=Float64)

Register the backend independent material benchmarks in `suite["materials_T"]`:
  - simple hyperelastic models
  - all modular SolidMaterial configurations (Iron & Basalt)
  - the single regimes of the equations of state
Material models run inside the G2P2G kernel, so all of them must be allocation-free.
"""
function add_material_benchmarks!(suite::BenchmarkGroup, ::Type{T}=Float64) where {T}
    grp = suite["materials_$(T)"] = BenchmarkGroup()

    # Moderate compression with a little shear
    F_comp = @SMatrix T[
        0.96  0.02  0.00;
        0.01  0.97 -0.01;
        0.00 -0.01  0.98
    ]
    C_comp = @SMatrix T[
        -15.0   4.0   0.0;
          4.0 -10.0   1.0;
          0.0   1.0 -12.0
    ]
    V0 = T(1.0e-6)
    dt = T(1.0e-7)
    dx = T(1.0e-2)

    # ------------------------------------------------------------------------ #
    #                        Simple Hyperelastic Models                        #
    # ------------------------------------------------------------------------ #
    lin_el = LinearElastic(E=T(200.0e9), ν=T(0.3), ρ=T(7874.0))
    lin_st = initial_material_state(lin_el)
    m_fe   = T(7874.0) * V0

    neohook = NeoHookean(E=T(200.0e9), ν=T(0.3), ρ=T(7874.0))
    neo_st  = initial_material_state(neohook)

    grp["simple"] = BenchmarkGroup()
    grp["simple"]["LinearElastic"] = @benchmarkable material_model(
        $lin_el, $lin_st, $F_comp, $C_comp, $V0, $m_fe, $dt, $dx
    )
    grp["simple"]["NeoHookean"] = @benchmarkable material_model(
        $neohook, $neo_st, $F_comp, $C_comp, $V0, $m_fe, $dt, $dx
    )
    expect_no_allocs!(grp["simple"])

    # ------------------------------------------------------------------------ #
    #                  Modular SolidMaterial (Iron & Basalt)                   #
    # ------------------------------------------------------------------------ #
    grp["solid_pipeline"] = BenchmarkGroup()
    for (mat_name, factory) in (("Iron", Iron), ("Basalt", Basalt)),
        eos_sym in (:murnaghan, :tillotson),
        el_sym in (:hypoelastic, :hyperelastic),
        (visc_name, visc) in (("NoVisc", SmashMPM.NoViscosity()), ("BulkVisc", BulkViscosity{T}()))

        mat = factory(T; eos=eos_sym, elasticity=el_sym, viscosity=visc)
        st  = initial_material_state(mat)
        m   = mat.ρ * V0
        key = "$(mat_name)_$(eos_sym)_$(el_sym)_$(visc_name)"

        grp["solid_pipeline"][key] = @benchmarkable material_model(
            $mat, $st, $F_comp, $C_comp, $V0, $m, $dt, $dx
        )
    end
    expect_no_allocs!(grp["solid_pipeline"])

    # ------------------------------------------------------------------------ #
    #                        Equation of State Regimes                         #
    # ------------------------------------------------------------------------ #
    # The Tillotson EoS branches on ρ and e, every branch is benchmarked on its own
    grp["eos"] = BenchmarkGroup()
    iron_till = Iron(T; eos=:tillotson).eos
    st_till   = init_state(iron_till)
    ρ0        = iron_till.ρ0

    # condensed (ρ >= ρ0)
    grp["eos"]["Tillotson_condensed"] = @benchmarkable update_eos(
        $iron_till, $st_till, $(T(1.05) * ρ0), $(T(1.0e5))
    )
    # cold expanded (ρ < ρ0, e < Eiv)
    grp["eos"]["Tillotson_cold_expanded"] = @benchmarkable update_eos(
        $iron_till, $st_till, $(T(0.92) * ρ0), $(T(0.5) * iron_till.Eiv)
    )
    # transition, blend of both branches (ρ < ρ0, Eiv <= e <= Ecv)
    grp["eos"]["Tillotson_transition"] = @benchmarkable update_eos(
        $iron_till, $st_till, $(T(0.90) * ρ0), $(T(0.5) * (iron_till.Eiv + iron_till.Ecv))
    )
    # hot expanded / gas (ρ < ρ0, e > Ecv)
    grp["eos"]["Tillotson_hot_expanded"] = @benchmarkable update_eos(
        $iron_till, $st_till, $(T(0.80) * ρ0), $(T(2.0) * iron_till.Ecv)
    )

    iron_murn = Iron(T; eos=:murnaghan).eos
    st_murn   = init_state(iron_murn)
    grp["eos"]["Murnaghan_compression"] = @benchmarkable update_eos(
        $iron_murn, $st_murn, $(T(1.05) * ρ0), $(zero(T))
    )
    expect_no_allocs!(grp["eos"])

    return grp
end
