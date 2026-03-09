import numpy as np
import jax.numpy as jnp
from functools import partial
from data_structures import State, HeartChamberPressureFlowRate

def get_heart_chamber_pressures_and_flow_rates(t, state, params):
    """Computes the pressures and flow rates in the heart chambers at a point in time."""
    s = state
    p = params

    def flux_through_valve(p1, p2, R):
        return ( p1 - p2 ) / R( p1, p2 )

    def time_varying_elastance(EA, EB, time_C, duration_C, duration_R, BPM):
        THB = 60. / BPM
        time_R = time_C + duration_C
        e = lambda t: 0.5 * ( 1 - jnp.cos( jnp.pi / duration_C * ( jnp.mod( t - time_C, THB ) ) ) ) * ( 0 <= jnp.mod( t - time_C, THB ) ) * ( jnp.mod( t - time_C, THB ) < duration_C ) + \
                        0.5 * ( 1 + jnp.cos( jnp.pi / duration_R * ( jnp.mod( t - time_R, THB ) ) ) ) * ( 0 <= jnp.mod( t - time_R, THB ) ) * ( jnp.mod( t - time_R, THB ) < duration_R )
        return lambda t: EA * jnp.clip(e(t), 0.0, 1.0) + EB

    def heavisideMY(x):
            return jnp.arctan( np.pi / 2 * x * 200 ) * 1 / np.pi + 0.5
        
    def scaled_heavyside(w, v, Rmin, Rmax):
        return 10.**( jnp.log10( Rmin ) + ( jnp.log10( Rmax ) - jnp.log10( Rmin ) ) * heavisideMY( v - w ) )
    
    E_LA = time_varying_elastance(p.EA_LA, p.EB_LA, p.tC_LA, p.TC_LA, p.TR_LA, p.BPM)
    E_LV = time_varying_elastance(p.EA_LV, p.EB_LV, p.tC_LV, p.TC_LV, p.TR_LV, p.BPM)
    E_RA = time_varying_elastance(p.EA_RA, p.EB_RA, p.tC_RA, p.TC_RA, p.TR_RA, p.BPM)
    E_RV = time_varying_elastance(p.EA_RV, p.EB_RV, p.tC_RV, p.TC_RV, p.TR_RV, p.BPM)

    ############ PV relationships
    P_LA_func = lambda V, t: E_LA(t) * ( V - p.V0_LA )
    P_LV_func = lambda V, t: E_LV(t) * ( V - p.V0_LV )
    P_RA_func = lambda V, t: E_RA(t) * ( V - p.V0_RA )
    P_RV_func = lambda V, t: E_RV(t) * ( V - p.V0_RV )
        
    P_LA = P_LA_func(s.V_LA, t)
    P_LV = P_LV_func(s.V_LV, t)
    P_RA = P_RA_func(s.V_RA, t)
    P_RV = P_RV_func(s.V_RV, t)

    # Valve resistances
    R_MV = R_AV = R_TV = R_PV = partial(scaled_heavyside, Rmin=p.Rmin, Rmax=p.Rmax)

    # Compute flow rates
    Q_MV = flux_through_valve( P_LA, P_LV, R_MV )
    Q_AV = flux_through_valve( P_LV, s.P_AA, R_AV ) # changed P_Sys_Ar to P_aa
    Q_TV = flux_through_valve( P_RA, P_RV, R_TV )
    Q_PV = flux_through_valve( P_RV, s.P_Ar_Pul, R_PV )

    return HeartChamberPressureFlowRate(
        P_LA      = P_LA, 
        P_LV      = P_LV, 
        P_RA      = P_RA, 
        P_RV      = P_RV, 
        Q_MV      = Q_MV, 
        Q_AV      = Q_AV, 
        Q_TV      = Q_TV, 
        Q_PV      = Q_PV
    )

def cardiovascular_model(t, state, params):
    s = state
    p = params
    hc = get_heart_chamber_pressures_and_flow_rates(t, state, params)
    
    # Chamber volume derivatives
    V_LA_dt = s.Q_Ven_Pul - hc.Q_MV 
    V_LV_dt = hc.Q_MV - hc.Q_AV
    V_RA_dt = s.Q_SVC + s.Q_IVC - hc.Q_TV
    V_RV_dt = hc.Q_TV - hc.Q_PV
    
    # Pulmonary circulation derivatives
    P_Ar_Pul_dt = (hc.Q_PV - s.Q_Ar_Pul) / p.C_Ar_Pul
    #C . dP/dt = Qin - Qout
    P_Ven_Pul_dt = (s.Q_RL2 + s.Q_LL2 - s.Q_Ven_Pul) / p.C_Ven_Pul
    #L dQ/dt = - R Q + Pin - Pout
    Q_Ar_Pul_dt = (-p.R_Ar_Pul * s.Q_Ar_Pul + s.P_Ar_Pul - s.P_Lung) / p.L_Ar_Pul
    Q_Ven_Pul_dt = (-p.R_Ven_Pul * s.Q_Ven_Pul + s.P_Ven_Pul - hc.P_LA) / p.L_Ven_Pul
    
    P_Lung_dt = (s.Q_Ar_Pul - s.Q_LL1 - s.Q_RL1) / p.C_Lung
    Q_RL1_dt = (-p.R_RL1 * s.Q_RL1 + s.P_Lung - s.P_RL) / p.L_RL1
    Q_LL1_dt = (-p.R_LL1 * s.Q_LL1 + s.P_Lung - s.P_LL) / p.L_LL1
    P_LL_dt = (s.Q_LL1 - s.Q_LL2) / p.C_LL
    P_RL_dt = (s.Q_RL1 - s.Q_RL2) / p.C_RL 
    Q_LL2_dt = (-p.R_LL2 * s.Q_LL2 + s.P_LL - s.P_Ven_Pul) / p.L_LL2
    Q_RL2_dt = (-p.R_RL2 * s.Q_RL2 + s.P_RL - s.P_Ven_Pul) / p.L_RL2

    # Systemic circulation derivatives
    P_ULimb_dt = (s.Q_Sub - s.Q_ULimb) / p.C_ULimb
    Q_ULimb_dt = (-p.R_ULimb * s.Q_ULimb + s.P_ULimb - s.Q_ULimb * p.R_ULimbT - s.P_SVC) / p.L_ULimb
    
    P_SVC_dt = (s.Q_ULimb + s.Q_Cer + s.Q_Fac - s.Q_SVC) / p.C_SVC
    Q_SVC_dt = (-p.R_SVC * s.Q_SVC + s.P_SVC - hc.P_RA) / p.L_SVC
    
    P_Cer_dt = (s.Q_ICar - s.Q_Cer) / p.C_Cer
    Q_Cer_dt = (-p.R_Cer * s.Q_Cer - s.Q_Cer * p.R_CerT + s.P_Cer - s.P_SVC) / p.L_Cer
    
    P_Ren_dt = (s.Q_Ren1 - s.Q_Ren2) / p.C_Ren
    Q_Ren2_dt = (-p.R_Ren2 * s.Q_Ren2  - s.Q_Ren2 * p.R_RenT + s.P_Ren - s.P_SVC) / p.L_Ren2
     
    P_AA_dt = (hc.Q_AV - s.Q_AA) / p.C_AA
    Q_AA_dt = (-p.R_AA * s.Q_AA + s.P_AA - s.P_AArc) / p.L_AA
    P_AArc_dt = (s.Q_AA - s.Q_Sub - s.Q_CCar - s.Q_DscA) / p.C_AArc
    
    Q_Sub_dt = (-p.R_Sub * s.Q_Sub + s.P_AArc - s.P_ULimb) / p.L_Sub
    
    P_DscA_dt = (s.Q_DscA - s.Q_Ren1 - s.Q_AbdA) / p.C_DscA
    P_AbdA_dt = (s.Q_AbdA - s.Q_Cel - s.Q_Mes1 - s.Q_LLimb1) / p.C_AbdA # correction
    
    P_Cel_dt = (s.Q_Cel - s.Q_Hep1 - s.Q_Spl1) / p.C_Cel
    Q_Cel_dt = (-p.R_Cel * s.Q_Cel + s.P_AbdA - s.P_Cel) / p.L_Cel # splanchnic -> spleen
    
    Q_DscA_dt = (-p.R_DscA * s.Q_DscA + s.P_AArc - s.P_DscA) / p.L_DscA
    Q_Ren1_dt = (-p.R_Ren1 * s.Q_Ren1 + s.P_DscA - s.P_Ren) / p.L_Ren1
    Q_AbdA_dt = (-p.R_AbdA * s.Q_AbdA + s.P_DscA - s.P_AbdA) / p.L_AbdA
    Q_LLimb1_dt = (-p.R_LLimb1 * s.Q_LLimb1 + s.P_AbdA - s.P_LLimb) / p.L_LLimb1
    
    Q_Spl1_dt = (-p.R_Spl1 * s.Q_Spl1 + s.P_Cel - s.P_Spl) / p.L_Spl1
    P_Spl_dt = (s.Q_Spl1 - s.Q_Spl2) / p.C_Spl
    Q_Spl2_dt = (-p.R_Spl2 * s.Q_Spl2 - s.Q_Spl2 * p.R_SplT + s.P_Spl - s.P_Ven_Por) / p.L_Spl2
    
    P_Hep_dt = (s.Q_Hep1  - s.Q_Hep2) / p.C_Hep
    # ----------------- #
    Q_Hep2_dt = (-p.R_Hep2 * s.Q_Hep2 + s.P_Hep - s.Q_Hep2 * p.R_HepT - s.P_SVC) / p.L_Hep2
    Q_Hep1_dt = (-p.R_Hep1 * s.Q_Hep1 + s.P_Cel - s.P_Hep) / p.L_Hep1
    
    P_Ven_Por_dt = (s.Q_Spl2 + s.Q_Mes2 - s.Q_Ven_Por) / p.C_Ven_Por
    Q_Ven_Por_dt = (-p.R_Ven_Por * s.Q_Ven_Por + s.P_Ven_Por - s.P_IVC) / p.L_Ven_Por
    
    P_Mes_dt = (s.Q_Mes1 - s.Q_Mes2) / p.C_Mes
    Q_Mes2_dt = (-p.R_Mes2 * s.Q_Mes2 - s.Q_Mes2 * p.R_MesT + s.P_Mes - s.P_Ven_Por) / p.L_Mes2
    Q_Mes1_dt = (-p.R_Mes1 * s.Q_Mes1 + s.P_AbdA - s.P_Mes) / p.L_Mes1
    
    P_LLimb_dt = (s.Q_LLimb1 - s.Q_LLimb2) / p.C_LLimb
    Q_LLimb2_dt = (-p.R_LLimb2 * s.Q_LLimb2 + s.P_LLimb - s.Q_LLimb2 * p.R_LLimbT - s.P_SVC) / p.L_LLimb2
    
    P_IVC_dt = (s.Q_Ren2 + s.Q_Hep2 + s.Q_Ven_Por + s.Q_LLimb2 - s.Q_IVC) / p.C_IVC
    Q_IVC_dt = (-p.R_IVC * s.Q_IVC + s.P_IVC - hc.P_RA) / p.L_IVC
    
    P_CCar_dt = (s.Q_CCar - s.Q_ECar - s.Q_ICar) / p.C_CCar
    Q_CCar_dt = (-p.R_CCar * s.Q_CCar + s.P_AArc - s.P_CCar) / p.L_CCar
    Q_ECar_dt = (-p.R_ECar * s.Q_ECar + s.P_CCar - s.P_Fac) / p.L_ECar
    Q_ICar_dt = (-p.R_ICar * s.Q_ICar + s.P_CCar - s.P_Cer) / p.L_ICar
    
    P_Fac_dt = (s.Q_ECar - s.Q_Fac) / p.C_Fac
    Q_Fac_dt = (-p.R_Fac * s.Q_Fac + s.P_Fac - s.Q_Fac * p.R_FacT - s.P_SVC) / p.L_Fac

    return State(
    V_LA      = V_LA_dt,
    V_LV      = V_LV_dt,
    V_RA      = V_RA_dt,
    V_RV      = V_RV_dt,
    P_Ar_Pul  = P_Ar_Pul_dt,
    P_Ven_Pul = P_Ven_Pul_dt,
    Q_Ar_Pul  = Q_Ar_Pul_dt,
    Q_Ven_Pul = Q_Ven_Pul_dt,
    P_Lung    = P_Lung_dt,
    Q_RL1     = Q_RL1_dt,
    Q_LL1     = Q_LL1_dt,
    P_LL      = P_LL_dt,
    P_RL      = P_RL_dt,
    Q_LL2     = Q_LL2_dt,
    Q_RL2     = Q_RL2_dt,
    P_ULimb   = P_ULimb_dt,
    Q_ULimb   = Q_ULimb_dt,
    P_SVC     = P_SVC_dt,
    Q_SVC     = Q_SVC_dt,
    P_Cer     = P_Cer_dt,
    Q_Cer     = Q_Cer_dt,
    P_Ren     = P_Ren_dt,
    Q_Ren2    = Q_Ren2_dt,
    P_AA      = P_AA_dt,
    Q_AA      = Q_AA_dt,
    P_AArc    = P_AArc_dt,
    Q_Sub     = Q_Sub_dt,
    P_DscA    = P_DscA_dt,
    P_AbdA    = P_AbdA_dt,
    P_Cel     = P_Cel_dt,
    Q_Cel     = Q_Cel_dt,
    Q_DscA    = Q_DscA_dt,
    Q_Ren1    = Q_Ren1_dt,
    Q_AbdA    = Q_AbdA_dt,
    Q_LLimb1  = Q_LLimb1_dt,
    Q_Spl1    = Q_Spl1_dt,
    P_Spl     = P_Spl_dt,
    Q_Spl2    = Q_Spl2_dt,
    P_Hep     = P_Hep_dt,
    Q_Hep1    = Q_Hep1_dt,
    Q_Hep2    = Q_Hep2_dt,
    P_Ven_Por = P_Ven_Por_dt,
    Q_Ven_Por = Q_Ven_Por_dt,
    P_Mes     = P_Mes_dt,
    Q_Mes1    = Q_Mes1_dt,
    Q_Mes2    = Q_Mes2_dt,
    P_LLimb   = P_LLimb_dt,
    Q_LLimb2  = Q_LLimb2_dt,
    P_IVC     = P_IVC_dt,
    Q_IVC     = Q_IVC_dt,
    P_CCar    = P_CCar_dt,
    Q_CCar    = Q_CCar_dt,
    Q_ECar    = Q_ECar_dt,
    Q_ICar    = Q_ICar_dt,
    P_Fac     = P_Fac_dt,
    Q_Fac     = Q_Fac_dt)
