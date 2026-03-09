from typing import NamedTuple
import jax.numpy as jnp
import equinox as eqx

class State(NamedTuple):
    """Class to store the state (volumes, pressures, flow rates) of the cardiovascular model."""
    V_LA: float
    V_LV: float
    V_RA: float
    V_RV: float
    P_Ar_Pul: float
    P_Ven_Pul: float
    Q_Ar_Pul: float
    Q_Ven_Pul: float
    P_Lung: float
    Q_RL1: float
    Q_LL1: float
    P_LL: float
    P_RL: float
    Q_LL2: float
    Q_RL2: float
    P_Ren: float
    Q_Ren2: float
    P_ULimb: float
    Q_ULimb: float
    P_Cer: float
    Q_Cer: float
    P_SVC: float
    Q_SVC: float
    P_AA: float
    Q_AA: float
    P_AArc: float
    Q_Sub: float
    Q_CCar: float
    P_DscA: float
    P_AbdA: float
    P_Cel: float
    Q_Cel: float
    Q_DscA: float
    Q_Ren1: float
    Q_AbdA: float
    Q_LLimb1: float
    Q_Spl1: float
    P_Spl: float
    Q_Spl2: float
    P_Hep: float
    Q_Hep2: float
    Q_Hep1: float
    P_Ven_Por: float
    Q_Ven_Por: float
    P_Mes: float
    Q_Mes2: float
    Q_Mes1: float
    P_LLimb: float
    Q_LLimb2: float
    P_IVC: float
    Q_IVC: float
    P_CCar: float
    Q_ECar: float
    Q_ICar: float
    P_Fac: float
    Q_Fac: float

    def to_array(self, names=None):
        lst = []
        if names is None:
            for name in self._fields:
                lst.append(getattr(self, name))
        else:
            for name in names:
                lst.append(getattr(self, name))
        return jnp.stack(lst)
    
    def update_from_array(self, arr, names):
        for i, name in enumerate(names):
            self = eqx.tree_at(lambda p: getattr(p, name), self, arr[i])
        return self
    
    @classmethod
    def from_array(cls, arr):#, names=None):
        return cls(*arr)
        # if names is None:
        #     return cls(*arr)
        # else:
        #     p = cls.from_dict({})
        #     for i, name in enumerate(names):
        #         p = eqx.tree_at(lambda p: getattr(p, name), p, arr[i])
        #     return p
    
    @classmethod
    def from_dict(cls, d):
        """Create a State object from a dictionary."""
        return cls( \
            V_LA       = float(d.get('V_LA'      ,  80.)),  # [ml]
            V_LV       = float(d.get('V_LV'      , 110.)),  # [ml]
            V_RA       = float(d.get('V_RA'      ,  80.)),  # [ml]
            V_RV       = float(d.get('V_RV'      , 110.)),  # [ml]
            P_Ar_Pul   = float(d.get('P_Ar_Pul'  ,  25.)),  # [mmHg]
            P_Ven_Pul  = float(d.get('P_Ven_Pul' ,  20.)),  # [mmHg]
            Q_Ar_Pul   = float(d.get('Q_Ar_Pul'  ,  10.)),  # [ml/s]
            Q_Ven_Pul  = float(d.get('Q_Ven_Pul' ,  10.)),  # [ml/s]
            P_Lung     = float(d.get('P_Lung'    ,  10.)),  # [mmHg]
            Q_LL2      = float(d.get('Q_LL2'     ,  10.)),  # [ml/s]
            Q_RL2      = float(d.get('Q_RL2'     ,  10.)),  # [ml/s]
            Q_LL1      = float(d.get('Q_LL1'     ,  10.)),  # [ml/s]
            Q_RL1      = float(d.get('Q_RL1'     ,  10.)),  # [ml/s]
            P_LL       = float(d.get('P_LL'      ,  10.)),  # [mmHg]
            P_RL       = float(d.get('P_RL'      ,  10.)),  # [mmHg]
            P_AA       = float(d.get('P_AA'      ,  40.)),  # [mmHg]
            Q_AA       = float(d.get('Q_AA'      ,   0.)),  # [ml/s]
            P_AArc     = float(d.get('P_AArc'    ,  40.)),  # [mmHg]
            P_Ren      = float(d.get('P_Ren'     ,  50.)),  # [mmHg]
            Q_Ren2     = float(d.get('Q_Ren2'    ,  20.)),  # [ml/s]
            P_DscA     = float(d.get('P_DscA'    ,  50.)),  # [mmHg]
            P_AbdA     = float(d.get('P_AbdA'    ,  40.)),  # [mmHg]
            Q_DscA     = float(d.get('Q_DscA'    ,   0.)),  # [ml/s]
            Q_Ren1     = float(d.get('Q_Ren1'    ,   0.)),  # [ml/s]
            Q_AbdA     = float(d.get('Q_AbdA'    ,   0.)),  # [ml/s]
            Q_LLimb1   = float(d.get('Q_LLimb1'  ,   0.)),  # [ml/s]
            P_ULimb    = float(d.get('P_ULimb'   ,  40.)),  # [mmHg]
            Q_Sub      = float(d.get('Q_Sub'     ,   0.)),  # [ml/s]
            P_Cer      = float(d.get('P_Cer'     ,  40.)),  # [mmHg]
            Q_Cer      = float(d.get('Q_Cer'     ,   0.)),  # [ml/s]
            P_Cel      = float(d.get('P_Cel'     ,  20.)),  # [mmHg]
            Q_Cel      = float(d.get('Q_Cel'     ,   0.)),  # [ml/s]
            P_Spl      = float(d.get('P_Spl'     ,  20.)),  # [mmHg]
            Q_Spl2     = float(d.get('Q_Spl2'    ,   0.)),  # [ml/s]
            Q_Spl1     = float(d.get('Q_Spl1'    ,   0.)),  # [ml/s]
            P_Hep      = float(d.get('P_Hep'     ,  20.)),  # [mmHg]
            Q_Hep1     = float(d.get('Q_Hep1'    ,   0.)),  # [ml/s]
            Q_Hep2     = float(d.get('Q_Hep2'    ,   0.)),  # [ml/s]
            P_Mes      = float(d.get('P_Mes'     ,  20.)),  # [mmHg]
            Q_Mes1     = float(d.get('Q_Mes1'    ,   0.)),  # [ml/s]
            Q_Mes2     = float(d.get('Q_Mes2'    ,   0.)),  # [ml/s]
            P_LLimb    = float(d.get('P_LLimb'   ,  20.)),  # [mmHg]
            Q_LLimb2   = float(d.get('Q_LLimb2'  ,   0.)),  # [ml/s]
            P_Ven_Por  = float(d.get('P_Ven_Por' ,  20.)),  # [mmHg]
            Q_Ven_Por  = float(d.get('Q_Ven_Por',   0.)),  # [ml/s]
            P_CCar     = float(d.get('P_CCar'    ,  20.)),  # [mmHg]
            Q_CCar     = float(d.get('Q_CCar'    ,   0.)),  # [ml/s]
            Q_ECar     = float(d.get('Q_ECar'    ,   0.)),  # [ml/s]
            Q_ICar     = float(d.get('Q_ICar'    ,   0.)),  # [ml/s]
            P_Fac      = float(d.get('P_Fac'     ,  20.)),  # [mmHg]
            Q_Fac      = float(d.get('Q_Fac'     ,   0.)),  # [ml/s]
            Q_ULimb     = float(d.get('Q_ULimb'    ,   0.)),  # [ml/s]
            P_SVC     = float(d.get('P_SVC'    ,   0.)),  # [ml/s]
            Q_SVC     = float(d.get('Q_SVC'    ,   0.)),  # [ml/s]
            P_IVC      = float(d.get('P_IVC'     ,  20.)),  # [mmHg]
            Q_IVC      = float(d.get('Q_IVC'     ,   0.)),  # [ml/s]
            
        )

class HeartChamberPressureFlowRate(NamedTuple):
    """Class to store the pressure and flow rate in the heart chambers. These can be computed from the `State`."""
    P_LA: float
    P_LV: float
    P_RA: float
    P_RV: float
    Q_MV: float
    Q_AV: float
    Q_TV: float
    Q_PV: float

class Params(NamedTuple):
    """Class to store the parameters (compliance, resistance, etc.) of the cardiovascular model."""
    
    # Heart rate
    BPM: float
    
    # Left Atrium
    EA_LA: float 
    EB_LA: float 
    TC_LA: float
    TR_LA: float 
    tC_LA: float 
    V0_LA: float
    
    # Left Ventricle
    EA_LV: float 
    EB_LV: float 
    TC_LV: float 
    TR_LV: float 
    tC_LV: float 
    V0_LV: float
    
    # Right Atrium
    EA_RA: float 
    EB_RA: float 
    TC_RA: float 
    TR_RA: float 
    tC_RA: float
    V0_RA: float 
    
    # Right Ventricle
    EA_RV: float 
    EB_RV: float 
    TC_RV: float 
    TR_RV: float 
    tC_RV: float 
    V0_RV: float 
    
    # Valve resistances
    Rmin: float 
    Rmax: float 

    # Pulmonary circulation
    R_Ar_Pul: float 
    C_Ar_Pul: float 
    R_Ven_Pul: float 
    C_Ven_Pul: float 
    L_Ar_Pul: float 
    L_Ven_Pul: float 
    
    # Lung compartments
    R_RL1: float 
    L_RL1: float 
    R_RL2: float 
    L_RL2: float 
    C_RL: float 
    R_LL1: float 
    L_LL1: float 
    R_LL2: float 
    L_LL2: float 
    C_LL: float 
    C_Lung: float 
    
    # Ascending Aorta
    R_AA: float 
    L_AA: float 
    C_AA: float 
    
    # Aortic Arch
    C_AArc: float 
    
    # Descending Aorta
    R_DscA: float 
    L_DscA: float 
    C_DscA: float 
    
    # Abdominal Aorta
    R_AbdA: float 
    L_AbdA: float 
    C_AbdA: float 
    
    # Cerebral circulation
    R_CerT: float 
    L_Cer: float 
    C_Cer: float 
    R_Cer: float 
    
    # Upper limbs
    R_ULimbT: float 
    L_ULimb: float 
    C_ULimb: float 
    R_ULimb: float 
    
    # Facial
    R_FacT: float 
    L_Fac: float 
    C_Fac: float 
    R_Fac: float 

    # Renal
    R_RenT: float 
    L_Ren1: float 
    L_Ren2: float 
    C_Ren: float 
    R_Ren1: float 
    R_Ren2: float 
    
    # Spleen
    R_SplT: float 
    L_Spl1: float 
    L_Spl2: float 
    C_Spl: float 
    R_Spl1: float 
    R_Spl2: float 
    
    # Lower limb
    R_LLimbT: float 
    L_LLimb1: float 
    L_LLimb2: float 
    C_LLimb: float 
    R_LLimb1: float 
    R_LLimb2: float 
    
    # Hepatic
    R_HepT: float 
    L_Hep1: float 
    L_Hep2: float 
    C_Hep: float 
    R_Hep1: float 
    R_Hep2: float 
    
    # Mesenteric
    R_MesT: float 
    L_Mes1: float 
    L_Mes2: float 
    C_Mes: float 
    R_Mes1: float 
    R_Mes2: float 
    
    # Portal vein
    R_Ven_PorT: float 
    L_Ven_Por: float 
    C_Ven_Por: float 
    R_Ven_Por: float 
 
    # Celiac
    R_Cel: float 
    L_Cel: float 
    C_Cel: float 
    
    # Subclavian
    R_Sub: float 
    L_Sub: float 
    
    # Carotid (common)
    R_CCar: float 
    L_CCar: float 
    C_CCar: float 
    
    # Internal carotid
    R_ICar: float 
    L_ICar: float 
    
    # External carotid
    R_ECar: float 
    L_ECar: float
    
    # Vena cava
    R_SVC: float 
    L_SVC: float 
    C_SVC: float 
    R_IVC: float 
    L_IVC: float 
    C_IVC: float 


    def to_array(self, names=None):
        lst = []
        if names is None:
            for name in self._fields:
                lst.append(getattr(self, name))
        else:
            for name in names:
                lst.append(getattr(self, name))
        return jnp.stack(lst)
    
    def update_from_array(self, arr, names):
        for i, name in enumerate(names):
            self = eqx.tree_at(lambda p: getattr(p, name), self, arr[i])
        return self
    
    @classmethod
    def from_array(cls, arr):#, names=None):
        return cls(*arr)
        # if names is None:
        #     return cls(*arr)
        # else:
        #     p = cls.from_dict({})
        #     for i, name in enumerate(names):
        #         p = eqx.tree_at(lambda p: getattr(p, name), p, arr[i])
        #     return p
    @classmethod
    def from_dict(cls, d):
        kwargs = dict()
        BPM = kwargs['BPM'] = float(d.get('BPM', 72.)) # [1 / min]
        THB = 60. / BPM # [s], Heartbeat period
        
        ############ Chambers
        # LA
        tmP_d = d.get('LA', dict())
        kwargs.update(
            EA_LA = float(tmP_d.get('EA', 0.07)), # [mmHg / ml]
            EB_LA = float(tmP_d.get('EB', 0.09)), # [mmHg / ml]
            TC_LA = float(tmP_d.get('TC', 0.17)) * THB,  # [s]
            TR_LA = float(tmP_d.get('TR', 0.17)) * THB,  # [s]
            tC_LA = float(tmP_d.get('tC', 0.80)) * THB,  # [s]
            V0_LA = float(tmP_d.get('V0', 4.0)), # [ml]
        )

        # LV
        tmP_d = d.get('LV', dict())
        kwargs.update(
            EA_LV = float(tmP_d.get('EA', 2.75)), # [mmHg / ml]
            EB_LV = float(tmP_d.get('EB', 0.08)), # [mmHg / ml]
            TC_LV = float(tmP_d.get('TC', 0.34)) * THB,  # [s]
            TR_LV = float(tmP_d.get('TR', 0.17)) * THB,  # [s]
            tC_LV = float(tmP_d.get('tC', 0.00)) * THB,  # [s]
            V0_LV = float(tmP_d.get('V0', 5.0)), # [ml]
        )

        # RA
        tmP_d = d.get('RA', dict())
        kwargs.update(
            EA_RA = float(tmP_d.get('EA', 0.06)), # [mmHg / ml]
            EB_RA = float(tmP_d.get('EB', 0.07)), # [mmHg / ml]
            TC_RA = float(tmP_d.get('TC', 0.17)) * THB,  # [s]
            TR_RA = float(tmP_d.get('TR', 0.17)) * THB,  # [s]
            tC_RA = float(tmP_d.get('tC', 0.80)) * THB,  # [s]
            V0_RA = float(tmP_d.get('V0', 4.0)), # [ml]
        )

        # RV
        tmP_d = d.get('RV', dict())
        kwargs.update(
            EA_RV = float(tmP_d.get('EA', 0.55)), # [mmHg / ml]
            EB_RV = float(tmP_d.get('EB', 0.05)), # [mmHg / ml]
            TC_RV = float(tmP_d.get('TC', 0.34)) * THB,  # [s]
            TR_RV = float(tmP_d.get('TR', 0.17)) * THB,  # [s]
            tC_RV = float(tmP_d.get('tC', 0.00)) * THB,  # [s]
            V0_RV = float(tmP_d.get('V0', 10.0)), # [ml]
        )

        # Valves           
        tmP_d = d.get('valves', dict())
        kwargs.update(
            Rmin = float(tmP_d.get('Rmin', 0.0075)), # [mmHg s / ml]
            Rmax = float(tmP_d.get('Rmax', 75006.2)), # [mmHg s / ml]
        )

        # Pulmonary circulation
        tmp_d = d.get('Pul', dict())
        kwargs.update(
            R_Ar_Pul  = float(tmp_d.get('R_Ar', 0.032)),    # [mmHg s / ml]
            C_Ar_Pul  = float(tmp_d.get('C_Ar', 10.0)),     # [ml / mmHg]
            R_Ven_Pul = float(tmp_d.get('R_Ven', 0.035)),   # [mmHg s / ml]
            C_Ven_Pul = float(tmp_d.get('C_Ven', 16.0)),    # [ml / mmHg]
            L_Ar_Pul  = float(tmp_d.get('L_Ar', 5e-4)),     # [mmHg s^2 / ml]
            L_Ven_Pul = float(tmp_d.get('L_Ven', 5e-4)),    # [mmHg s^2 / ml]
        )

        # Lung compartments
        tmp_d = d.get('Lung', dict())
        kwargs.update(
            R_RL1 = float(tmp_d.get('R_RL1', 0.0088)),       # [mmHg s / ml]
            L_RL1 = float(tmp_d.get('L_RL1', 0.0000296)),    # [mmHg s^2 / ml]
            R_RL2 = float(tmp_d.get('R_RL2', 0.0081)),
            L_RL2 = float(tmp_d.get('L_RL2', 0.0000296)),
            C_RL  = float(tmp_d.get('C_RL', 127.0)),
            R_LL1 = float(tmp_d.get('R_LL1', 0.0123)),
            L_LL1 = float(tmp_d.get('L_LL1', 0.0000331)),
            R_LL2 = float(tmp_d.get('R_LL2', 0.0081)),
            L_LL2 = float(tmp_d.get('L_LL2', 0.0000296)),
            C_LL  = float(tmp_d.get('C_LL', 11.34)),
            C_Lung = float(tmp_d.get('C_Lung', 771.0)),
        )
    
        tmp_d = d.get('Compliance', dict())
        kwargs.update(
            C_AA       = float(tmp_d.get('C_AA', 1.5)),
            C_AArc     = float(tmp_d.get('C_AArc', 1.0)),
            C_DscA     = float(tmp_d.get('C_DscA', 1.0)),
            C_AbdA     = float(tmp_d.get('C_AbdA', 1.0)),
            C_CCar     = float(tmp_d.get('C_CCar', 0.8)),
            C_Cel      = float(tmp_d.get('C_Cel', 0.5)),
            C_Ren      = float(tmp_d.get('C_Ren', 0.5)),
            C_Mes      = float(tmp_d.get('C_Mes', 0.5)),
            C_Hep      = float(tmp_d.get('C_Hep', 0.5)),
            C_Spl      = float(tmp_d.get('C_Spl', 0.3)),
            C_ULimb    = float(tmp_d.get('C_ULimb', 0.7)),
            C_LLimb    = float(tmp_d.get('C_LLimb', 0.7)),
            C_Cer      = float(tmp_d.get('C_Cer', 0.3)),
            C_Fac      = float(tmp_d.get('C_Fac', 0.3)),
            C_SVC      = float(tmp_d.get('C_SVC', 50)),
            C_IVC      = float(tmp_d.get('C_IVC', 200)),
            C_Ven_Por  = float(tmp_d.get('C_Ven_Por', 20)),
            C_Ar_Pul   = float(tmp_d.get('C_Ar_Pul', 8)),
            C_Ven_Pul  = float(tmp_d.get('C_Ven_Pul', 8)),
            C_Lung     = float(tmp_d.get('C_Lung', 10)),
            C_LL       = float(tmp_d.get('C_LL', 5)),
            C_RL       = float(tmp_d.get('C_RL', 5)),
        )

        # Resistance
        tmp_d = d.get('Resistance', dict())
        kwargs.update(
            R_Ar_Pul        = float(tmp_d.get('R_Ar', 0.016)),
            R_Ven_Pul       = float(tmp_d.get('R_Ven', 0.0175)),
            R_RL1       = float(tmp_d.get('R_RL1', 0.0335)),
            R_RL2       = float(tmp_d.get('R_RL2', 0.0335)),
            R_LL1       = float(tmp_d.get('R_LL1', 0.0335)),
            R_LL2       = float(tmp_d.get('R_LL2', 0.0335)),
            R_AA        = float(tmp_d.get('R_AA', 0.02)),
            R_DscA      = float(tmp_d.get('R_DscA', 0.01)),
            R_AbdA      = float(tmp_d.get('R_AbdA', 0.04)),
            R_Cel       = float(tmp_d.get('R_Cel', 0.1)),
            R_RenT      = float(tmp_d.get('R_RenT', 1.5)),
            R_Ren1      = float(tmp_d.get('R_Ren1', 0.1)),
            R_Ren2      = float(tmp_d.get('R_Ren2', 0.9)),
            R_MesT      = float(tmp_d.get('R_MesT', 4.0)),
            R_Mes1      = float(tmp_d.get('R_Mes1', 1.2)),
            R_Mes2      = float(tmp_d.get('R_Mes2', 1.2)),
            R_HepT      = float(tmp_d.get('R_HepT', 4.0)),
            R_Hep1      = float(tmp_d.get('R_Hep1', 1.2)),
            R_Hep2      = float(tmp_d.get('R_Hep2', 1.2)),
            R_SplT      = float(tmp_d.get('R_SplT', 12.0)),
            R_Spl1      = float(tmp_d.get('R_Spl1', 4.0)),
            R_Spl2      = float(tmp_d.get('R_Spl2', 4.0)),
            R_LLimbT    = float(tmp_d.get('R_LLimbT', 4.0)),
            R_LLimb1    = float(tmp_d.get('R_LLimb1', 1.2)),
            R_LLimb2    = float(tmp_d.get('R_LLimb2', 1.2)),
            R_CCar      = float(tmp_d.get('R_CCar', 0.08)),
            R_ICar      = float(tmp_d.get('R_ICar', 0.08)),
            R_CerT      = float(tmp_d.get('R_CerT', 20.0)),
            R_Cer       = float(tmp_d.get('R_Cer', 0.1)),
            R_Sub       = float(tmp_d.get('R_Sub', 1.4)),
            R_ULimbT    = float(tmp_d.get('R_ULimbT', 7.0)),
            R_ULimb     = float(tmp_d.get('R_ULimb', 2.1)),
            R_ECar      = float(tmp_d.get('R_ECar', 0.6)),
            R_FacT      = float(tmp_d.get('R_FacT', 35.0)),
            R_Fac       = float(tmp_d.get('R_Fac', 0.3)),
            R_Ven_PorT  = float(tmp_d.get('R_Ven_PorT', 4.2)),
            R_Ven_Por  = float(tmp_d.get('R_Ven_Por', 3.0)),
            R_SVC       = float(tmp_d.get('R_SVC', 0.457)),
            R_IVC       = float(tmp_d.get('R_IVC', 0.183)),
        )
        
        # Inductance (also called Inertia)
        tmp_d = d.get('Inductance', dict())
        kwargs.update(
            L_Ar_Pul        = float(tmp_d.get('L_Ar', 0.00025)),
            L_Ven_Pul       = float(tmp_d.get('L_Ven', 0.00025)),
            L_RL1       = float(tmp_d.get('L_RL1', 0.0005)),
            L_RL2       = float(tmp_d.get('L_RL2', 0.0005)),
            L_LL1       = float(tmp_d.get('L_LL1', 0.0005)),
            L_LL2       = float(tmp_d.get('L_LL2', 0.0005)),
            L_AA        = float(tmp_d.get('L_AA', 0.00125)),
            L_DscA      = float(tmp_d.get('L_DscA', 0.00125)),
            L_AbdA      = float(tmp_d.get('L_AbdA', 0.00125)),
            L_Cel       = float(tmp_d.get('L_Cel', 0.00107)),
            L_Ren1      = float(tmp_d.get('L_Ren1', 0.00107)),
            L_Ren2      = float(tmp_d.get('L_Ren2', 0.00107)),
            L_Mes1      = float(tmp_d.get('L_Mes1', 0.00107)),
            L_Mes2      = float(tmp_d.get('L_Mes2', 0.00107)),
            L_Hep1      = float(tmp_d.get('L_Hep1', 0.00107)),
            L_Hep2      = float(tmp_d.get('L_Hep2', 0.00107)),
            L_Spl1      = float(tmp_d.get('L_Spl1', 0.00429)),
            L_Spl2      = float(tmp_d.get('L_Spl2', 0.00429)),
            L_LLimb1    = float(tmp_d.get('L_LLimb1', 0.00086)),
            L_LLimb2    = float(tmp_d.get('L_LLimb2', 0.00086)),
            L_CCar      = float(tmp_d.get('L_CCar', 0.00179)),
            L_ICar      = float(tmp_d.get('L_ICar', 0.00179)),
            L_Cer       = float(tmp_d.get('L_Cer', 0.00179)),
            L_Sub       = float(tmp_d.get('L_Sub', 0.00268)),
            L_ULimb     = float(tmp_d.get('L_ULimb', 0.00268)),
            L_ECar      = float(tmp_d.get('L_ECar', 0.00536)),
            L_Fac       = float(tmp_d.get('L_Fac', 0.00536)),
            L_Ven_Por  = float(tmp_d.get('L_Ven_Por', 0.00107)),
            L_SVC       = float(tmp_d.get('L_SVC', 0.00008)),
            L_IVC       = float(tmp_d.get('L_IVC', 0.00004)),
        )

        return cls(**kwargs)
