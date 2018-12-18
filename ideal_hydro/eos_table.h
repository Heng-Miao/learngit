#ifndef __EOS_H__
#define __EOS_H__

#include "real_type.h"

/** \breif EOS EOSI, s95p-PCE165-v0 from TECHQM */
/** Pressure as a function of energy density in units GeV/fm^3 */
/** P = dof/90*pi^2*T^4 */
/** s = (e + P)/T */
#ifdef EOSI
#define dof     (169.0f/4.0f)  // 7/8*2*3*2.5*2+2*8 = 169/4
#define hbarc1  0.1973269f     // GeV*fm
#define hbarc3  pow(0.1973269631f, 3.0f)
#define coef    (M_PI_F*M_PI_F/30.0f) 

inline real P(real eps, read_only image2d_t eos_table){
    return eps/3.0f;
}

inline real T(real eps, read_only image2d_t eos_table){
    return hbarc1*pow( (real)1.0f/(dof*coef)*eps/hbarc1, (real)0.25f );
}

inline real S(real eps, read_only image2d_t eos_table){
    return ( eps + P(eps, eos_table) )/fmax( (real)1.0E-10f, T(eps, eos_table) );
}

inline real CS2(real eps, read_only image2d_t eos_table){
    return 0.33333333f;
}

#endif

