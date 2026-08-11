#ifndef NEWRADX_HXX
#define NEWRADX_HXX

#include <cctk.h>
#include <loop_device.hxx>

namespace NewRadX {

/* Mixed precision: both entry points below are templated over the floating
 * point width of the grid functions they are given, so that they can be used
 * by evolution thorns whose state variables are declared as CCTK_REAL4 or
 * CCTK_REAL2 (the sized real types of the Cactus mixed-precision extensions)
 * rather than as plain CCTK_REAL.
 *
 * The templates are explicitly instantiated in newradx.cxx for
 *
 *   T  (state variable and its RHS)              CCTK_REAL8, CCTK_REAL4,
 *                                                CCTK_REAL2
 *   TC (coordinates and Jacobians, multipatch)   CCTK_REAL8, CCTK_REAL4,
 *                                                CCTK_REAL2
 *
 * in every combination, subject to the flesh actually providing the type
 * (CCTK_REAL4 and CCTK_REAL2 are optional; CCTK_REAL2 in particular requires
 * a compiler with `_Float16`).  CCTK_REAL is a preprocessor alias for one of
 * the sized types, so callers that pass CCTK_REAL grid functions -- i.e. all
 * pre-existing callers -- keep resolving to the same code as before.
 *
 * Within one call, `var` and `rhs` must share a width, and so must all twelve
 * coordinate/Jacobian grid functions of the multipatch overload; the widths of
 * those two groups are independent of each other.  Any other T is a link
 * error, not a compile error, since only the declarations are visible here.
 *
 * `var0`, `v0` and `radpower` are CCTK_REAL irrespective of T, and all of the
 * boundary-condition arithmetic is performed in CCTK_REAL as well; only the
 * value finally stored into `rhs` is narrowed to T.  See the comment at the
 * top of newradx.cxx for why.
 */

/**
 * @brief Applies radiative boundary condition to the RHS of a state variable
 *
 * Adapted from NewRad thorn by E. Schnetter, used with Carpet. Original code
 * adapted from BSSN_MoL's files NewRad.F and newrad.h. This code was probably
 * originally written by Miguel Alcubierre.
 *
 * @param cctkGH Pointer to Cactus grid hierarchy struct.
 * @param var State variable which will have boundary conditions applied to it.
 * @param rhs RHS of the evolution equation for @param var
 * @param var0 Value at infinity.
 * @param v0 Propagation speed.
 * @param radpower Radial fall-off exponent
 */
template <typename T>
void NewRadX_Apply(const cGH *restrict const cctkGH,
                   const Loop::GF3D2<const T> &var, const Loop::GF3D2<T> &rhs,
                   const CCTK_REAL var0, const CCTK_REAL v0,
                   const CCTK_REAL radpower);

#define NEWRADX_MULTIPATCH_QUANTITIES                                          \
  vcoordx, vcoordy, vcoordz, vJ_da_dx, vJ_da_dy, vJ_da_dz, vJ_db_dx, vJ_db_dy, \
      vJ_db_dz, vJ_dc_dx, vJ_dc_dy, vJ_dc_dz

/**
 * @brief Applies radiative boundary condition to the RHS of a state variable
 * on multipatch grids. See above for original credits.
 *
 * @param cctkGH Pointer to Cactus grid hierarchy struct.
 * @param var State variable which will have boundary conditions applied to it.
 * @param rhs RHS of the evolution equation for @param var
 * @param vcoordx Global x vertex coordinates grid function. Provided by
 * CoordinatesX
 * @param vcoordy Global y vertex coordinates grid function. Provided by
 * CoordinatesX
 * @param vcoordz Global z vertex coordinates grid function. Provided by
 * CoordinatesX
 * @param vJ_da_dx Vertex centered coordinate transformation Jacobian da/dx.
 * Provided by any thorn implementing the Multipatch interface.
 * @param vJ_da_dy Vertex centered coordinate transformation Jacobian da/dy.
 * Provided by any thorn implementing the Multipatch interface.
 * @param vJ_da_dz Vertex centered coordinate transformation Jacobian da/dz.
 * Provided by any thorn implementing the Multipatch interface.
 * @param vJ_db_dx Vertex centered coordinate transformation Jacobian db/dx.
 * Provided by any thorn implementing the Multipatch interface.
 * @param vJ_db_dy Vertex centered coordinate transformation Jacobian db/dy.
 * Provided by any thorn implementing the Multipatch interface.
 * @param vJ_db_dz Vertex centered coordinate transformation Jacobian db/dz.
 * Provided by any thorn implementing the Multipatch interface.
 * @param vJ_dc_dx Vertex centered coordinate transformation Jacobian dc/dx.
 * Provided by any thorn implementing the Multipatch interface.
 * @param vJ_dc_dy Vertex centered coordinate transformation Jacobian dc/dy.
 * Provided by any thorn implementing the Multipatch interface.
 * @param vJ_dc_dz Vertex centered coordinate transformation Jacobian dc/dz.
 * Provided by any thorn implementing the Multipatch interface.
 * @param var0 Value at infinity.
 * @param v0 Propagation speed.
 * @param radpower Radial fall-off exponent
 */
template <typename T, typename TC>
void NewRadX_Apply(const cGH *restrict const cctkGH,
                   const Loop::GF3D2<const T> &var, const Loop::GF3D2<T> &rhs,
                   const Loop::GF3D2<const TC> &vcoordx,
                   const Loop::GF3D2<const TC> &vcoordy,
                   const Loop::GF3D2<const TC> &vcoordz,
                   const Loop::GF3D2<const TC> &vJ_da_dx,
                   const Loop::GF3D2<const TC> &vJ_da_dy,
                   const Loop::GF3D2<const TC> &vJ_da_dz,
                   const Loop::GF3D2<const TC> &vJ_db_dx,
                   const Loop::GF3D2<const TC> &vJ_db_dy,
                   const Loop::GF3D2<const TC> &vJ_db_dz,
                   const Loop::GF3D2<const TC> &vJ_dc_dx,
                   const Loop::GF3D2<const TC> &vJ_dc_dy,
                   const Loop::GF3D2<const TC> &vJ_dc_dz, const CCTK_REAL var0,
                   const CCTK_REAL v0, const CCTK_REAL radpower);

} // namespace NewRadX

#endif // NEWRADX_HXX
