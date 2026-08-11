#include <cctk.h>
#include <cctk_Parameters.h>

#include <loop_device.hxx>
#include <driver.hxx>

#include "newradx.hxx"

#include <cmath>
#include <type_traits>

namespace NewRadX {

using namespace Loop;
using namespace std;

/* Mixed precision (see newradx.hxx): the state variable and its RHS -- and,
 * for the multipatch overload, the coordinates and Jacobians -- are read and
 * written at whatever width they are declared with, but all of the arithmetic
 * in between is done in CCTK_REAL. Reasons:
 *
 *   - Everything the grid function values are combined with is CCTK_REAL
 *     anyway (p.x/p.y/p.z, p.DX, var0, v0, radpower), so for the pre-existing
 *     CCTK_REAL case this is bit-for-bit the previous behaviour, and for
 *     CCTK_REAL4 grid functions the usual C++ promotion rules would have
 *     produced the same thing.
 *   - The stencils below take differences of neighbouring values of a smooth
 *     field, which cancels leading digits; accumulated in CCTK_REAL2 (11
 *     significant bits) such a derivative would lose most of its
 *     significance.
 *   - This routine only touches the outermost layer of interior points, so
 *     the wider arithmetic costs next to nothing.
 *   - Under nvcc CCTK_REAL2 is `__half`, whose arithmetic operators are
 *     ambiguous for mixed operands (`__half * double`), so the values have to
 *     be converted explicitly in any case.
 *
 * Every such conversion is spelled `CCTK_REAL(gf(I))` on read and `T(value)`
 * on write, and is the only place where the declared width enters.
 */

/* The explicit instantiations at the bottom of this file cover CCTK_REAL8,
 * CCTK_REAL4 and CCTK_REAL2; CCTK_REAL aliases one of those unless the flesh
 * was configured with REAL_PRECISION=16, which CarpetX does not support
 * (amrex::Real is double). Fail loudly rather than with an undefined symbol.
 */
#ifdef HAVE_CCTK_REAL4
static_assert(std::is_same_v<CCTK_REAL, CCTK_REAL8> ||
                  std::is_same_v<CCTK_REAL, CCTK_REAL4>,
              "NewRadX supports CCTK_REAL == CCTK_REAL8 or CCTK_REAL4");
#else
static_assert(std::is_same_v<CCTK_REAL, CCTK_REAL8>,
              "NewRadX supports CCTK_REAL == CCTK_REAL8");
#endif

template <typename T> static inline constexpr T pow2(const T x) {
  return x * x;
}

template <std::size_t dir, typename T>
static inline CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_DEVICE CCTK_REAL
c2o(const Loop::PointDesc &p, const vect<int, dim> &pI,
    const Loop::GF3D2<const T> &gf) noexcept {
  const CCTK_REAL num{CCTK_REAL(gf(pI + p.DI[dir])) -
                      CCTK_REAL(gf(pI - p.DI[dir]))};
  const CCTK_REAL den{1.0 / (2.0 * p.DX[dir])};
  return num * den;
}

template <std::size_t dir, typename T>
static inline CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_DEVICE CCTK_REAL
l2o(const Loop::PointDesc &p, const vect<int, dim> &pI,
    const Loop::GF3D2<const T> &gf) noexcept {
  const CCTK_REAL num{-3.0 * CCTK_REAL(gf(pI)) +
                      4.0 * CCTK_REAL(gf(pI + p.DI[dir])) -
                      CCTK_REAL(gf(pI + 2 * p.DI[dir]))};
  const CCTK_REAL den{1.0 / (2.0 * p.DX[dir])};
  return num * den;
}

template <std::size_t dir, typename T>
static inline CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_DEVICE CCTK_REAL
r2o(const Loop::PointDesc &p, const vect<int, dim> &pI,
    const Loop::GF3D2<const T> &gf) noexcept {
  const CCTK_REAL num{3.0 * CCTK_REAL(gf(pI)) -
                      4.0 * CCTK_REAL(gf(pI - p.DI[dir])) +
                      CCTK_REAL(gf(pI - 2 * p.DI[dir]))};
  const CCTK_REAL den{1.0 / (2.0 * p.DX[dir])};
  return num * den;
}

template <std::size_t dir, typename T>
static inline CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_DEVICE CCTK_REAL
calc_deriv(const Loop::PointDesc &p,
           const Loop::GF3D2<const T> &gf) noexcept {
  if (p.NI[dir] == 0) {
    /* interior
     * in direction parallel to face/edge, apply symmetric stencil
     * currently second-order accurate
     * TODO: apply same finite-difference order as interior
     */
    return c2o<dir>(p, p.I, gf);

  } else if (p.NI[dir] == +1) {
    /* upper boundary
     * in direction normal to face/edge/corner, apply asymmetric stencil
     * currently second-order accurate
     * TODO: apply same finite-difference order as interior
     */
    return r2o<dir>(p, p.I, gf);

  } else if (p.NI[dir] == -1) {
    /* lower boundary
     * in direction normal to face/edge/corner, apply asymmetric stencil
     * currently second-order accurate
     * TODO: apply same finite-difference order as interior
     */
    return l2o<dir>(p, p.I, gf);

  } else {
#if !defined(__CUDACC__) && !defined(__HIP_PLATFORM_AMD__) &&                  \
    !defined(__HIP_PLATFORM_HCC__) && !defined(__INTEL_LLVM_COMPILER)
    CCTK_VERROR("Internal error: Unreconized point location %d", p.NI[dir]);
#endif
    amrex::Abort();
  }
}

template <typename T>
void NewRadX_Apply(const cGH *restrict const cctkGH,
                   const Loop::GF3D2<const T> &var, const Loop::GF3D2<T> &rhs,
                   const CCTK_REAL var0, const CCTK_REAL v0,
                   const CCTK_REAL radpower) {
  DECLARE_CCTK_ARGUMENTS;

  const auto symmetries = CarpetX::ghext->patchdata.at(cctk_patch).symmetries;
  const vect<vect<bool, Loop::dim>, 2> is_sym_bnd{
      {symmetries[0][0] != CarpetX::symmetry_t::none,
       symmetries[0][1] != CarpetX::symmetry_t::none,
       symmetries[0][2] != CarpetX::symmetry_t::none},
      {symmetries[1][0] != CarpetX::symmetry_t::none,
       symmetries[1][1] != CarpetX::symmetry_t::none,
       symmetries[1][2] != CarpetX::symmetry_t::none}};

  const Loop::GridDescBaseDevice grid(cctkGH);
  grid.loop_outermost_int_device<0, 0, 0>(
      grid.nghostzones, is_sym_bnd,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        // The main part of the boundary condition assumes that we have an
        // outgoing radial wave with some speed v0:
        //
        //    var  =  var0 + u(r-v0*t)/r
        //
        // This implies the following differential equation:
        //
        //    d_t var  =  - v^i d_i var  -  v0 (var - var0) / r
        //
        // where  vi = v0 xi/r

        // coordinate radius at p.I
        const auto r = sqrt(pow2(p.x) + pow2(p.y) + pow2(p.z));

        // Find local wave speeds at radiative boundary point p.I
        const auto vx = v0 * p.x / r;
        const auto vy = v0 * p.y / r;
        const auto vz = v0 * p.z / r;

        // Derivatives
        const auto varx = calc_deriv<0>(p, var);
        const auto vary = calc_deriv<1>(p, var);
        const auto varz = calc_deriv<2>(p, var);

        // radiative rhs
        //
        // Mixed precision: accumulated in CCTK_REAL and stored to rhs (at its
        // declared width) once, at the end -- narrowing an intermediate
        // result and reading it back would round twice. The Coulomb
        // correction below reads rhs at the *interior* point intp, never at
        // p.I, so the deferred store changes nothing else.
        CCTK_REAL rhs_val = -vx * varx - vy * vary - vz * varz -
                            v0 * (CCTK_REAL(var(p.I)) - var0) / r;

        if (radpower > 0) {
          // When solution is known to have a Coulomb-like component
          // asymptotically, this is estimated and extrapolated from
          // physical interior points to the radiative boundary. I.e.:
          //
          //    var  =  var0 + u(r-v0*t)/r + h(t)/r^n
          //
          // This implies the following differential equation:
          //
          //    d_t var  =  - v^i d_i var  -  v0 (var - var0) / r
          //             + v0 (1 - n) h / r^n+1 + d_t h / r^n
          //             ~  - v^i d_i var  -  v0 (var - var0) / r
          //             + d_t h / r^n + O(1/r^n+1)
          //
          // where  vi = v0 xi/r
          //
          // the Coulomb term, d_t h / r ^n , is estimated by extrapolating
          // from the nearest interior points
          //
          // Displacement to get from p.I to interior point placed
          // nghostpoints away
          const vect<int, dim> displacement{grid.nghostzones[0] * p.NI[0],
                                            grid.nghostzones[1] * p.NI[1],
                                            grid.nghostzones[2] * p.NI[2]};
          const vect<int, dim> intp = p.I - displacement;

          assert(intp[0] >= grid.nghostzones[0]);
          assert(intp[1] >= grid.nghostzones[1]);
          assert(intp[2] >= grid.nghostzones[2]);
          assert(intp[0] <= grid.lsh[0] - grid.nghostzones[0] - 1);
          assert(intp[1] <= grid.lsh[1] - grid.nghostzones[1] - 1);
          assert(intp[2] <= grid.lsh[2] - grid.nghostzones[2] - 1);

          // coordinates at p.I-displacement
          const auto xint = p.x - displacement[0] * p.DX[0];
          const auto yint = p.y - displacement[1] * p.DX[1];
          const auto zint = p.z - displacement[2] * p.DX[2];
          const auto rint = sqrt(pow2(xint) + pow2(yint) + pow2(zint));

          // Find local wave speeds at physical point p.I-displacement
          const auto vxint = v0 * xint / rint;
          const auto vyint = v0 * yint / rint;
          const auto vzint = v0 * zint / rint;

          // Derivatives at physical point p.I-displacement
          const auto varxint = c2o<0>(p, intp, var);
          const auto varyint = c2o<1>(p, intp, var);
          const auto varzint = c2o<2>(p, intp, var);

          // Eextrapolate Coulomb component, rescale to account for radial
          // fall-off
          const auto rad = -vxint * varxint - vyint * varyint -
                           vzint * varzint -
                           v0 * (CCTK_REAL(var(intp)) - var0) / rint;
          const auto aux =
              (CCTK_REAL(rhs(intp)) - rad) * pow(rint / r, radpower);

          // Radiative rhs with extrapolated Coulomb correction
          rhs_val += aux;
        }

        rhs(p.I) = T(rhs_val);
      });
}

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
                   const CCTK_REAL v0, const CCTK_REAL radpower) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  const auto symmetries = CarpetX::ghext->patchdata.at(cctk_patch).symmetries;
  const vect<vect<bool, Loop::dim>, 2> is_sym_bnd{
      {symmetries[0][0] != CarpetX::symmetry_t::none,
       symmetries[0][1] != CarpetX::symmetry_t::none,
       symmetries[0][2] != CarpetX::symmetry_t::none},
      {symmetries[1][0] != CarpetX::symmetry_t::none,
       symmetries[1][1] != CarpetX::symmetry_t::none,
       symmetries[1][2] != CarpetX::symmetry_t::none}};

  const Loop::GridDescBaseDevice grid(cctkGH);
  grid.loop_outermost_int_device<0, 0, 0>(
      grid.nghostzones, is_sym_bnd,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        // Skip inner radial boundary unless requested
        if (!apply_inner_boundary && p.NI[2] == -1) {
          return;
        }

        // At the outer boundary absorb outgoing waves; at the inner
        // boundary absorb incoming waves (flip sign of v0).
        const CCTK_REAL sign = (p.NI[2] == -1) ? -1.0 : 1.0;
        const auto sv0 = sign * v0;

        // Find local wave speeds at radiative boundary point p.I
        const auto x = CCTK_REAL(vcoordx(p.I));
        const auto y = CCTK_REAL(vcoordy(p.I));
        const auto z = CCTK_REAL(vcoordz(p.I));
        const auto r = sqrt(pow2(x) + pow2(y) + pow2(z));

        const auto vx = sv0 * x / r;
        const auto vy = sv0 * y / r;
        const auto vz = sv0 * z / r;

        // Local derivatives
        const auto dgf_da = calc_deriv<0>(p, var);
        const auto dgf_db = calc_deriv<1>(p, var);
        const auto dgf_dc = calc_deriv<2>(p, var);

        // Jacobians
        const auto da_dx = CCTK_REAL(vJ_da_dx(p.I));
        const auto da_dy = CCTK_REAL(vJ_da_dy(p.I));
        const auto da_dz = CCTK_REAL(vJ_da_dz(p.I));
        const auto db_dx = CCTK_REAL(vJ_db_dx(p.I));
        const auto db_dy = CCTK_REAL(vJ_db_dy(p.I));
        const auto db_dz = CCTK_REAL(vJ_db_dz(p.I));
        const auto dc_dx = CCTK_REAL(vJ_dc_dx(p.I));
        const auto dc_dy = CCTK_REAL(vJ_dc_dy(p.I));
        const auto dc_dz = CCTK_REAL(vJ_dc_dz(p.I));

        // Global derivatives
        const auto dgf_dx = dgf_da * da_dx + dgf_db * db_dx + dgf_dc * dc_dx;
        const auto dgf_dy = dgf_da * da_dy + dgf_db * db_dy + dgf_dc * dc_dy;
        const auto dgf_dz = dgf_da * da_dz + dgf_db * db_dz + dgf_dc * dc_dz;

        // radiative rhs
        //
        // Mixed precision: accumulated in CCTK_REAL and stored to rhs (at its
        // declared width) once, at the end -- see the single-patch overload
        // above.
        CCTK_REAL rhs_val = -vx * dgf_dx - vy * dgf_dy - vz * dgf_dz -
                            sv0 * (CCTK_REAL(var(p.I)) - var0) / r;

        // Coulomb correction: estimate and extrapolate the 1/r^n
        // component from interior points to the radiative boundary.
        // Only applied at the outer boundary — at the inner boundary
        // the asymptotic 1/r^n assumption breaks down (r is small,
        // rint > r, so the rescaling amplifies rather than attenuates).
        if (radpower > 0.0 && p.NI[2] != -1) {
          // Displacement to get from p.I to interior point placed
          // nghostpoints away
          const vect<int, dim> displacement{grid.nghostzones[0] * p.NI[0],
                                            grid.nghostzones[1] * p.NI[1],
                                            grid.nghostzones[2] * p.NI[2]};
          const vect<int, dim> intp = p.I - displacement;

          assert(intp[0] >= grid.nghostzones[0]);
          assert(intp[1] >= grid.nghostzones[1]);
          assert(intp[2] >= grid.nghostzones[2]);
          assert(intp[0] <= grid.lsh[0] - grid.nghostzones[0] - 1);
          assert(intp[1] <= grid.lsh[1] - grid.nghostzones[1] - 1);
          assert(intp[2] <= grid.lsh[2] - grid.nghostzones[2] - 1);

          // Global coordinates at interior point
          const auto xint = CCTK_REAL(vcoordx(intp));
          const auto yint = CCTK_REAL(vcoordy(intp));
          const auto zint = CCTK_REAL(vcoordz(intp));
          const auto rint = sqrt(pow2(xint) + pow2(yint) + pow2(zint));

          // Find local wave speeds at interior point
          const auto vxint = sv0 * xint / rint;
          const auto vyint = sv0 * yint / rint;
          const auto vzint = sv0 * zint / rint;

          // Local derivatives at interior point
          const auto dgf_da = c2o<0>(p, intp, var);
          const auto dgf_db = c2o<1>(p, intp, var);
          const auto dgf_dc = c2o<2>(p, intp, var);

          // Jacobians at interior point
          const auto da_dx = CCTK_REAL(vJ_da_dx(intp));
          const auto da_dy = CCTK_REAL(vJ_da_dy(intp));
          const auto da_dz = CCTK_REAL(vJ_da_dz(intp));
          const auto db_dx = CCTK_REAL(vJ_db_dx(intp));
          const auto db_dy = CCTK_REAL(vJ_db_dy(intp));
          const auto db_dz = CCTK_REAL(vJ_db_dz(intp));
          const auto dc_dx = CCTK_REAL(vJ_dc_dx(intp));
          const auto dc_dy = CCTK_REAL(vJ_dc_dy(intp));
          const auto dc_dz = CCTK_REAL(vJ_dc_dz(intp));

          // Global derivatives at interior point
          const auto dgf_dx = dgf_da * da_dx + dgf_db * db_dx + dgf_dc * dc_dx;
          const auto dgf_dy = dgf_da * da_dy + dgf_db * db_dy + dgf_dc * dc_dy;
          const auto dgf_dz = dgf_da * da_dz + dgf_db * db_dz + dgf_dc * dc_dz;

          // Radiative part at interior point
          const auto rad = -vxint * dgf_dx - vyint * dgf_dy - vzint * dgf_dz -
                           sv0 * (CCTK_REAL(var(intp)) - var0) / rint;

          // Extrapolate Coulomb component, rescale to account for radial
          // fall-off
          const auto aux =
              (CCTK_REAL(rhs(intp)) - rad) * pow(rint / r, radpower);

          // Radiative rhs with extrapolated Coulomb correction
          rhs_val += aux;
        }

        rhs(p.I) = T(rhs_val);
      });
}

/* Explicit instantiations.
 *
 * Only these combinations exist; the header carries the declarations, so a
 * caller using any other width gets an undefined reference to
 * `NewRadX::NewRadX_Apply<...>` at link time.
 *
 * There is deliberately no separate instantiation for CCTK_REAL: it is a
 * preprocessor alias for one of the sized types (CCTK_REAL8 normally, see
 * cctk_Types.h), so a CCTK_REAL call site resolves to the instantiation for
 * that type -- and an extra `NEWRADX_INSTANTIATE_APPLY(CCTK_REAL)` would be a
 * duplicate instantiation error.
 *
 * CCTK_REAL8 (double) is assumed to exist, as it must for CarpetX
 * (amrex::Real). CCTK_REAL4 and CCTK_REAL2 are optional flesh types, hence
 * the guards; a build without them keeps exactly the pre-mixed-precision set
 * of instantiations.
 */
#define NEWRADX_INSTANTIATE_APPLY(T)                                           \
  template void NewRadX_Apply<T>(                                              \
      const cGH *restrict const, const Loop::GF3D2<const T> &,                 \
      const Loop::GF3D2<T> &, const CCTK_REAL, const CCTK_REAL,                \
      const CCTK_REAL);

#define NEWRADX_INSTANTIATE_APPLY_MULTIPATCH(T, TC)                            \
  template void NewRadX_Apply<T, TC>(                                          \
      const cGH *restrict const, const Loop::GF3D2<const T> &,                 \
      const Loop::GF3D2<T> &, const Loop::GF3D2<const TC> &,                   \
      const Loop::GF3D2<const TC> &, const Loop::GF3D2<const TC> &,            \
      const Loop::GF3D2<const TC> &, const Loop::GF3D2<const TC> &,            \
      const Loop::GF3D2<const TC> &, const Loop::GF3D2<const TC> &,            \
      const Loop::GF3D2<const TC> &, const Loop::GF3D2<const TC> &,            \
      const Loop::GF3D2<const TC> &, const Loop::GF3D2<const TC> &,            \
      const Loop::GF3D2<const TC> &, const CCTK_REAL, const CCTK_REAL,         \
      const CCTK_REAL);

// state variable in CCTK_REAL8
NEWRADX_INSTANTIATE_APPLY(CCTK_REAL8)
NEWRADX_INSTANTIATE_APPLY_MULTIPATCH(CCTK_REAL8, CCTK_REAL8)
#ifdef HAVE_CCTK_REAL4
NEWRADX_INSTANTIATE_APPLY_MULTIPATCH(CCTK_REAL8, CCTK_REAL4)
#endif
#ifdef HAVE_CCTK_REAL2
NEWRADX_INSTANTIATE_APPLY_MULTIPATCH(CCTK_REAL8, CCTK_REAL2)
#endif

// state variable in CCTK_REAL4
#ifdef HAVE_CCTK_REAL4
NEWRADX_INSTANTIATE_APPLY(CCTK_REAL4)
NEWRADX_INSTANTIATE_APPLY_MULTIPATCH(CCTK_REAL4, CCTK_REAL8)
NEWRADX_INSTANTIATE_APPLY_MULTIPATCH(CCTK_REAL4, CCTK_REAL4)
#ifdef HAVE_CCTK_REAL2
NEWRADX_INSTANTIATE_APPLY_MULTIPATCH(CCTK_REAL4, CCTK_REAL2)
#endif
#endif

// state variable in CCTK_REAL2
#ifdef HAVE_CCTK_REAL2
NEWRADX_INSTANTIATE_APPLY(CCTK_REAL2)
NEWRADX_INSTANTIATE_APPLY_MULTIPATCH(CCTK_REAL2, CCTK_REAL8)
#ifdef HAVE_CCTK_REAL4
NEWRADX_INSTANTIATE_APPLY_MULTIPATCH(CCTK_REAL2, CCTK_REAL4)
#endif
NEWRADX_INSTANTIATE_APPLY_MULTIPATCH(CCTK_REAL2, CCTK_REAL2)
#endif

#undef NEWRADX_INSTANTIATE_APPLY
#undef NEWRADX_INSTANTIATE_APPLY_MULTIPATCH

} // namespace NewRadX
