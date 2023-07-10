/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(spherharm,PairSH)

#else

#ifndef LMP_PAIR_SH_H
#define LMP_PAIR_SH_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSH : public Pair {
 public:
  PairSH(class LAMMPS *);
  virtual ~PairSH();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);

 protected:
  double **cut{};

  class AtomVecSpherharm *avec{};

  virtual void allocate();

 private:

  // per-type coefficients, set in pair coeff command
  double ***normal_coeffs{};
  int *typetosh{};
  int matchtypes;
  double exponent;

  void matchtype();
  int refine_cap_angle(int &kk_count, int ishtype, int jshtype, double iang,  double radj,
                       double (&iquat_cont)[4], double (&iquat_sf_bf)[4], const double xi[3],
                       const double xj[3], double (&jrot)[3][3]);

double find_inner_radius_directly(double rad_body,double radtol, double theta_sf, double phi_sf,double theta_proj, double phi_proj,const double xi[3], const double xj[3], double radi,  double radj,int ishtype,int jshtype,double (&irot)[3][3],double (&jrot)[3][3]);

// double find_inner_radius(const double ix_sf[3],
//                                             const double xi[3], const double xj[3],  double radj, int jshtype,  double (&jrot)[3][3]);

  void calc_norm_force_torque(int kk_count, int ishtype, int jshtype, double iang, double radi,  double radj,
                         double (&iquat_cont)[4], double (&iquat_sf_bf)[4], const double xi[3],
                         const double xj[3], double (&irot)[3][3],  double (&jrot)[3][3],
                         double &vol_overlap, double (&iforce)[3], double (&torsum)[3],
                         double &factor, bool &first_call, int ii, int jj);
  void calc_tang_force_torque(double mu, int ishtype, int jshtype, double const (&normforce)[3], double const (&vr)[3],
                           double const (&omegaa)[3], double const (&omegab)[3], double const (&cp)[3],
                           double const (&rot_sf_bf_a)[3][3], double const (&rot_sf_bf_b)[3][3],
                           double (&tforce)[3]);
  static void sphere_sphere_norm_force_torque(double ri, double rj, double delta, const double x1[3],
                                       const double x2[3], double (&iforce)[3], double (&torsum)[3],
                                       double &voloverlap);
  double find_intersection_by_bisection(double rad_body, double radtol, double theta_sf, double phi_sf,
                                                const double xi[3], const double xj[3], double radj, int jshtype,
                                                double (&jrot)[3][3]);
  double find_intersection_by_newton(const double ix_sf[3], const double xi[3], const double xj[3],
                                     double theta_n, double phi_n, double t_n, double radtol,
                                     int sht, const double jrot[3][3]);

               

  double cur_time;
  int file_count;
  int write_surfpoints_to_file(double *x, bool append_file, int cont, int ifnorm, double *norm) const;
 int write_vol_overlap_to_file(int, int, double (&iforce)[3], double, double, bool);
  // Gaussian quadrature arrays
  double *abscissa{};          // Abscissa of gaussian quadrature (same for all shapes)
  double *weights{};           // Weights of gaussian quadrature (same for all shapes)
  int num_pole_quad;
  double radius_tol;

  void get_quadrature_values(int num_quadrature);

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
