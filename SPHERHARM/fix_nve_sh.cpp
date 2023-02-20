/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------
   Contributing authors: James Young (UoE)
                         Mohammad Imaran (UoE)
                         Kevin Hanley (UoE)

   Please cite the related publication:
   TBC
------------------------------------------------------------------------- */

#include "fix_nve_sh.h"
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "error.h"
#include "atom_vec_spherharm.h"


using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVESh::FixNVESh(LAMMPS *lmp, int narg, char **arg) :
  FixNVE(lmp, narg, arg) {}

/* ---------------------------------------------------------------------- */

void FixNVESh::init()
{
  avec = (AtomVecSpherharm *) atom->style_match("spherharm");
  if (!avec)
    error->all(FLERR,"Fix nve/sh requires atom style spherharm");

  // check that all particles are finite-size ellipsoids
  // no point particles allowed, spherical is OK

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *shtype = atom->shtype;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      if (shtype[i] < 0)
        error->one(FLERR,"Fix nve/sh requires spherical harmonic particles");

  FixNVE::init();
}

/* ---------------------------------------------------------------------- */

void FixNVESh::initial_integrate(int /*vflag*/)
{
  double dtfm;

  double **omega = atom->omega;
  double **quat = atom->quat;
  double **inertia = atom->pinertia_byshape;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **angmom = atom->angmom;
  double **torque = atom->torque;
  double *mass = atom->rmass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int *shtype = atom->shtype;
  int ishtype;

  // set timestep here since dt may have changed or come via rRESPA

  dtq = 0.5 * dtv;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {

      dtfm = dtf / mass[i];
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];

      // update angular momentum by 1/2 step

      angmom[i][0] += dtf * torque[i][0];
      angmom[i][1] += dtf * torque[i][1];
      angmom[i][2] += dtf * torque[i][2];

      ishtype = shtype[i];

      // compute omega at 1/2 step from angmom at 1/2 step and current q
      // update quaternion a full step via Richardson iteration
      // returns new normalized quaternion

      MathExtra::mq_to_omega(angmom[i],quat[i],inertia[ishtype],omega[i]);
      MathExtra::richardson(quat[i],angmom[i],omega[i],inertia[ishtype],dtq);
    }
}

/* ---------------------------------------------------------------------- */

void FixNVESh::final_integrate()
{
  double dtfm;

  double **v = atom->v;
  double **f = atom->f;
  double **angmom = atom->angmom;
  double **torque = atom->torque;
  double *mass = atom->rmass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {

      dtfm = dtf / mass[i];
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];

      angmom[i][0] += dtf * torque[i][0];
      angmom[i][1] += dtf * torque[i][1];
      angmom[i][2] += dtf * torque[i][2];
    }
}
