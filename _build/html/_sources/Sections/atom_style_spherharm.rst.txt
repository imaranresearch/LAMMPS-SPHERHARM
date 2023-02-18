.. index:: atom_style spherharm

atom_style spherharm command
==============================

Syntax
""""""

.. parsed-literal::

   atom_style spherharm N numerical_quad1 shape_coeff.dat

* ID, group-ID are documented in atom_style  command
* spherharm = style name of this atom_style command

Examples
""""""""

.. code-block:: LAMMPS

   atom_style spherharm 21 45 shape_1.dat

Description
"""""""""""

Perform constant NVE integration to update position, velocity, angular
velocity, and quaternion orientation for spherical harmonic 
particles in the group each timestep.  V is volume; E is energy.  This
creates a system trajectory consistent with the microcanonical
ensemble.

This fix differs from the :doc:`fix nve <fix_nve>` command, which
assumes point particles and only updates their position and velocity.
It also differs from the :doc:`fix nve/bpm/sphere <fix_nve_bpm_sphere>`
command which assumes only finite-size spherical particles.


----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>`.  None of the :doc:`fix_modify <fix_modify>` options are
relevant to this fix.  No global or per-atom quantities are stored by
this fix for access by various :doc:`output commands <Howto_output>`.
No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during
:doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix requires that atoms store torque, angular velocity (omega), a
radius, and a quaternion as defined by the :doc:`atom_style spherharm
<atom_style>` command.

All particles in the group must be finite-size particles with
quaternions.  They cannot be point particles.

Related commands
""""""""""""""""

:doc:`fix nve <fix_nve>`, :doc:`fix nve/bpm/sphere <fix_nve_bpm_sphere>`

Default
"""""""

none

