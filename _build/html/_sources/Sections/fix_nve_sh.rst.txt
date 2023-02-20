.. index:: fix nve/sh

fix nve/sh
==========================

Syntax
""""""

.. parsed-literal::

   fix ID group-ID nve/sh

* ID, group-ID are documented in :doc:`fix <fix>` command
* nve/sh = style name of this fix command

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all nve/sh

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

