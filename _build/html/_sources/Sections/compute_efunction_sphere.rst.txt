.. index:: compute efunction/spherharm

compute erotate/spherharm command
===================================

Syntax
""""""

.. parsed-literal::

   compute ID group-ID efunction/spherespherharm

* ID, group-ID are documented in :doc:`compute <compute>` command
* efunction/spherharm = style name of this compute command

Examples
""""""""

.. code-block:: LAMMPS

   compute 1 all efunction/spherharm

Description
"""""""""""

Define a computation that calculates the potential energy of
a group of spherical harmonic particles.

The rotational energy is computed as 1/2 I w\^2, where I is the moment
of inertia for a sphere and w is the particle's angular velocity.


Output info
"""""""""""

This compute calculates a global scalar (the KE).  This value can be
used by any command that uses a global scalar value from a compute as
input.  See the :doc:`Howto output <Howto_output>` page for an
overview of LAMMPS output options.

The scalar value calculated by this compute is "extensive".  The
scalar value will be in energy :doc:`units <units>`.

Restrictions
""""""""""""

This compute requires that atoms store a radius and angular velocity
(omega) as defined by the :doc:`atom_style spherharm <atom_style>` command.


Related commands
""""""""""""""""

:doc:`compute erotate/asphere <compute_erotate_asphere>`

Default
"""""""

none
