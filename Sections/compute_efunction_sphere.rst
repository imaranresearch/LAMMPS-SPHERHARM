.. index:: compute efunction/spherharm

compute efunction/spherharm command
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

The work done by the potential over a time step is a combination of that done from the "back half" :math:`t -> t+dt/2` and the "front half" :math:`t+dt/2->t+dt` of the time step. At the end of the time step, i.e :math:`t+dt`, it is not possible to calculate the "back half" of the work done, so this must be carried forward from the previous time step. 
These compute styles calculate global scalars and can be used by other commands that use a global scalar value from a compute as input.



Output info
"""""""""""

This compute calculates a global scalar (the PE).  This value can be
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

:doc:`compute erotate/spherharm <compute_erotate_spherharm>`

Default
"""""""

none
