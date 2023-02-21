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

Atom style defines the set of properties of an atom used during a simulation in communication and input-output operations. Unlike traditional atom styles in LAMMPS, spherical harmonic particles are constructed in the simulation by supplying the shape coefficients. Any instance of this atom style accesses the appropriate coefficients. Spherical harmonic representation of the particles requires the multiple per-shape variables: shape coefficients :math:`a_{nm}`, principle inertia :math:`I_{1,2,3}`, initial quaternion :math:`q_i`, and degree of spherical harmonic expansion :math:`n`. Modified set commands further link these per-shape variables to per-atom or per-group properties.


.. note:: 
   
   It should be noted that  per-shape properties are not the same as the per-type properties. A spherical harmonic atom type can contain only one shape, although a shape can belong to multiple types of spherical harmonic particles.



----------


Restrictions
""""""""""""

This fix requires that atoms store torque, angular velocity (omega), a
radius, and a quaternion as defined by the :doc:`atom_style spherharm
<atom_style>` command.

All particles in the group must be finite-size particles with
quaternions.  They cannot be point particles.

Related commands
""""""""""""""""

:doc:`read_data, pair_style`

Default
"""""""

none

