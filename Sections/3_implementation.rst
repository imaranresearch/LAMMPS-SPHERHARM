Implementation of SPHERHARM in LAMMPS
=====================================


Atom style
-------------
Atom style defines the set of properties of an atom used during a simulation in communication and input-output operations. Unlike traditional atom styles in LAMMPS, spherical harmonic particles are constructed in the simulation by supplying the shape coefficients. Any instance of this atom style accesses the appropriate coefficients. Spherical harmonic representation of the particles requires the multiple per-shape variables: shape coefficients :math:`a_{nm}`, principle inertia :math:`I_{1,2,3}`, initial quaternion :math:`q_i`, and degree of spherical harmonic expansion :math:`n`. Modified set commands further link these per-shape variables to per-atom or per-group properties. In order to add support for these quantities, a new data structure was created, which can be accessed using the following command:




Syntax
""""""

.. parsed-literal::

   atom_style spherharm N numerical_quad1 shape_coeff.dat

* ID, group-ID are documented in compute command
* erotate/sphere = style name of this compute command

.. note:: 
   
   It should be noted that  per-shape properties are not the same as the per-type properties. A spherical harmonic atom type can contain only one shape, although a shape can belong to multiple types of spherical harmonic particles.

Pair style
------------------

In LAMMPS, two-body and multi-body interactions are implemented as the inter-particle potentials. In the current implementation of the *SPHERHARM*, inter-particle interaction is calculated as a function of the overlapped volume between the particles [feng paper]. 
The particles are represented by their bounding spheres when building neighbour lists and checking for potential contacts. Calculation of overlap volume,  in addition to other properties required by the contact theory, is handled through numerical integration over the spherical caps formed by the overlap of bounding spheres as detailed in section [ref section contact detection]. This is managed through the addition of standalone routines for spherical harmonic and quadrature functions, defined similarly to those in  MathExtra. The commands for pair style are as follows:

Syntax
""""""""

.. code-block:: LAMMPS

   pair_style spherharm
   pair_coeff I J Kn exponent numerical_quad2


Fix style
--------------------


The dynamics of a model are implemented by the use of different fix styles in LAMMPS. It contains a list of commands to perform specified operations during the dynamic time step. Two variants of the fix are implemented in the *SPHERHARM* package. Due to additional properties of the spherical harmonic particles, i.e. quaternion, and shtype, a new method based on the Velocity Verlet algorithm for time-stepping has been added. It can be accessed by the following command. 


fix NVE/sh
""""""""""""""""
In the simulation of granular particles, walls are often used to describe the boundary condition or flow to a certain region in space. This boundary condition can be stationary as well as moving. LAMMPS offers various fix wall/* commands to implement such rigid boundaries. However, these walls can not be used with a particle defined using the spherharm atom style. The spherharm atom style necessitates using the spherharm pair style to compute the interaction forces. Thus the newly implemented walls calculate the repulsive force for a particle-wall interaction by following the same methodology as for particle-particle interactions. The tangential friction force is calculated by  *calc_vel Coulomb_force_torque*  method, based on the velocity-dependent Coulomb friction described in \texttt{pair granular}.
A suitable wall surface can be  provided by


Syntax
""""""""


.. code-block:: LAMMPS

   fix ID group_ID wall/spherharm Kn exponent Kt wallstyle args keyword values

This wall style takes **translate** keyword and velocities :math: `v_x, v_y, v_z`  as values to implement moving walls. 

Compute styles
----------------------

Compute styles in LAMMPS are used to calculate the properties of the system at different instances as they are invoked. They operate on specified groups or chunks of atoms, and they produce output which is stored internally for use by other commands. In *SPHERHARM* package, two compute styles are implemented to calculate the system's rotational kinetic energy o and potential energy. The following compute style is used to calculate the rotational kinetic energy o of a group of spherical harmonic particles.

.. code-block:: LAMMPS

   compute ID group-ID erotate/spherharm

The rotational energy is computed as :math:`E_{rot} = 0.5*I.\omega^2`, where :math: `I` is the moment of inertia  and :math:`\omega` is the particle's angular velocity. Similarly, another compute is implemented to calculate the potential energy of a particle group by calculating the incremental work done by the potential. This compute style can be invoked by 

.. code-block:: LAMMPS
   
   compute ID group-ID efunction/spherharm

The work done by the potential over a time step is a combination of that done from the "back half" :math:`t -> t+dt/2` and the "front half" :math:`t+dt/2->t+dt` of the time step. At the end of the time step, i.e :math:`t+dt`, it is not possible to calculate the "back half" of the work done, so this must be carried forward from the previous time step. 
These compute styles calculate global scalars and can be used by other commands that use a global scalar value from a compute as input.
