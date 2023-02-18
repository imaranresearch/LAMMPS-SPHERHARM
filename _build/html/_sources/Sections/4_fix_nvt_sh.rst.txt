
fix nvt/sh command
======================


Syntax
""""""

.. parsed-literal::

   fix ID group-ID nvt/sphere keyword value ...

* ID, group-ID are documented in 'fix' command
* nvt/sphere = style name of this fix command
* zero or more keyword/value pairs may be appended
* keyword = *disc*

  .. parsed-literal::

       *disc* value = none = treat particles as 2d discs, not spheres

* additional thermostat related keyword/value pairs from the command can be appended

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all nvt/sphere temp 300.0 300.0 100.0
   fix 1 all nvt/sphere temp 300.0 300.0 100.0 disc
   fix 1 all nvt/sphere temp 300.0 300.0 100.0 drag 0.2

Description
"""""""""""

Perform constant NVT integration to update position, velocity, and
angular velocity each timestep for finite-size spherical particles in
the group using a Nose/Hoover temperature thermostat.  V is volume; T
is temperature.  This creates a system trajectory consistent with the
canonical ensemble.

Default
"""""""

none
