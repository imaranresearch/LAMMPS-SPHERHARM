Validation tests
========================
Energy conservation test
""""""""""""""""""""""""""""""""""""""""""

The elastic collision between two irregularly shaped particles is used as the first benchmark test to verify and demonstrate the energy conservation of the spherical-harmonic-based contact detection algorithm and the volume-based energy-conserving contact model.  Two identical and randomly oriented particles are generated with the spherical harmonic degree of expansion of $N=10$. Initially, these particles with equal and opposite velocities are separated by a smaller distance.  


.. code-block:: LAMMPS

    # Test

    variable	name string spherical_harmonics_testing

    atom_style	spherharm 1	120 A-anm-00001.dat
    units		si
    newton off
    # newton on

    ###############################################
    # Geometry-related parameters
    ###############################################

    variable	boxx equal 100
    variable	boxy equal 100
    variable	boxz equal 100


    #############
    processors * * 1
    region		boxreg block 0 ${boxx} 0 ${boxy} 0 ${boxz}
    create_box	2 boxreg
    change_box	all boundary f f f

    create_atoms 	1 single 35.5 50.0 50.0
    create_atoms 	2 single 52.5 50.0 50.0

    #mass *      1.0

    set 		atom 1 vx 1.0 vy 0.0 vz 0.0
    set 		atom 2 vx -1.0 vy 0.0 vz 0.0

    set		atom 1 sh/shape 1
    set		atom 2 sh/shape 1
    set		group all sh/quat/random 4

    fix		time_fix all nve/sh

    neigh_modify	delay 0 every 1 check yes
    comm_modify	vel yes

    thermo 1

    compute TransKE all ke
    compute RotKE all erotate/spherharm
    compute PEcalc all efunction/spherharm
    variable ETotal equal "c_PEcalc + c_TransKE + c_RotKE"
    variable ERot equal "C_RotKE"
    thermo_style custom step c_PEcalc c_TransKE c_RotKE v_ETotal

    pair_style 	spherharm
    pair_coeff 	* * 5.0e-6 2.0   30

    timestep	1.0e-3
    variable 	step equal step
    #fix     1 all ave/time 100 1 100 c_PEcalc c_TransKE c_RotKE v_ETotal  file energy_spher.dat
    #fix 		2 all print 1 "${step} ${ETotal}" file energy_spher.dat screen no
    run		1000


The normal stiffness coefficient :math:`k_n` is fixed to allow a large overlap. Fig. 1 displays the total energy of the system over time, where the contact work done is the sum of the incremental contact work done:

.. math::
    d W = -d t ({F}\cdot {v}+ {M}\cdot {\omega})

.. figure:: ../Figures/Energy_balance_elastic_impact.png
    :width: 40%
    :align: center
    
    Plot of energy (total, translational, rotational, and contact work) over time for an impact and excessive overlap of two particles

Absfsifdoa
