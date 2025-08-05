# HTSims_GIT
Github oriented version of HTSims so that version control can be managed

  main_shreyas.py - contains the original code from which we started bulding
  
  htsims_v1.py - contains our version of the code with updated formulas for convection and changed solver
  
  htsims_v2.py - updated version of v1 where we have accounted the mount btw esc and bf as a separate entity and have written the equations accordingly

  htsims_v2_1.py -updated version of v2 where the ESC mount is attached to the shell of the nacelle as a heat mitigation plan, physics and equations are yet to be confirmed but the results seem completely normal
  
  htsims_v3.py - updated version of v1 shich includes day and night variation and external radiation as well as solar data collection script but does not account for mount as a separate entity
  
  htsims_vM - master version of the code containing the most advanced scripts with everything from v1, v2, v3 and broken into multiple different script for simplicity and further updation 

  sens_analysis - trying to do a sensitivity analysis on the code pref v2 to check what all variables affect the final result the most when perturbed 

  sens_analysis_v2 - updated version of v1 which will also give us all the heat paths for every node so as to give us a better understanding of the effectivness of different modes of heat transfer 