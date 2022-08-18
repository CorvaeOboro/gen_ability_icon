# RUN HOUDINI , GENERATE PDG TOPs for ABILITY ICONS
# set you local houdini directory as variable
import os

houdini_directory = "C:/PROGRAMS/Houdini_18_5_633/"

houdinipythonlib = houdini_directory + "houdini/python2.7libs/pdgjob/topcook.py"
hythonCmd = houdini_directory + "bin/hython.exe " + houdinipythonlib 

current1 = hythonCmd + " --hip houdini/GEN_ABILITY_ICON.hip --toppath /obj/topnet1/output0"
print("Executing first render ", current1)
os.system(current1)