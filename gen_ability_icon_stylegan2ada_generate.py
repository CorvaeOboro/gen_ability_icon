#import stylegan2-ada-pytorch.generate as stylegan2adagen
import random
import datetime
import subprocess
import os.path
from os import path
from os.path import abspath
import requests

# DOWNLOAD ABILITY ICON trained network checkpoint pkl
network="./datasets/gen_ability_icon_stylegan2ada_20220801.pkl"
if path.exists(network) :
        print("EXISTING PKL FOUND == " + str(network))
else:
        save_output_path = "./datasets/"
        try: 
                os.mkdir(save_output_path) 
        except OSError as error: 
                print(error)  
        network_absolute = abspath(network)
        print("GEN ABILITY PKL MISSING  == " + str(network_absolute))
        network_url = '''https://github.com/CorvaeOboro/gen_ability_icon/releases/download/gen_ability_icon_stylegan2ada_20220801/gen_ability_icon_stylegan2ada_20220801.pkl'''
        print("DOWNLOAD PKL  == " + str(network_url))
        response = requests.get(network_url)
        print("DOWNLOADING....")
        with open(network_absolute, 'wb') as f:
                f.write(response.content)
        print("====DOWNLOAD ABILITY ICON pkl COMPLETED====")

# RANDOM SEED
now = datetime.datetime.now()
timebased_seed = int(now.strftime("%Y%m%d%H%M%S"))
random.seed(timebased_seed)
random_seed = int(random.random()*2147483648.0)

# GENERATE MULTIPLE ICONS
total_icons = 100
truncation = 0.5  # truncation is like chaos , 0 = average 1 = furthest outliers of network
random_seed_end = random_seed + total_icons
seeds_final = str(random_seed) +"-" +str(random_seed_end)

# OUTPUT FOLDER
outdir="./output"
try: 
    os.mkdir(outdir) 
except OSError as error: 
    print(outdir + "  EXISTS")  

# GENERATE STYLEGAN2ADA SEEDS //=====================================================
stylegan2ada_generate="./stylegan2-ada-pytorch/generate.py"
if path.exists(stylegan2ada_generate) :
        print("EXISTING STYLEGAN2ADA FOUND == " + str(stylegan2ada_generate))
        p = subprocess.call(['python', 'stylegan2-ada-pytorch/generate.py', '--seeds', str(seeds_final), '--trunc', str(truncation), '--outdir', str(outdir), '--network', str(network)])
else:
        print("==== ERROR - STYLEGAN2ADAPYTORCH NOT FOUND , GIT CLONE ====")


