
echo $PYTHONPATH
::set PATH=C:\Program Files\Python 3.9;%PATH%
::set PYTHONPATH=C:\Program Files\Python 3.9;%PYTHONPATH%

pip install psd_tools

python gen_ability_icon_collage.py --input_path="./icons/" --resolution=256
pause