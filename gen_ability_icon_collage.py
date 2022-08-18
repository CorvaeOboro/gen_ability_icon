#=================================
# IMAGE_COLLAGE - a dataset expansion by mixing
# input directory of images , output randomized composited mixed images with randomized colors
# post processing on result with normalization , sharpening , contrast , saturation to avoid bland averaged results 
#=================================
from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageDraw, ImageFont ,ImageEnhance,ImageFilter , ImageChops , ImageOps
import PIL
import os.path
import re
import numpy as np
import colorsys
import random
import blend_modes
from pathlib import Path
import glob, os
import cv2
import datetime
import argparse

#//====================================================
#// ARGS PARSER
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help='folder path of images', default="./icons/")
parser.add_argument('--resolution', type=int, help='image resolution', default=256)

args = parser.parse_args()

#//====================================================
#// SEED DATETIME
current_datetime = datetime.datetime.now()
current_timeseed_string = (current_datetime.strftime("%Y%m%d%H%M%S")) + "00"
current_timeseed_int = int(current_timeseed_string)
print (str(current_timeseed_int))
input_seed = current_timeseed_int
#//====================================================
#// VARIABLE RANGES
#image_resolution = 1024
image_resolution = args.resolution
#input_directory = 'H:/CODE/DATASET/ICON/'
input_directory = args.input_path
total_number_iter = 200 # how many iterations
layers_num = 9 # min layers to stack 
layers_num_max = 14 # max layers to stack 
opacity_min = 0.25 # 0.25 min
opacity_max = 0.4 # 0.4 max default
rgb_opacity_min = 0.9 # default 0.2 , 
rgb_opacity_max = 1.1 # default rgb opacity max = 1.5
colorize_on = 0 # random hue
mirror_x = 1

normalize_final = 1 # clip percent 
normalize_final_blend = 0.7 # clip percent lerp
sharpen_final = 0.15 # 0.15 sharpen after mixing multiple low opacity images
brightness_final = 0.6 # darkens the image , after the normalization - default 0.6
contrast_final = 1.5 # increase the contrast after darkening
saturation_final = 1.5 # increases the final saturation after autotone and blending of randomized hue

noise_amount = 0.5 # 1.0
noise_blend_max = 0.01 # 0.1
brightness_range =  1.2 #  brightness_max multiplier , 1.5 default
sharpen_blend_amount = 0.05 # 0.4 sharpen
normalize_clahe_amount = 0.2 # 0.5 default
rotation_range = 1 # 1 degree of rotation 

regex_numbers_no_letters = re.compile('_[0-9]*_|_[0-9]*\.')
regex_underscore = re.compile('_')
regex_dot = re.compile('\.')

#//==== OUTPUT FOLDER  ===================================
save_output_path = "./output/"
try: 
    os.mkdir(save_output_path) 
except OSError as error: 
    print(save_output_path + "EXISTS")  

#//==== NOISE  ===================================
def add_pepper(image, amount):
  output = np.copy(np.array(image))

  # add salt
  nb_salt = np.ceil(amount * output.size * 0.5)
  coords = [np.random.randint(0, i - 1, int(nb_salt)) for i in output.shape]
  output[coords] = 1

  # add pepper
  nb_pepper = np.ceil(amount* output.size * 0.5)
  coords = [np.random.randint(0, i - 1, int(nb_pepper)) for i in output.shape]
  output[coords] = 0

  return Image.fromarray(output)

# SHIFT HUE TO DIFFENTIATE  / COLORIZE  ===============================
rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    # Colorize PIL image `original` with the given
    # `hue` (hue within 0-360); returns another PIL image.
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')
    return new_img

def colorizeRGB(image,local_seed):
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    ra, ga, ba, aa = np.rollaxis(arr, axis=-1)
    random.seed(local_seed*44)
    ra = ra*(random.uniform(rgb_opacity_min,rgb_opacity_max))
    random.seed(local_seed*55)
    ga = ga*(random.uniform(rgb_opacity_min,rgb_opacity_max))
    random.seed(local_seed*66)
    ba = ba*(random.uniform(rgb_opacity_min,rgb_opacity_max))
    arr = np.dstack((ra, ga, ba, aa))
    new_img = Image.fromarray(arr.astype('uint8'), 'RGBA')
    return new_img
    
#//==== NOISE  ===================================
def augment_brightness(image_for_aug, brightness_current ):
    #brightness
    image_current = image_for_aug
    enhancer = ImageEnhance.Brightness(image_current)
    image_current = enhancer.enhance(brightness_current)
    new_img_aug = image_current
    return new_img_aug

#//==== CLAHE NORMALIZATION  ===================================
def normalization_clahe_method_v4(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.convert('RGBA')
    #print(img.mode)
    #img_converted = (img.astype('uint8') * 255)
    arr = np.array(np.asarray(img).astype('uint8'))
    #arr = np.array(np.asarray(img).astype('float'))

    #im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_RGB2BGR)
    # RGB TO LAB
    #print(arr.size)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    #lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    #lab_planes[0] = clahe.apply(lab_planes[0])
    #lab = cv2.merge(lab_planes)
    lab[...,0] = clahe.apply(lab[...,0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Less 'clipLimit' value less effect
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #res = clahe.apply(img)

    #img = np.hstack([img, bgr])
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(bgr)

    return output_image


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(input_image, clip_hist_percent=10):
    input_image = input_image.convert('RGBA')
    image_array = np.array(np.asarray(input_image).astype('uint8'))
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[image_resolution],[0,image_resolution])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image_bgr, alpha=alpha, beta=beta)
    image_output = cv2.cvtColor(auto_result, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(image_output)
    return (output_image)


#//==== COMBINE SHAPES FROM images FOLDER  ===================================
def combine_shapes(base_directory, target_directory):
  seed = input_seed
  shape_set = glob.glob(os.path.join(target_directory, "*.jpg")) + glob.glob(os.path.join(target_directory, "*.png"))
  shape_set_length = len(shape_set)
  # TOAL NUMBER OF ITERATIONS ==============================
  for current_infile in range(total_number_iter):
    seed = seed+1
    random.seed(current_infile + seed)
    current_random_shape_num =  random.randrange(shape_set_length)
    infile = shape_set[current_random_shape_num]
    image_base = Image.open(infile)
    if (image_base != None): 
        print(" IMAGE_BASE MODE=== " + image_base.mode)
        #image_base_rgba = cv2.cvtColor(image_base, cv2.COLOR_RGB2RGBA)
        image_base = image_base.convert('RGBA')
        print("IMAGE_BASE = " + infile)
        layer_num_rand = round(random.randrange(layers_num_max-layers_num) + layers_num) # RANDOM LAYER NUM MIN MAX
        for current_layer in range(layer_num_rand):
            seed = seed+1
            random.seed(current_infile + seed)
            current_random_shape_num =  random.randrange(shape_set_length)
            infile = shape_set[current_random_shape_num]
            print("LAYER = " + infile)
            if (infile != None): 
                image = Image.open(infile)
                image.convert('RGBA')
                #image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
                # RANDOM RGB =======================================
                image = colorizeRGB(image,seed)
                # RANDOM MIRROR ==================================
                if ( mirror_x == 1):
                    rand_mirror = random.randrange(1)
                    if ( rand_mirror > 0.5 ):
                        image = cv2.flip(image,1)
                # RANDOM COLOR HUE SHIFT ==========================
                rand_hue = random.randrange(360)
                if colorize_on == 1:
                    image = colorize(image,rand_hue)
                # RANDOM ROATION ==================================
                rand_rot = ( ( random.randrange(rotation_range) ) *2 ) -rotation_range #
                image = image.rotate(rand_rot, PIL.Image.NEAREST, expand = 0) # NEAREST BILINEAR BICUBIC
                # RANDOM BRIGHTNESS ===============================
                random.seed(seed*11)
                rand_brightness = random.random()*brightness_range
                image = augment_brightness(image,rand_brightness)
                # RANDOM NOISE ====================================
                random.seed(seed*222)
                noise_amount_rand = random.random()
                random.seed(seed*333)
                noise_blend_amount_rand = random.random()*noise_blend_max
                image_noised = add_pepper(image,noise_amount_rand)
                image = Image.blend(image, image_noised, noise_blend_amount_rand)
                # BLEND LAYER ====================================
                #image = Image.fromarray(image)
                #image_base = Image.fromarray(image_base)
                image = image.resize((image_resolution,image_resolution))
                image_base = image_base.resize((image_resolution,image_resolution))
                image = image.convert('RGBA')
                if image_base == None :
                    image_base = PIL.Image.new(mode="RGBA", size=(image_resolution, image_resolution))
                image_base = image_base.convert('RGBA')
                print(" IMAGE_BASE MODE=== " + image_base.mode)
                opacity_rand = random.uniform(opacity_min, opacity_max) # RANDOM OPACITY
                image_base = Image.blend(image_base, image, opacity_rand)
                #Normalize
                image_normalized_pre = image_base
                image_normalized = normalization_clahe_method_v4(image_base)
                image_normalized = image_normalized.convert('RGBA')
                image_normalized_final = Image.blend(image_normalized_pre, image_normalized, normalize_clahe_amount)
                #opacity min max
                opacity_rand = random.uniform(opacity_min, opacity_max) # RANDOM OPACITY
                image_base = Image.blend(image_base, image_normalized_final, opacity_rand)
        # SHARPEN FINAL ===============================
        image_final_sharpened = image_base.filter(ImageFilter.SHARPEN);        
        # NORMALIZE FINAL ===============================
        image_final_normalize =  automatic_brightness_and_contrast(image_final_sharpened,normalize_final) 
        image_final_normalize = image_final_normalize.convert('RGBA')
        image_final_adjusted = Image.blend(image_final_sharpened, image_final_normalize, normalize_final_blend)
        image_base = image_final_adjusted
        # DARKEN FINAL ===============================
        enhancer = ImageEnhance.Brightness(image_base)
        im_output = enhancer.enhance(brightness_final)
        image_base = im_output
        # CONTRAST FINAL ===============================
        contrast_enhancer = ImageEnhance.Contrast(image_base)
        im_output = contrast_enhancer.enhance(contrast_final)
        image_base = im_output
        # SATURATION FINAL ===============================
        color_enhancer = ImageEnhance.Color(image_base)
        im_output = color_enhancer.enhance(saturation_final)
        image_base = im_output
        # SHARPEN FINAL ===============================
        image_final_adjusted = image_base.filter(ImageFilter.SHARPEN);   
        image_base = image_final_adjusted
        # SAVE IMAGE ===============================
        output_path_final = base_directory + "/" + save_output_path + "/" + str(seed) + ".png"
        print("OUTPUT FILE ====== " + output_path_final)
        image_base.save(output_path_final)

# MAIN for windows ===============================
if __name__ == '__main__':
  directory = os.path.dirname(os.path.abspath(__file__))  #// directory of current py file
  #directory_sub = str(directory) + "/images/"
  directory_sub = input_directory
  print("STARTING DIRECTORY === " + directory_sub)
  combine_shapes(directory,directory_sub)