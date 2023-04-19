import os
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from functools import partial
import numpy as np

from scipy.stats.mstats import winsorize

import matplotlib.pyplot as plt

def winsorize_pixel(a, limits, moment_f):
    winsor_a = winsorize(a, limits=limits)
    
    return np.abs(moment_f(a) - moment_f(winsor_a))

def get_moment(moment):
    
    if moment == 'mean':
        return np.mean
    
    if moment == 'std':
        return np.std
    
    return None
    
def main():
    
    parser = argparse.ArgumentParser(description="Compute statistics for each scenes")
    parser.add_argument('--folder', type=str, help="data folder with scenes files", required=True)
    parser.add_argument('--spp', type=int, help="expected number of samples", required=True)
    parser.add_argument('--output', type=str, help="folder where to extract will be saved or are saved", required=True)
    parser.add_argument('--limits', type=str, help="winsor right interval size (example: 0, 0.05)", required=False, default='0, 0.05')
    parser.add_argument('--moment', type=str, help="moment to compute and compare with winsor approach", required=False, choices=['mean', 'std'], default='mean')
    
    args = parser.parse_args()
    
    scenes_folder   = args.folder
    output_folder   = args.output
    spp             = args.spp
    limits          = list(map(float, args.limits.split(',')))
    moment          = args.moment
    
    scenes = sorted(os.listdir(scenes_folder))
    n_scenes = len(scenes)
    
    # get the expected moment function
    moment_f = get_moment(moment)
    
    winsorize_pixel_array = partial(winsorize_pixel, limits=limits, moment_f=moment_f)
    
    for id_scene, scene in enumerate(scenes):
        
        scene_path = os.path.join(scenes_folder, scene)
        
        scene_images = sorted(os.listdir(scene_path))
        
        # limit number of images to take into account
        scene_images = scene_images[:spp]
        n_scene_images = len(scene_images)
        
        # if necessary to extract data
        stack_images = []
        for idx, image_name in enumerate(scene_images):
            
            image_path = os.path.join(scene_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            stack_images.append(image)
            
            print(f'[Scene n°{id_scene + 1} of {n_scenes}] -- {scene} -- extract progress: {(idx + 1) / n_scene_images * 100.:.2f}%', \
            end='\r' if idx + 1 < n_scene_images else '\n')
            
        # compute firefly detection using winsorization
        stack_images = np.array(stack_images)
        
        h_size, w_size, _ = stack_images.shape[1:]
        final_array = np.empty(stack_images.shape[1:], dtype=np.float32)
        
        # apply winsor per pixel (in order to keep progress track)
        n_pixels = h_size * w_size
        n_passed = 0
        for i in range(h_size):
            for j in range(w_size):
            
                pixel = stack_images[:, i, j, :]
                final_array[i, j, :] = np.apply_along_axis(winsorize_pixel_array, 0, pixel)
            
                print(f' -- [{scene}] -- extract firefly map progress: {(n_passed + 1) / n_pixels * 100.:.2f}%', \
                    end='\r' if n_passed + 1 < n_pixels else '\n')
                n_passed += 1
        
        # plt.imshow(final_array)
        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(os.path.join(output_folder, f'{scene}_{moment}_{spp}.exr'), final_array)
            
        
 
      
if __name__ == "__main__":
    main()