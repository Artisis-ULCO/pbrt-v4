import os
import argparse
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np

def extract_data(image: np.ndarray, tracked_values: dict):
    
    image = np.array(image)
    
    if not 'sum' in tracked_values:
        
        tracked_values['sum'] = image
        tracked_values['squared_sum'] = np.power(image, 2)
        tracked_values['cubic_sum'] = np.power(image, 3)
        tracked_values['fourth_sum'] = np.power(image, 4)
    
    else:
        tracked_values['sum'] = np.sum([tracked_values['sum'], image], axis=0)
        tracked_values['squared_sum'] = np.sum([tracked_values['squared_sum'], np.power(image, 2)], axis=0)
        tracked_values['cubic_sum'] = np.sum([tracked_values['cubic_sum'], np.power(image, 3)], axis=0)
        tracked_values['fourth_sum'] = np.sum([tracked_values['fourth_sum'], np.power(image, 4)], axis=0)

def main():
    
    parser = argparse.ArgumentParser(description="Compute statistics for each scenes")
    parser.add_argument('--folder', type=str, help="data folder with scenes files", required=True)
    parser.add_argument('--spp', type=int, help="expected number of samples", required=True)
    parser.add_argument('--output', type=str, help="folder where to extract will be saved or are saved", required=True)
    parser.add_argument('--extract', type=bool, help="extract data again or not", required=False, default=True)
    parser.add_argument('--statistics', type=str, help="output statistics folder", required=False, default=None)
    
    args = parser.parse_args()
    
    p_folder = args.folder
    p_output = args.output
    p_spp = args.spp
    p_extract = args.extract
    p_statistics = args.statistics
    
    scenes = sorted(os.listdir(p_folder))
    n_scenes = len(scenes)
    
    for id_scene, scene in enumerate(scenes):
        
        tracked_values = {}
        
        scene_path = os.path.join(p_folder, scene)
        
        scene_images = sorted(os.listdir(scene_path))
        
        # limit number of images to take into account
        scene_images = scene_images[:p_spp]
        n_scene_images = len(scene_images)
        
        # if necessary to extract data
        if p_extract:
        
            for idx, image_name in enumerate(scene_images):
                
                image_path = os.path.join(scene_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                
                extract_data(image, tracked_values)
                
                print(f'[Scene n°{id_scene + 1} of {n_scenes}] -- {scene} -- extract progress: {(idx + 1) / n_scene_images * 100.:.2f}%', \
                end='\r' if idx + 1 < n_scene_images else '\n')
                
            output_scene_path = os.path.join(p_output, scene)
            os.makedirs(output_scene_path, exist_ok=True)
            
            for key, image in tracked_values.items():
                
                output_image_path = os.path.join(output_scene_path, f'{key}.exr')
                cv2.imwrite(output_image_path, image)
                
        # now extract statistics if asked
        if p_statistics is not None:
            
            extract_scene_path = os.path.join(p_output, scene)
            
            sum_image_path = os.path.join(extract_scene_path, 'sum.exr')
            squared_sum_image_path = os.path.join(extract_scene_path, 'squared_sum.exr')
            cubic_sum_image_path = os.path.join(extract_scene_path, 'cubic_sum.exr')
            fourth_sum_image_path = os.path.join(extract_scene_path, 'fourth_sum.exr')
            
            sum_image = cv2.imread(sum_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            squared_sum = cv2.imread(squared_sum_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            cubic_sum = cv2.imread(cubic_sum_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            fourth_sum = cv2.imread(fourth_sum_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            
            # compute all moments
            # => https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/04%3A_Expected_Value/4.04%3A_Skewness_and_Kurtosis
            
            print(f'[Scene n°{id_scene + 1} of {n_scenes}] -- {scene} -- extract stats')
            
            n = p_spp
            # moment 1: mean
            mean_image = sum_image / n

            # moment 2: variance \sigma²
            onlineM2 = squared_sum / n - np.power(mean_image, 2)
            onlineM3 = (cubic_sum - 3 * mean_image * squared_sum) / n + 2 * np.power(mean_image, 3)
            onlineM4 = (fourth_sum - 4 * mean_image * cubic_sum) / n \
                + 6 * np.power(mean_image, 2) * onlineM2 \
                + 3 * np.power(mean_image, 4)
            
            onlineSkew = (onlineM3) / np.power(onlineM2, 1.5)
            onlineKurt = (onlineM4) / np.power(onlineM2, 2)
            
            moments = {
                'mean': mean_image,
                'variance': onlineM2,
                'skewness': onlineSkew,
                'kurtosis': onlineKurt
            }
            
            # save all moments
            statistics_scene_path = os.path.join(p_statistics, scene)
            os.makedirs(statistics_scene_path, exist_ok=True)
            
            for key, image in moments.items():
                
                statistics_image_path = os.path.join(statistics_scene_path, f'{key}.exr')
                cv2.imwrite(statistics_image_path, image)
                
if __name__ == "__main__":
    main()