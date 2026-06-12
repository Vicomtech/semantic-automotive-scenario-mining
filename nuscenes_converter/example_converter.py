# install c++ build tools if its gonna be used in Windows: https://www.youtube.com/watch?v=rcI1_e38BWs
# conda install git
# conda create --name nuscenes python=3.7
# pip install nuscenes-devkit
# conda install -n nuscenes nb_conda_kernels
# If it still fails ensure that c++ build tools are properly installed or try the following commands
# pip install cython
# pip install numpy
# pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
# jupyter notebook .
from nuscenes_converter import Converter
import augmentation
import os
import vcd.core as core
"""
The folder containing the dataset should be structured as follows:
+-- v1.0-mini
|   +-- can_bus (optional, download needed)
|   |    +-- *.json 
|   +-- maps 
|   |    +-- basemap (optional, download needed)
|   |    +-- expansion (optional, download needed)
|   |    +-- prediction (optional, download needed)
|   |    +-- *.png 
|   +-- samples 
|   +-- sweeps 
|   +-- v1.0-mini
|   |    +-- attribute.json
|   |    +-- calibrated_sensor.json
|   |    +-- category.json
|   |    +-- ego_pose.json
|   |    +-- instance.json
|   |    +-- log.json
|   |    +-- map.json
|   |    +-- sample.json
|   |    +-- sample_annotation.json
|   |    +-- sample_data.json
|   |    +-- scene.json
|   |    +-- sensor.json
|   |    +-- visibility.json
"""
# Change the paths to your local paths
output_path = "vcd_nuscenes/"
dataset_path = "D:/projects/PROVEN/Repositories/nuscenesConverter/v1.0-mini"
nuscenes_folder = "v1.0-mini"
can_expansion_path = "D:/projects/PROVEN/Repositories/nuscenesConverter/v1.0-mini/can_bus/" # can_bus must be downloaded into the "can_bus" dataset folder
map_expansion_path = "D:/projects/PROVEN/Repositories/nuscenesConverter/v1.0-mini/" # maps folder should be downloaded into the "map" dataset folder

def convert_nuscenes_mini():
    # Change the paths to your local paths
    converter = Converter(dataset_path=dataset_path, nuscenes_folder=nuscenes_folder, openlabel_path=output_path)
    converter.generate_vcds(can_expansion_path=can_expansion_path, map_expansion_path=map_expansion_path)
    for path, subdirs, files in os.walk(output_path):
        for name in files:
            vcd = core.OpenLABEL()
            vcd.load_from_file(os.path.join(path, name))
            new_vcd = augmentation.add_distances(vcd, add_following=True)
            new_vcd = augmentation.add_ego_velocity_vec(new_vcd)
            new_vcd = augmentation.add_ttc(new_vcd)
            new_vcd.save(os.path.join(path, name), pretty=True)

def convert_nuscenes_training():
    output_path = "vcd_nuscenes_full/"
    dataset_path = "."
    nuscenes_folder = "v1.0-trainval"
    can_expansion_path = "../v1.0-mini/can_bus/"  # can_bus must be downloaded into the "can_bus" dataset folder
    map_expansion_path = "./"  # maps folder should be downloaded into the "map" dataset folder

    # These Scenes do not have CAN data expansion, so we need to skip them in the augmentation
    filter = ['0d2cc345342a460e94ff54748338ac22', 'bc4fd5a05a004333b9411754630f4cba',
              '6ab3d1e9476d4cd89d4949d69d056901', '75a4ec12042542149b0a77a0a10d6330',
              '8fbbe701baf641359129ea166e1674ec', '15e1fa06e30e438a98430cc1fd0e8a69',
              '448cf480b0b8400a86881d28c2c5f734', 'bef135921f374f838bf0badae55cac83',
              'ab6eea0e06c84f70be411a9d36636a7a', '2ffd7e2a1daf4b928464ddb2ed3dca59',
              '634a8c5835e44aec912604a9a1972a5d', '8931a57994764c9b945a7a1b352c9ae5',
              'fd5a3c6d3ad44954a8045edbe9d93763', '7cf32f906f50415786414ce8bbe10e9b',
              'c7492bdc08f8450fa580b7787331f0c9']

    converter = Converter(dataset_path=dataset_path, nuscenes_folder=nuscenes_folder, openlabel_path=output_path)
    converter.generate_vcds(can_expansion_path=can_expansion_path, map_expansion_path=map_expansion_path)
    # converter.generate_vcds()
    for path, subdirs, files in os.walk(output_path):
        for name in files:
            for f in filter:
                if f in name:
                    continue
            vcd = core.OpenLABEL(os.path.join(path, name))
            new_vcd = augmentation.add_distances(vcd, add_following=True)
            new_vcd = augmentation.add_ego_velocity_vec(new_vcd)
            new_vcd = augmentation.add_ttc(new_vcd)
            new_vcd.save(os.path.join(path, name), pretty=True)

convert_nuscenes_mini()