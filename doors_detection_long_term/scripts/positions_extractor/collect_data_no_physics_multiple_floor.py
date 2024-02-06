import cupy
import numpy as np
from generic_dataset.dataset_folder_manager import DatasetFolderManager
from gibson.envs.no_physiscs_env import TurtlebotNavigateNoPhysicsEnv
import argparse

from doors_detection_long_term.positions_extractor.doors_dataset.door_sample import DoorSample
from doors_detection_long_term.positions_extractor.gibson_environments_data import GibsonEnvironmentsData
from doors_detection_long_term.positions_extractor.gibson_run_config import GibsonConfigRun
from doors_detection_long_term.positions_extractor.voronoi_graph_generator import VoronoiGraphGenerator

env_name = 'house1'

config_file = GibsonConfigRun(simulation_env=TurtlebotNavigateNoPhysicsEnv, world_name=env_name, floor=0) \
    .is_discrete(False).write_to_file()

if __name__ == '__main__':

    # Disable the cuda memory pool (otherwise the gpu memory will be saturated)
    cupy.cuda.set_allocator(None)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    # Create the dataset
    dataset_path = '/home/antonazzi/myfiles/doors_dataset'

    folder_manager = DatasetFolderManager(dataset_path=dataset_path, folder_name=env_name, sample_class=DoorSample, max_treads=8)
    print('Be patient, the simulator is loading the world...')
    # Start Gibson
    env = TurtlebotNavigateNoPhysicsEnv(config=args.config)
    env.reset()


    house1_data = GibsonEnvironmentsData().get_environment_data(env_name=env_name)
    floors = house1_data[GibsonEnvironmentsData.KEY_FLOORS]

    # Consider all environment's floor and collect data
    for floor in floors.keys():

        floor_height = floors[floor][GibsonEnvironmentsData.KEY_FLOOR_HEIGHT]
        voronoi_graph_generator = VoronoiGraphGenerator(env_name=env_name, floor=floor)

        # Get positions using voronoi graph
        voronoi_graph_generator.generate_voronoi_bitmap(save_to_file=True)
        graph = voronoi_graph_generator.get_voronoi_graph()
        positions = graph.get_real_position(0.20)

        for position in positions:

            (x, y) = position.to_real_point()

            heights = [floor_height + i * 0.35 for i in range(3)]
            angles = [i * (2 * np.pi / 8) for i in range(8)]

            for z, angle in [(h, a) for h in heights for a in angles]:
                env.robot.set_position([x, y, z])
                env.robot.set_orientation(x=angle)

                robot_pose = {'x': x, 'y': y, 'z': z, 'orn': angle}

                # Remember to call env.step before every robot position change
                rendered_data, _, _, _ = env.step([0.0, 0.0])

                # Get data rendered by GibsonEnv
                rgb_image = rendered_data['rgb_filled']
                depth_data = np.around(rendered_data['depth'].reshape(rendered_data['depth'].shape[:2]), decimals=4)
                semantic_image = rendered_data['semantics']

                sample = DoorSample(label=0) \
                    .set_bgr_image(value=rgb_image) \
                    .set_depth_data(value=depth_data) \
                    .set_semantic_image(value=semantic_image) \
                    .set_pretty_semantic_image(value=semantic_image.copy()) \
                    .set_robot_pose(value=robot_pose)

                # Fix the data

                # Gibson's images are rgb encoded but opencv prefer the bgr encoding
                # So the bgr_image must be converted from rgb to bgr.
                pipeline_rgb_to_bgr_image = sample.pipeline_fix_bgr_image().run(use_gpu=True)

                # Generate the depth image from depth data
                pipeline_fix_semantic_image = sample.pipeline_fix_semantic_image().run(use_gpu=True)

                # While the gpu creates the other data, the cpu can create pretty semantic image
                #sample.create_pretty_semantic_image(color=Color(red=0, green=255, blue=0))

                # Get the data of the elaborated fields
                pipeline_rgb_to_bgr_image.get_data()
                pipeline_fix_semantic_image.get_data()

                # Calculate positiveness
                sample.calculate_positiveness(threshold=2.5)

                # Save sample
                folder_manager.save_sample(sample, use_thread=True)

                #print(depth_data, depth_data.shape)

    folder_manager.save_metadata()

