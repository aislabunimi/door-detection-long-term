import cv2
import numpy as np

from doors_detection_long_term.positions_extractor.voronoi_graph_generator import VoronoiGraphGenerator

# Load house1 map
voronoi_graph_generator = VoronoiGraphGenerator(env_name='house10', floor=1)
house1_map = voronoi_graph_generator.get_map()
cv2.imshow('map', house1_map)
cv2.waitKey()

# Generate voronoi bitmap and the relative graph starting from the map of house 1
voronoi_bitmap = voronoi_graph_generator.generate_voronoi_bitmap(save_to_file=False)
cv2.imshow('voronoi bitmap', voronoi_bitmap)
cv2.waitKey()

# Get positions every 10cm and print them in the map
graph = voronoi_graph_generator.get_voronoi_graph()
positions = graph.get_real_position(0.40)
house1_map = cv2.cvtColor(house1_map,cv2.COLOR_GRAY2RGB)
for position in positions:
    house1_map[position.to_img_index()] = (0, 0, 200)
    #house1_map = cv2.circle(house1_map, position.to_img_index()[::-1], radius=3, color=(0, 0, 200), thickness=2)
cv2.imshow('map with selected positions', house1_map)
cv2.waitKey()


