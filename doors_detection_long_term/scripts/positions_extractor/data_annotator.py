import cv2
import numpy as np
from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.utilities.color import Color
import os

from doors_detection_long_term.positions_extractor.doors_dataset.door_sample import DoorSample, DOOR_LABELS


dataset_load_path = '/home/michele/myfiles/doors_dataset'
dataset_save_path = '/home/michele/myfiles/doors_dataset_labelled'
folder_name = 'house1'
folder_manager_load = DatasetFolderManager(dataset_path=dataset_load_path, folder_name=folder_name, sample_class=DoorSample)
folder_manager_save = DatasetFolderManager(dataset_path=dataset_save_path, folder_name=folder_name, sample_class=DoorSample)

possible_labels = sorted(list(DOOR_LABELS.keys()))
colors = {0: (0, 0, 255), 1:(255, 0, 0), 2: (0, 255, 0)}
class_index = 0
img_absolute_counts = folder_manager_load.get_samples_absolute_counts(label=1)[0:]
img_indexes_iter = iter(img_absolute_counts)

sample: DoorSample = None
img_objects = []

mouse_x = 0
mouse_y = 0
point_1 = (-1, -1)
point_2 = (-1, -1)


def next_img_index():
    global img_index, sample, img_objects
    try:
        img_index = next(img_indexes_iter)
    except StopIteration:
        print('Samples ended')

    sample = folder_manager_load.load_sample_using_absolute_count(img_index, use_thread=False)
    sample.set_pretty_semantic_image(sample.get_semantic_image().copy())
    sample.create_pretty_semantic_image(color=Color(red=0, blue=0, green=255))

    img_objects = [(0, *box) for box in sample.get_bboxes_from_semantic_image(threshold=0.03)]

    # Apply criterion to filder bounding boxes:
    # The bounding boxe indicates a door that is too close it is discarted. The thoresold is 0.5m
    threshold = 0.3
    new_img_object = []
    for label, x1, y1, width, height in img_objects:
        depth_data = sample.get_depth_data()
        mask = np.zeros(list(depth_data.shape), dtype=np.uint8)
        points = np.array([[[x1, y1], [x1 + width, y1], [x1 + width, y1 + height], [x1, y1 + height]]], dtype=np.int32)
        cv2.fillPoly(mask, points, 255)
        pixels = depth_data[mask == 255]
        mean = np.mean(pixels)

        if mean >= threshold:
            new_img_object.append((label, x1, y1, width, height))
        else:
            print('DISCARTED')

    img_objects = new_img_object

    cv2.displayOverlay(WINDOW_NAME, "Showing image " + str(img_index), 1000)


def change_class_index(x):
    global class_index
    class_index = x
    cv2.displayOverlay(WINDOW_NAME, "Selected class "
                                    "" + str(class_index) + "/"
                                                            "" + str(last_class_index) + ""
                                                                                         "\n " + class_list[class_index],3000)


def draw_edges(tmp_img):
    blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
    edges = cv2.Canny(blur, 150, 250, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    tmp_img = np.bitwise_or(tmp_img, edges)
    #tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
    return tmp_img


def decrease_index(current_index, last_index):
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index


def increase_index(current_index, last_index):
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index


def draw_line(img, x, y, height, width):
    cv2.line(img, (x, 0), (x, height), (0, 255, 255))
    cv2.line(img, (0, y), (width, y), (0, 255, 255))


def yolo_format(class_index, point_1, point_2, height, width):
    # YOLO wants everything normalized
    x_center = (point_1[0] + point_2[0]) / float(2.0 * height)
    y_center = (point_1[1] + point_2[1]) / float(2.0 * width)
    x_width = float(abs(point_2[0] - point_1[0])) / height
    y_height = float(abs(point_2[1] - point_1[1])) / width
    return str(class_index) + " " + str(x_center) \
           + " " + str(y_center) + " " + str(x_width) + " " + str(y_height)


def get_txt_path(img_path):
    img_name = img_path.split('/')[-1]
    img_type = img_path.split('.')[-1]
    return bb_dir + img_name.replace(img_type, 'txt')


def save_bb(txt_path, line):
    with open(txt_path, 'a') as myfile:
        myfile.write(line + "\n") # append line


def delete_bb(txt_path, line_index):
    with open(txt_path, "r") as old_file:
        lines = old_file.readlines()

    with open(txt_path, "w") as new_file:
        counter = 0
        for line in lines:
            if counter is not line_index:
                new_file.write(line)
            counter += 1

def yolo_to_x_y(x_center, y_center, x_width, y_height, width, height):
    x_center *= width
    y_center *= height
    x_width *= width
    y_height *= height
    x_width /= 2.0
    y_height /= 2.0
    return int(x_center - x_width), int(y_center - y_height), int(x_center + x_width), int(y_center + y_height)


def draw_bboxes(tmp_img, width, height):
    for label, *obj in img_objects:
        cv2.rectangle(tmp_img, obj, colors[label], 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        x1, y1, x2, y2 = obj
        cv2.putText(tmp_img, DOOR_LABELS[label], (x1, y1 - 5), font, 0.6, colors[label], 2, cv2.LINE_AA)
    return tmp_img

# mouse callback function
def draw_roi(event, x, y, flags, param):
    global mouse_x, mouse_y, point_1, point_2
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    elif event == cv2.EVENT_LBUTTONDOWN:
        if point_1[0] is -1:
            # first click (start drawing a bounding box or delete an item)
            point_1 = (x, y)
        else:
            # second click
            point_2 = (x, y)


def is_mouse_inside_points(x1, y1, x2, y2):
    return mouse_x >= x1 and mouse_x <= x2 and mouse_y >= y1 and mouse_y <= y2

def is_mouse_inside_box(x1, y1, height, width):
    return mouse_x >= x1 and mouse_x <= x1 + height and mouse_y >= y1 and mouse_y <= y1 + width


def get_close_icon(x1, y1, height, width):
    """percentage = 0.1
    height = -1
    while height < 15 and percentage < 1.0:
        height = int((y2 - y1) * percentage)
        percentage += 0.1
    return (x2 - height), y1, x2, (y1 + height)"""
    return x1, y1, x1 + 15, y1 + 15


def draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c):
    red = (0,0,255)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), red, -1)
    white = (255, 255, 255)
    cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
    return tmp_img


def draw_info_if_bb_selected(tmp_img):
    for label, *obj in img_objects:
        x1, y1, x2, y2 = obj
        if is_mouse_inside_box(x1, y1, x2, y2):
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            tmp_img = draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
    return tmp_img


# create window
WINDOW_NAME = 'Bounding Box Labeler'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, 1000, 700)
cv2.setMouseCallback(WINDOW_NAME, draw_roi)

# selected image
TRACKBAR_IMG = 'Image'
#cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, last_img_index, change_img_index)

# selected class
TRACKBAR_CLASS = 'Class'
#cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, 0, last_class_index, change_class_index)

# initialize
edges_on = False

cv2.displayOverlay(WINDOW_NAME, "Welcome!\n Press [h] for help.", 4000)
print(" Welcome!\n Select the window and press [h] for help.")


# Initialize images
next_img_index()

# loop
while True:
    # clone the img
    tmp_img = sample.get_bgr_image().copy()
    height, width = tmp_img.shape[:2]
    if edges_on == True:
        # draw edges
        tmp_img = draw_edges(tmp_img)
    # draw vertical and horizong yellow guide lines
    draw_line(tmp_img, mouse_x, mouse_y, height, width)

    # draw already done bounding boxes
    tmp_img = draw_bboxes(tmp_img, width, height)
    # if bounding box is selected add extra info
    tmp_img = draw_info_if_bb_selected(tmp_img)
    # if first click
    if point_1[0] is not -1:
        removed_an_object = False
        # if clicked inside a delete button, then remove that object
        for label, *obj in img_objects:
            x1, y1, x2, y2 = obj
            x1, y1, x2, y2 = get_close_icon(x1, y1, x2, y2)
            if is_mouse_inside_box(x1, y1, x2 - x1, y2 - y1):
                # Remove bbox
                img_objects.remove((label, *obj))
                removed_an_object = True
                point_1 = (-1, -1)
                break

        if not removed_an_object:
            color = colors[class_index]
            # draw partial bbox
            cv2.rectangle(tmp_img, point_1, (mouse_x, mouse_y), color, 1)
            # if second click
            if point_2[0] is not -1:
                # save the bounding box
                #line = yolo_format(class_index, point_1, point_2, width, height)
                #save_bb(txt_path, line)
                # reset the points
                img_objects.append((possible_labels[class_index], point_1[0], point_1[1], point_2[0] - point_1[0] + 1, point_2[1] - point_1[1] + 1))
                point_1 = (-1, -1)
                point_2 = (-1, -1)
            else:
                cv2.displayOverlay(WINDOW_NAME, f"Selected label: {DOOR_LABELS[possible_labels[class_index]]}\nPress [w] or [s] to change.", 120)

    cv2.putText(tmp_img, f'Image {img_index - 1}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, tmp_img)
    pressed_key = cv2.waitKey(100)

    """ Key Listeners START """
    if pressed_key == ord('a') or pressed_key == ord('d'):
        # Discard image
        if pressed_key == ord('a'):
            next_img_index()
        # show next image key listener
        elif pressed_key == ord('d'):
            sample.set_bounding_boxes(value=np.array(img_objects, dtype=int))
            folder_manager_save.save_sample(sample, use_thread=True)
            next_img_index()
        #cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
    elif pressed_key == ord('s') or pressed_key == ord('w'):
        # change down current class key listener
        if pressed_key == ord('s'):
            class_index = max(0, class_index - 1)
        # change up current class key listener
        elif pressed_key == ord('w'):
            class_index = min(len(possible_labels) - 1, class_index + 1)
        #cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    # help key listener
    elif pressed_key == ord('h'):
        cv2.displayOverlay(WINDOW_NAME, "[e] to show edges;\n"
                                        "[q] to quit;\n"
                                        "[a] or [d] to change Image;\n"
                                        "[w] or [s] to change Class.", 6000)
    # show edges key listener
    elif pressed_key == ord('e'):
        if edges_on == True:
            edges_on = False
            cv2.displayOverlay(WINDOW_NAME, "Edges turned OFF!", 1000)
        else:
            cv2.displayOverlay(WINDOW_NAME, "Edges turned ON!", 1000)
            edges_on = True
    # quit key listener
    elif pressed_key == ord('q'):
        break
    """ Key Listeners END """

    # if window gets closed then quit
    if cv2.getWindowProperty(WINDOW_NAME,cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()