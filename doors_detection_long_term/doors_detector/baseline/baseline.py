import cv2
import numpy as np
import math

from scripts.dataset_configurator import get_deep_doors_2_labelled_sets

class BaselineMethod:

    def get_bounding_boxes(self, image):

        [h, w, _] = image.shape

        image = cv2.resize(image, (int(round(w / 2)), int(round(h / 2))), interpolation=cv2.INTER_CUBIC)

        #cv2.imshow('', image)
        #cv2.waitKey()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('', image)
        #cv2.waitKey()

        image = cv2.GaussianBlur(image, (5, 5), sigmaX=0.9, sigmaY=0.9)
        #cv2.imshow('', image)
        #cv2.waitKey()

        canny = cv2.Canny(image, 10, 80, L2gradient=True)

        harris = cv2.cornerHarris(image, 5, 3, 0.06)

        # Suppression of false corners
        suppressed = np.array([[True for _ in range(harris.shape[1])] for _ in range(harris.shape[0])])
        corners_locations = harris > 0.01 * harris.max()
        c = 2
        for x in range(harris.shape[0]):
            for y in range(harris.shape[1]):

                if corners_locations[x, y]:
                    for i in range(-c, c + 1):
                        for j in range(-c, c + 1):
                            if not (i == 0 and j == 0) and 0 <= x + i < harris.shape[0] and 0 <= y + j < harris.shape[1]:
                                if harris[x + i, y + j] > harris[x, y]:
                                    corners_locations[x, y] = False
                # Also the contours that touches the image boundaries are considered as corners
                if (x == 0 or y == 0 or x == harris.shape[0] - 1 or y == harris.shape[1] - 1) and canny[x, y] == 255:
                    corners_locations[x, y] = True

        # Set as edges also the image contours
        min_l, max_l = -1, -1
        min_r, max_r = -1, -1
        for x in range(canny.shape[0]):
            if canny[x, 0] == 255:
                if min_l == -1:
                    min_l = x
                else:
                    max_l = x

            if canny[x, canny.shape[1] - 1] == 255:
                if min_r == -1:
                    min_r = x
                else:
                    max_r = x

        for x in range(min_l, max_l):
            canny[x, 0] = 255
        for x in range(min_r, max_r):
            canny[x, canny.shape[1] - 1] = 255

        min_t, max_t = -1, -1
        min_b, max_b = -1, -1
        for y in range(canny.shape[1]):
            if canny[0, y] == 255:
                if min_t == -1:
                    min_t = y
                else:
                    max_t = y

            if canny[canny.shape[0] - 1, y] == 255:
                if min_b == -1:
                    min_b = y
                else:
                    max_b = y

        for y in range(min_t, max_t):
            canny[0, y] = 255
        for y in range(min_b, max_b):
            canny[canny.shape[0] - 1, y] = 255
        canny_bgr = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        canny_bgr[np.all(canny_bgr == [255, 255, 255], axis=2)] = [255, 0, 0]
        #cv2.imshow('', canny_bgr)
        #cv2.waitKey()
        #cv2.imshow('', canny)
        #cv2.waitKey()
        #print(corners_locations.shape)
        canny_bgr[corners_locations] = [0, 0, 255]
        #cv2.imshow('', canny_bgr)
        #cv2.waitKey()

        # Thresholds
        height_thresh_L = 0.4
        height_thresh_H = 1
        width_thresh_L = 0.1
        width_thresh_H = 0.9
        direction_thresh_L = 18
        direction_thresh_H = 85
        parallel_thresh = 6
        ratio_thresh_L = 1.7
        ratio_thresh_H = 3

        [h, w] = corners_locations.shape
        candidates_groups = []
        corners_coordinates: list = np.transpose(corners_locations.nonzero()).tolist()

        def check_width(l_corner, r_corner):
            return l_corner[1] < r_corner[1] and l_corner != r_corner and w * width_thresh_L < r_corner[1] - l_corner[1] < w * width_thresh_H

        def check_height(up_corner, down_corner):
            return up_corner[0] < down_corner[0] and up_corner != down_corner and h * height_thresh_L < down_corner[0] - up_corner[0] < h * height_thresh_H

        def check_parallel_edge(l_corner, r_corner):
            try:
                return math.atan(abs(l_corner[0] - r_corner[0]) / abs(l_corner[1] - r_corner[1])) * (180 / math.pi) < direction_thresh_L
            except ZeroDivisionError:
                return math.atan(abs(l_corner[0] - r_corner[0]) / 0.0001) * (180 / math.pi) < direction_thresh_L

        def check_perpendicular_edge(up_corner, down_corner):
            try:
                return math.atan(abs(up_corner[0] - down_corner[0]) / abs(up_corner[1] - down_corner[1])) * (180 / math.pi) > direction_thresh_H
            except ZeroDivisionError:
                return math.atan(abs(up_corner[0] - down_corner[0]) / 0.0001) * (180 / math.pi) > direction_thresh_H

        def check_vertical_edged_perpendicular(e1, e2):
            e1_up, e1_down = e1[0], e1[1]
            e2_up, e2_down = e2[0], e2[1]

            try:
                e1_dir = math.atan(abs(e1_up[0] - e1_down[0]) / abs(e1_up[1] - e1_down[1])) * (180 / math.pi)
            except ZeroDivisionError:
                e1_dir = math.atan(abs(e1_up[0] - e1_down[0]) / 0.0001) * (180 / math.pi)

            try:
                e2_dir = math.atan(abs(e2_up[0] - e2_down[0]) / abs(e2_up[1] - e2_down[1])) * (180 / math.pi)
            except ZeroDivisionError:
                e2_dir = math.atan(abs(e2_up[0] - e2_down[0]) / 0.00001) * (180 / math.pi)

            return abs(e1_dir - e2_dir) < parallel_thresh

        def check_ratio(c1, c2, c3, c4):
            return ratio_thresh_L < ((c4[0] - c1[0]) + (c3[0] - c2[0])) / ((c2[1] - c1[1]) + (c3[1] - c4[1])) < ratio_thresh_H

        #corners_coordinates = [[1, 1], [2, 2], [10, 205], [400, 201], [401, 3], [200, 200]]
        for c1 in corners_coordinates:

            # Determine c2 candidates
            # 1) c2 must be in an interval on the right
            c2_candidates = filter(lambda c2: check_width(l_corner=c1, r_corner=c2), corners_coordinates)
            # 2) the edge c1-c2 must be almost parallel with the horizontal axis
            c2_candidates = filter(lambda c2: check_parallel_edge(l_corner=c1, r_corner=c2), c2_candidates)

            for c2 in c2_candidates:
                # Determine c3 candidates
                # 1) c3 must be in an interval below c2
                c3_candidates = filter(lambda c3: check_height(up_corner=c2, down_corner=c3), corners_coordinates)
                # 2) the edge c2-c3 must be almost perpendicular with the horizontal axis
                c3_candidates = filter(lambda c3: check_perpendicular_edge(up_corner=c2, down_corner=c3), c3_candidates)

                for c3 in c3_candidates:
                    # Determine the c3 candidates
                    # 1) c4 must be in an interval on the left of c3
                    c4_candidates = filter(lambda c4: check_width(l_corner=c4, r_corner=c3), corners_coordinates)

                    # 2) c4 must be in an interval below c1
                    c4_candidates = filter(lambda c4: check_height(up_corner=c1, down_corner=c4), c4_candidates)
                    # 3) the edge c1-c4 must be almost perpendicular with the horizontal axis
                    c4_candidates = filter(lambda c4: check_perpendicular_edge(up_corner=c1, down_corner=c4), c4_candidates)
                    # 4) the edge c3-c4 must be almost parallel with the horizontal axis
                    c4_candidates = filter(lambda c4: check_parallel_edge(l_corner=c4, r_corner=c3), c4_candidates)

                    # 5) vertical lines must be parallel
                    c4_candidates = filter(lambda c4: check_vertical_edged_perpendicular(e1=(c1, c4), e2=(c2, c3)), c4_candidates)

                    # 6) check ratio
                    c4_candidates = filter(lambda c4: check_ratio(c1, c2, c3, c4), c4_candidates)

                    [candidates_groups.append([c1[::-1], c2[::-1], c3[::-1], c4[::-1]]) for c4 in c4_candidates]

        #print(len(candidates_groups))


        # Select the doors candidates according to edges
        ratio_thresh_L = 0.5
        ration_thresh_H = 0.85
        get_distance = lambda p1, p2: math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        final_doors = []
        #candidates_groups = [[[20, 50][::-1], [23, 156][::-1], [255, 144][::-1], [259, 54][::-1]]]
        for [c1, c2, c3, c4] in candidates_groups:

            # Ratio c1-c2
            mask = np.array([[0 for _ in range(abs(c1[0] - c2[0]) + 1)] for _ in range(abs(c1[1] - c2[1]) + 1)], dtype=np.uint8)
            canny_cut = canny[c1[1]:c2[1] + 1, c1[0]:c2[0] + 1] if c1[1] < c2[1] else canny[c2[1]:c1[1] + 1, c1[0]:c2[0] + 1]
            c1_new, c2_new = ([0, 0], [mask.shape[1] - 1, mask.shape[0] - 1]) if c1[1] < c2[1] else ([mask.shape[1] - 1, 0], [0, mask.shape[0] - 1])
            cv2.line(mask, c1_new, c2_new, 255, thickness=2)

            ratio_c1_c2 = np.count_nonzero(np.logical_and(mask, canny_cut)) / get_distance(c1, c2)
            if ratio_c1_c2 <= ratio_thresh_L:
                continue

            # Ratio c2-c3
            mask = np.array([[0 for _ in range(abs(c2[0] - c3[0]) + 1)] for _ in range(abs(c2[1] - c3[1]) + 1)], dtype=np.uint8)
            canny_cut = canny[c2[1]:c3[1] + 1, c2[0]:c3[0] + 1] if c2[0] < c3[0] else canny[c2[1]:c3[1] + 1, c3[0]:c2[0] + 1]
            c2_new, c3_new = ([0, 0], [mask.shape[1] - 1, mask.shape[0] - 1]) if c2[0] < c3[0] else ([0, mask.shape[0] - 1], [mask.shape[1] - 1, 0])
            cv2.line(mask, c2_new, c3_new, 255, thickness=2)
            ratio_c2_c3 = np.count_nonzero(np.logical_and(mask, canny_cut)) / get_distance(c2, c3)
            if ratio_c2_c3 <= ratio_thresh_L:
                continue

            # Ratio c3-c4
            mask = np.array([[0 for _ in range(abs(c3[0] - c4[0]) + 1)] for _ in range(abs(c3[1] - c4[1]) + 1)], dtype=np.uint8)
            canny_cut = canny[c4[1]:c3[1] + 1, c4[0]:c3[0] + 1] if c4[1] < c3[1] else canny[c3[1]:c4[1] + 1, c4[0]:c3[0] + 1]

            c4_new, c3_new = ([0, 0], [mask.shape[1] - 1, mask.shape[0] - 1]) if c4[1] < c3[1] else ([mask.shape[1] - 1, 0], [0, mask.shape[0] - 1])
            cv2.line(mask, c3_new, c4_new, 255, thickness=2)
            ratio_c3_c4 = np.count_nonzero(np.logical_and(mask, canny_cut)) / get_distance(c3, c4)
            if ratio_c3_c4 <= ratio_thresh_L:
                continue

            # Ratio c4-c1
            mask = np.array([[0 for _ in range(abs(c4[0] - c1[0]) + 1)] for _ in range(abs(c4[1] - c1[1]) + 1)], dtype=np.uint8)
            canny_cut = canny[c1[1]:c4[1] + 1, c1[0]:c4[0] + 1] if c1[0] < c4[0] else canny[c1[1]:c4[1] + 1, c4[0]:c1[0] + 1]
            c1_new, c4_new = ([0, 0], [mask.shape[1] - 1, mask.shape[0] - 1]) if c1[0] < c4[0] else ([0, mask.shape[0] - 1], [mask.shape[1] - 1, 0])
            cv2.line(mask, c1_new, c4_new, 255, thickness=2)

            ratio_c1_c4 = np.count_nonzero(np.logical_and(mask, canny_cut)) / get_distance(c1, c4)
            if ratio_c1_c4 <= ratio_thresh_L:
                continue

            #if (ratio_c1_c2 + ratio_c2_c3 + ratio_c3_c4 + ratio_c1_c4) / 4 > 0.7:
            final_doors.append([c1, c2, c3, c4])

        #print(len(final_doors))
        #for [c1, c2, c3, c4] in final_doors:
        #    cv2.polylines(canny_bgr, np.array([[c1, c2, c3, c4]]), True, (0, 255, 0))
        #cv2.imshow('ff', canny_bgr)
        #cv2.waitKey()
        return final_doors









