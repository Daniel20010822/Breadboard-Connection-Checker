import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Scaling factors
H, W = 1513, 4621

# Data used in 'find_coordinate()'
x_bound = [147, 218, 289, 360, 432, 502, 572, 643, 715, 788,
           859, 929, 1000, 1072, 1143, 1214, 1285, 1356, 1427, 1497,
           1567, 1638, 1709, 1779, 1849, 1919, 1989, 2059, 2129, 2199,
           2269, 2339, 2409, 2479, 2549, 2620, 2690, 2760, 2831, 2901,
           2971, 3042, 3113, 3183, 3254, 3325, 3397, 3469, 3540, 3612,
           3684, 3755, 3827, 3899, 3970, 4041, 4113, 4186, 4258, 4330, 4402, 4473]
y_bound = [125, 263, 508, 755, 1002, 1245, 1382]



def draw_all_element_polygon(lines, image=None, show_image=False, save_image=False, **kwargs):
    """
    Draw all polygons of predicted elememts.
    input:
    - lines: list, [line1, line2, ...]
    - image: np.array, cv2.imread("path"), [H, W, rgb] ([H,W,3])
    """
    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    for line in lines:
        points = line[1:-1].split()
        points = np.array(points, dtype=np.float32) # list -> array
        x = points[::2] * W
        y = points[1::2] * H
        pts = np.column_stack((x, y)).astype(np.int32)

        cls = int(line[0]) #class
        if cls==1:
            image = cv2.polylines(image.copy(), pts=[pts], isClosed=True, color=(0,255,0), thickness=10) #Need .copy()
        elif cls==2:
            image = cv2.polylines(image.copy(), pts=[pts], isClosed=True, color=(255,0,0), thickness=10) #Need .copy()
    labeled_image = image

    if show_image:
        plt.imshow(labeled_image) # RGB
        plt.show()
    if save_image:
        image_save = labeled_image[:,:,::-1] # BGR
        if kwargs:
            cv2.imwrite(f"result_images/{kwargs['filename']}.jpg", image_save)
            print(f"Saving '{kwargs['filename']}.jpg' to /result_images ...")
        else:
            cv2.imwrite(f"result_images/polygons.jpg", image_save)
            print("Saving 'polygons.jpg' to /result_images ...")


def image_grid_check(show_image=False, save_image=False):
    """
    Shows the decision boundaries so that we can observe which line the element locates.
    inputs:
    - show_image: bool
    - save_image: bool
    """
    img = cv2.imread('result_images/get-pin-result.jpg') #BGR
    img = img[:,:,::-1] # RGB
    background = np.zeros((H, W, 3), dtype = np.uint8)
    for x in x_bound:
        background = cv2.line(background, (x, 0), (x, H), (255,255,255), thickness=5)
    # for y in y_bound:
    #     background = cv2.line(background, (0, y), (W, y), (127,127,127), thickness=5)
    output = cv2.addWeighted(img, 0.5, background, 0.3, 50)

    if show_image:
        plt.imshow(output)
        plt.show()
    if save_image:
        cv2.imwrite('result_images/image-grid.jpg', output[:,:,::-1])
        print("Saving 'image-grid.jpg' to /result_images ...")


def get_element_pin(line):
    """
    Enter ONE line in yolo_predect.txt to find PIN point
    input:
    - line: str, (class x1 y1 x2 y2 ....)
    output:
    - PIN: list of np.array, [[x1, y1], [x2, y2], [xc, yc]]
    """

    points = line[1:-1].split()
    points = np.array(points, dtype=np.float32) # list -> array
    x = points[::2] * W
    y = points[1::2] * H
    pts = np.column_stack((x, y)).astype(np.int32)


    # # Method1:
    # # When x_range is larger, then get [max(x), corresp. y], [min(x), corresp. y]
    # # Same when y_range is larger.
    # x_range = np.max(x) - np.min(x)
    # y_range = np.max(y) - np.min(y)
    # if x_range > y_range:
    #     idx_max = np.argmax(x)
    #     idx_min = np.argmin(x)
    #     edge_point1 = np.int32(pts[idx_max])
    #     edge_point2 = np.int32(pts[idx_min])
    # else:
    #     idx_max = np.argmax(y)
    #     idx_min = np.argmin(y)
    #     edge_point1 = np.int32(pts[idx_max])
    #     edge_point2 = np.int32(pts[idx_min])
    center = np.array([0.5*(np.max(x)+np.min(x)), 0.5*(np.max(y)+np.min(y))]).astype(np.int32)


    # Method2:
    # Find the farest point from center
    # Find edge_point1: max dist(edge_point1, center)
    diff = (pts - center) ** 2
    dist = np.sum(diff, axis=1)
    idx = np.argsort(dist)
    edge_point1 = (pts[idx[-1]])
    edge_point1 = np.int32(edge_point1)
    # Find edge_point2: max dist(edge_point1, edge_point2)
    diff = (pts - edge_point1) ** 2
    dist = np.sum(diff, axis=1)
    idx = np.argsort(dist)
    edge_point2 = (pts[idx[-1]])
    edge_point2 = np.int32(edge_point2)
    PIN = [edge_point1, edge_point2, center]

    return PIN


def get_all_element_pin(lines, image=None, show_image=False, save_image=False):
    """
    Enter (1) lines from yolo_predect.txt and (2) image then find location
    input:
    - lines: lines from yolo_predect.txt
    - image: np.array, cv2.imread("path"), [H, W, rgb] ([H,W,3])
    output:
    - PINs: list, [C1_PIN, C2_PIN, ...],  C1_PIN = list of np.array, [cls_type, location1, location2]
    """
    if not os.path.exists("result_images"):
        print('./result_images not found')
        print('Creating ./result_images ...')
        os.mkdir("result_images")

    PINs = []
    for line in lines:
        edge_point1, edge_point2, center = get_element_pin(line)
        edge_point1_info = find_coordinate(edge_point1)
        edge_point2_info = find_coordinate(edge_point2)

        cls = int(line[0]) # Check class
        cls_name = 'W' if cls == 1 else 'R'
        PINs.append([cls_name, edge_point1_info, edge_point2_info])
        PINs = [x for i, x in enumerate(PINs) if x not in PINs[:i]] # removing recurring elements

        if show_image or save_image:
            image = image.copy() #If no this line, cv2.circle will be wrong
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            cls_color = (0,255,0) if cls == 1 else (255,0,0)
            image = cv2.circle(image, center, radius=10, color=cls_color, thickness=10) #G
            image = cv2.circle(image, edge_point1, radius=10, color=(0,0,255), thickness=10) #B
            image = cv2.circle(image, edge_point2, radius=10, color=(0,0,255), thickness=10) #B

    if show_image:
        plt.imshow(image) #RGB
        plt.show()
    if save_image:
        image_save = image[:,:,::-1] #BGR
        cv2.imwrite(f"result_images/get-pin-result.jpg", image_save)
        print("Saving 'get-pin-result.jpg' to /result_images ...")
    return PINs


def find_coordinate(point):
    """
    Find the relative position of the component on the breadboard.
    input:
    - (x,y)
    output:
    - coord. on breadboard
    """
    x, y = point[0], point[1]

    # Find x -> which col
    col = sum([1 for bound in x_bound if x > bound]) + 1

    # Find y -> gnd, vdd, abcde, fghij, gnd, vdd
    if y < y_bound[0]:
        row = 'gnd'
    elif y < y_bound[1]:
        row = 'vdd'
    elif y < y_bound[3]:
        row = 'abcde'
    elif y < y_bound[5]:
        row = 'fghij'
    elif y < y_bound[6]:
        row = 'gnd'
    else:
        row = 'vdd'

    return [row] if row == 'vdd' or row == 'gnd' else [row, col]


def check_connection(object1: list, object2: list) -> bool:
    """
    Checks whether two elements are connected.
    input:
    - object1: pins of ONE component, list of list, ex. ['R', ['abcde', 20], ['abcde', 24]]
    - object2: pins of ONE component, list of list, ex. ['R', ['abcde', 15], ['abcde', 20]]
    output:
    - connection: bool
    """
    # Fetch only edge points
    object1, object2 = object1[1:], object2[1:]

    # Check connections
    connection = False
    for pin in object1:
        if pin in object2:
            connection = True
    return connection


def find_isolated_component(PINs):
    """
    Find out if there are any component that both edges are not connected to any other component.
    The result of this function is the subset of the one in
    input:
    - PINs
    """
    isolated_components = []
    for C1 in PINs:
        connections = []
        for C2 in [pin for pin in PINs if pin != C1]:
            connection = check_connection(C1, C2)
            connections.append(connection)
        if not any(connections): # any(connections) = False, when connections all false
            isolated_components.append(C1)
    return isolated_components


def find_open_component(PINs):
    """
    Find all elements that is open circuited. That is, all elements that has at least one edge not connected,
    excluding the fact that the element is connected to 'vdd' or 'gnd'.
    """
    opened_components = []
    opened_components_number = []
    for i, C1 in enumerate(PINs):
        # print("Current component:", C1)
        edge_point1, edge_point2 = C1[1], C1[2]
        connections = [False, False]
        for C2 in [pin for pin in PINs if pin != C1]:
            connected_to_power = check_connection([None, ['vdd']], C1) or check_connection([None, ['gnd']], C1)
            if check_connection([None, edge_point1], C2) or connected_to_power:
                connections[0] = True
            if check_connection([None, edge_point2], C2) or connected_to_power:
                connections[1] = True
        # print(connections)

        # If one edge is not connected
        if not all(connections):
            opened_components.append(C1)
            opened_components_number.append(i)

    return opened_components, opened_components_number


############################## find_short_component  ##############################
class DisjointSet:
    def __init__(self):
        self.parent = {}

    def make_set(self, item):
        self.parent[item] = item

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  # 路徑壓縮
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            for key in self.parent.keys():
                if self.parent[key] == root1:
                    self.parent[key] = root2

def update_equivalence(items, equivalences):
    disjoint_set = DisjointSet()

    for item in items:
        disjoint_set.make_set(item)

    for equivalence in equivalences:
        if len(equivalence) > 1:
            disjoint_set.union(equivalence[0], equivalence[1])

    return disjoint_set.parent
# items = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
# equivalences = [["a", "b"], ["c", "b"], ["e", "g"], ["f", "h"], ["i", "h"]]
# result = update_equivalence(items, equivalences)
############################## find_short_component  ##############################

list1 = [ str(["abcde", ii+1]) for ii in range(63)]
list2 = [ str(['fghij', ii+1]) for ii in range(63)]
AllPINList = ["['vdd', 0]", "['gnd', 0]"] + list1 + list2  #['vdd', 'gnd', ['abcde', 1], ['abcde', 2], ['abcde', 3], ['abcde', 4], ['abcde', 5], ...]

def find_short_component(PINs):
    """
    Find short components:
    (1) R1_pin1 = R1_pin2
    (2) R1_pin1 -> wire -> wire -> ... -> R2_pin2
    input:
    - PINs: [C1, C2, ...], C1 = ["W", ['abcde', 20], ['abcde', 24]]
    output:
    - short_components: [R1, R2, ...]
    """
    # let ['vdd'] -> ['vdd', 0]
    PINs_ = PINs
    PINs = []
    for pin in PINs_:  # let ['vdd'] -> ['vdd', 0]
        if pin[1][0] == 'vdd':
            pin[1] = ['vdd', 0]
        elif pin[1][0] == 'gnd':
            pin[1] = ['gnd', 0]
        elif pin[2][0] == 'vdd':
            pin[2] = ['vdd', 0]
        elif pin[2][0] == 'gnd':
            pin[2] = ['gnd', 0]
        PINs.append(pin)
    # find "wire" component
    equivalences = [ [str(pin[1]), str(pin[2])] for pin in PINs if "W" in pin ]
    # print(equivalences)
    # print(len(equivalences))
    AllPINList_updated = update_equivalence(AllPINList, equivalences)  #update list
    # Uncomment line 334 to 336 to see more.
    # for ii in range(len(equivalences)):
    #     print(f"{equivalences[ii][0]} -> {AllPINList_updated[equivalences[ii][0]]}")
    #     print(f"{equivalences[ii][1]} -> {AllPINList_updated[equivalences[ii][1]]}")
    short_components = []
    Rs = [R for R in PINs if "R" in R]
    for R in Rs:
        if AllPINList_updated[str(R[1])] == AllPINList_updated[str(R[2])]:
            short_components.append(R)
    return short_components


# TODO: create_circuit_path is still not complete
# def create_circuit_path(PINs):
#     """
#     TODO: Check explanation
#     Find the path from vdd to gnd. If the path cannot successfully connected to gnd,
#     the result will only be appended until the last element connected.
#     NOTE: Current version only works when there are no parallel paths!
#     inputs:
#     - PINs
#     output:
#     - PINs that follow the order of vdd -> gnd
#     """
#     path = [PINs[0]]
#     PINs = PINs[1:]
#     # Precussions

#     # # Find the element connected to vdd
#     # for component in PINs:
#     #     if check_connection(component, [None, ['vdd']]):
#     #         path.append(component)
#     #         PINs = [pin for pin in PINs if pin != component] # remove current pin from PINs
#     #         break
#     # # If not connected to vdd, then return

#     # if not path:
#     #     return

#     while PINs:
#         prev_len = len(path)
#         for component in PINs:
#             if check_connection(component, path[-1]):
#                 path.append(component)
#                 PINs = [pin for pin in PINs if pin != component] # remove current pin from PINs
#                 continue
#         # path not connected, then interrupt
#         if len(path) == prev_len:
#             break

#     return path