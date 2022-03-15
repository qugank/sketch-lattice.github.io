import numpy as np
import cv2
import random
from rdp import rdp
from sklearn.metrics.pairwise import euclidean_distances as cdist
import time

from torchvision.transforms import transforms
import torch

patch_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

"""
Tips:

ours 数据集    (x, y, p1, p2, p3) 其中x,y用的是绝对坐标
google 数据集  (deltaX, deltaY, p) 其中deltaX,deltaY用的是相对坐标
RDP算法 必须使用绝对坐标
"""


def canvas_size_google(sketch):
    """
    读取quickDraw的画布大小及起始点
    :param sketch: google sketch, quickDraw
    :param padding: 因为n*3的数据格式是相对坐标, 所以可以直接在此部分加入padding
    :return: int list,[x, y, h, w]
    """
    # get canvas size

    vertical_sum = np.cumsum(sketch, axis=0)  # 累加
    xmin_ymin = np.min(vertical_sum, axis=0)
    xmax_ymax = np.max(vertical_sum, axis=0)
    w = xmax_ymax[0] - xmin_ymin[0]
    h = xmax_ymax[1] - xmin_ymin[1]
    start_x = -xmin_ymin[0]
    start_y = -xmin_ymin[1]
    return [int(start_x), int(start_y), int(h), int(w)], sketch[:]


def canvas_size_ours(sketch):
    """
    获得canvas大小
    :param sketch:
    :return: return w, h, sketch[:]
    """
    if sketch.dtype != np.int:
        sketch = sketch.round().astype("int16")
    w, h = np.max(sketch, axis=0)[:2]
    return w, h


def draw_three(sketch, thickness=2, ):
    """
    此处主要包含画图部分，从canvas_size_google()获得画布的大小和起始点的位置，根据strokes来画
    :param sketches: google quickDraw, (n, 3)
    :param window_name: pass
    :param thickness: pass
    :return: None
    """

    [start_x, start_y, h, w], sketch = canvas_size_google(sketch=sketch)
    canvas = np.ones((h, w, 3), dtype='uint8') * 255
    color = (0, 0, 0)
    pen_now = np.array([start_x, start_y])
    first_zero = False
    for stroke in sketch:
        delta_x_y: np.ndarray = stroke[0:0 + 2]
        delta_x_y = delta_x_y.astype(np.int)
        state = stroke[2:]
        if first_zero:
            pen_now += delta_x_y
            first_zero = False
            continue
        cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
        if int(state) != 0:  # next stroke
            first_zero = True
            color = (0, 0, 0)
        pen_now += delta_x_y
    return canvas


def draw_five(sketch, windowname="ours", time=1, thickness=2, drawing=False, show=True):
    """
    根据ours的数据格式n*5，来画图
    注意, 此函数应当只负责画图, 其他的处理工作交由其他函数完成.
    :param sketch: ours format, n*5
    :param windowname:  窗口名字
    :param thickness:  线条粗细
    :return:  None
    """
    if sketch.dtype != np.int:
        sketch = np.around(sketch).astype(np.int)
    w, h = canvas_size_ours(sketch)  # 获得长宽
    canvas = np.ones((h, w, 3), dtype="uint8") * 255

    color = (random.randint(0, 255),
             random.randint(0, 255),
             random.randint(0, 255))
    for index, strokes in enumerate(sketch):  # Drawing.
        if int(sketch[index][3]) == 1:
            color = (random.randint(0, 255),
                     random.randint(0, 255),
                     random.randint(0, 255))
            continue
        if index == len(sketch) - 1:
            break
        pre_point = tuple(sketch[index][0:0 + 2])
        nxt_point = tuple(sketch[index + 1][0:0 + 2])
        cv2.line(canvas, pre_point, nxt_point, color, thickness=thickness)
        if drawing:
            cv2.imshow(windowname, canvas)
            key = cv2.waitKeyEx(time)
            if key == 27:  # esc
                cv2.destroyAllWindows()
                exit(0)
        else:
            cv2.imshow(windowname, canvas)
    if show:
        key = cv2.waitKeyEx()
        if key == 27:  # esc
            cv2.destroyAllWindows()
            exit(0)
    return canvas


def trans_google2ours(google_sketch):
    """
    将 google n*3 数据格式转化成 ours 数据格式 n*5
    :param google_sketch:
    :return:
    """
    [start_x, start_y, _, _], google_sketch = canvas_size_google(google_sketch)
    pen_now_x = start_x
    pen_now_y = start_y
    result = np.zeros((len(google_sketch) + 1, 5), dtype="int16")
    result[0] = np.array([start_x, start_y, 1, 0, 0], dtype="int16")
    for index, stroke in enumerate(google_sketch):
        pen_now_x += stroke[0]
        pen_now_y += stroke[1]
        if stroke[2] == 0:
            result[index + 1] = np.array([pen_now_x, pen_now_y, 1, 0, 0], dtype="int16")
        else:
            result[index + 1] = np.array([pen_now_x, pen_now_y, 0, 1, 0], dtype="int16")

    result[-1][2:] = np.array([0, 0, 1], dtype="int16")
    return result


def trans_ours2google(our_sketch):
    """
    将ours n*5 数据格式转化成 google原生数据格式 n*3
    :param our_sketch:
    :return:
    """
    result = np.zeros((len(our_sketch) - 1, 3), dtype="int16")
    for count in range(len(our_sketch)):
        if count == len(our_sketch) - 1:  # last one.
            break
        result[count][:2] = our_sketch[count + 1][:2] - our_sketch[count][:2]
        if int(our_sketch[count + 1][2]) == 1:
            result[count][2] = 0
        else:
            result[count][2] = 1
    return result


def five2png_save(savepathname, sketch, resize=(48, 48)):
    """
    ours的数据结构(n, 5)    保存成三通道png图
    注：含有参数 thickness 可以根据连续度的要求来加粗笔画，但是有可能会混淆笔画间
    :param savepathname:
    :param sketch:
    :param resize:
    :return:
    """
    if sketch.dtype != np.int:
        sketch = np.around(sketch).astype(np.int)

    w, h = canvas_size_ours(sketch)
    canvas = np.ones((h, w, 3), dtype="uint8") * 255
    color = (0, 0, 0)
    for count in range(len(sketch)):
        if int(sketch[count][3]) == 1:
            continue
        if int(sketch[count][4]) == 1:
            break
        pre_point = sketch[count][0:2]
        nxt_point = sketch[count + 1][0:2]
        cv2.line(canvas, tuple(pre_point), tuple(nxt_point), color, thickness=5)
    # 3 channels
    canvas = cv2.resize(canvas, resize, cv2.INTER_LINEAR)
    cv2.imwrite(f"{savepathname}", canvas)
    print(f"{savepathname} success")
    return None


def five_rdp_with_discard(sketch, epsilon=2, discard=4):
    """
    使用rdp算法对矢量进行压缩，一般使用epsilon=2
    :param sketch:  n*5, our format
    :param epsilon:  threshold  # rdp算法的距离阈值
    :param discard:  丢弃的笔画数(包含)  # 低于x折线笔画的stroke 省略.
    :return:
    """
    flag = 0
    result = np.array([[0, 0, 0, 0, 0]], dtype="int16")
    for idx, stroke in enumerate(sketch):
        if int(stroke[2]) == 0:
            temblock = sketch[flag: idx + 1]
            points = rdp(temblock[:, 0:2], epsilon=epsilon)
            temblock[-points.shape[0]:, 0:2] = points
            resultblock = temblock[-points.shape[0]:]
            if resultblock.shape[0] <= discard:
                flag = idx + 1
                continue
            result = np.concatenate((result, resultblock), axis=0)
            flag = idx + 1
    result[-1:, 2:] = np.array([[0, 0, 1]])
    return result[1:]


def remove_white_space_image(img_np: np.ndarray, padding: int):
    """
    获取白底图片中, 物体的bbox; 此处白底必须是纯白色.
    其中, 白底有两种表示方法, 分别是 1.0 以及 255; 在开始时进行检查并且匹配
    对最大值为255的图片进行操作.
    三通道的图无法直接使用255进行操作, 为了减小计算, 直接将三通道相加, 值为255*3的pix 认为是白底.
    :param img_np:
    :return:
    """
    if np.max(img_np) <= 1.0:  # 1.0 <= 1.0 True
        img_np = (img_np * 255).astype("uint8")
    else:
        img_np = img_np.astype("uint8")

    h, w, c = img_np.shape
    img_np_single = np.sum(img_np, axis=2)
    Y, X = np.where(img_np_single <= 760)  # max = 765
    ymin, ymax, xmin, xmax = np.min(Y), np.max(Y), np.min(X), np.max(X)
    img_cropped = img_np[max(0, ymin - padding):min(h, ymax + padding), max(0, xmin - padding):min(w, xmax + padding),
                  :]
    return img_cropped


def resize_image_by_ratio(img_np: np.ndarray, size: int):
    """
    按照比例resize
    :param img_np:
    :param size:
    :return:
    """
    if len(img_np.shape) == 2:
        h, w = img_np.shape
    elif len(img_np.shape) == 3:
        h, w, _ = img_np.shape
    else:
        assert 0

    ratio = h / w
    if h > w:
        new_img = cv2.resize(img_np, (int(size / ratio), size,))  # resize is w, h  (fx, fy...)
    else:
        new_img = cv2.resize(img_np, (size, int(size * ratio),))
    return new_img


def make_img_square(img_np: np.ndarray):
    if len(img_np.shape) == 2:
        h, w = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1)) * np.max(img_np)
            white2 = np.ones((h, delta2)) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w)) * np.max(img_np)
            white2 = np.ones((delta2, w)) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img
    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1, c)) * np.max(img_np)
            white2 = np.ones((h, delta2, c)) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w, c)) * np.max(img_np)
            white2 = np.ones((delta2, w, c)) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img


def remove_white_space_sketch(sketch):
    """
    删除留白
    :param sketch:
    :return:
    """
    min_list = np.min(sketch, axis=0)
    sketch[:, :2] = sketch[:, :2] - np.array(min_list[:2])
    return sketch


def five_sort(sketch):
    """
    按照笔画从长到短进行重排序，模仿人类先画大轮廓，但是有可能group之间有穿插
    :param sketch:  n*5, our format
    :return:
    """
    sketch = np.copy(sketch)
    sketch[-1, 2:5] = np.array([0, 1, 0])
    block1 = []
    tmp_index1 = 0
    for index, stroke in enumerate(sketch):
        if stroke[2] == 0:
            block1.append(sketch[tmp_index1:index + 1])
            tmp_index1 = index + 1
    sketch_google = trans_ours2google(sketch)
    block2 = []
    block2_length = []  # length for each mini block
    tmp_index2 = 0
    for index, stroke in enumerate(sketch_google):
        if stroke[2] == 1:
            block2.append(sketch_google[tmp_index2: index + 1])
            length = 0
            for s in sketch_google[tmp_index2: index + 1]:
                length += np.sqrt(s[0] ** 2 + s[1] ** 2)
            block2_length.append(length)
            tmp_index2 = index + 1
    block2_length_sort_index = sorted(enumerate(block2_length), key=lambda x: x[1])[::-1]
    final_sketch_list = []
    for i in block2_length_sort_index:
        block_index = i[0]
        final_sketch_list.append(block1[block_index])
    final_sketch = np.concatenate(final_sketch_list, axis=0)
    return final_sketch


def padding_image(img_np: np.ndarray, padding: int):
    h, w, c = img_np.shape

    vertical = np.ones((padding, w, c), dtype=img_np.dtype) * img_np.max()
    horizon = np.ones((h + 2 * padding, padding, c), dtype=img_np.dtype) * img_np.max()
    img_np = np.vstack([vertical, img_np, vertical])
    img_np = np.hstack([horizon, img_np, horizon])
    return img_np


def get_node_coordinates_graph(img_np: np.ndarray,
                               row_count: int, col_count: int,
                               maxPointFilled: int,
                               mask_prob: float,
                               max_pixel_value: int, dThreshold: float = 0.2) -> [np.ndarray,
                                                                                  np.ndarray]:
    """
    :param img_np:
    :param row_count:
    :param col_count:
    :return: list[[x, y], [c, r]]
    """
    h, w, _ = img_np.shape
    assert h <= 256 and w <= 256
    coordinates = []

    row_interval = h // row_count
    col_interval = w // col_count

    for i in range(row_count):
        row_index = row_interval // 2 + i * row_interval
        firstBlack = False
        for c_index, c_value in enumerate(img_np[row_index, :, 0]):
            if c_value != 0:  # not black
                firstBlack = False
                continue
            if not firstBlack:
                firstBlack = True
                coordinates.append([c_index, row_index, ])

    for i in range(col_count):
        col_index = col_interval // 2 + i * col_interval
        firstBlack = False
        for r_index, r_value in enumerate(img_np[:, col_index, 0]):
            if r_value != 0:  # not black
                firstBlack = False
                continue
            if not firstBlack:
                firstBlack = True
                coordinates.append([col_index, r_index])

    random.shuffle(coordinates)

    coorLen = len(coordinates)
    if coorLen == 0:
        coordinates.append([1, 1])
        coordinates.append([1, 1])
        coorLen = 2
    tmp_result = np.array(coordinates, dtype=np.float)

    AdjMatrix = np.zeros((maxPointFilled, maxPointFilled), dtype=np.float)

    eucDistance = cdist(tmp_result, tmp_result)
    reverseEucDistance: np.ndarray = 1 - eucDistance / (255 * np.sqrt(2))
    reverseEucDistance -= np.eye(coorLen)  # eye with be added in the graph normalization.

    """以距离为权重"""
    reverseEucDistance[np.where(reverseEucDistance < (1 - dThreshold))] = 0
    finalDistance = reverseEucDistance

    if coorLen > maxPointFilled:
        AdjMatrix[:coorLen, :coorLen] = finalDistance[:maxPointFilled, :maxPointFilled]
    else:
        AdjMatrix[:coorLen, :coorLen] = finalDistance[:, :]

    """此处的result 及 AdjMatrix"""

    if coorLen < maxPointFilled:  # 满足maxPointFilled训练要求
        fill = np.zeros((maxPointFilled - coorLen, 2), dtype=np.float)
        return np.concatenate([tmp_result, fill]), AdjMatrix
    else:
        return tmp_result[:maxPointFilled], AdjMatrix


def get_node_coordinates_graph_patch(img_np: np.ndarray,
                                     row_count: int, col_count: int,
                                     maxPointFilled: int,
                                     mask_prob: float,
                                     max_pixel_value: int) -> [np.ndarray,
                                                               np.ndarray]:
    """
    :param img_np:
    :param row_count:
    :param col_count:
    :return: list[[x, y], [c, r]]
    """
    h, w, _ = img_np.shape
    assert h <= 256 and w <= 256
    patches = []
    result_patches = torch.zeros((maxPointFilled, 1, 128, 128))
    patch_size = 256 // row_count
    coordinates = []

    row_interval = h // row_count
    col_interval = w // col_count

    for i in range(row_count):
        row_index = row_interval // 2 + i * row_interval
        firstBlack = False
        for c_index, c_value in enumerate(img_np[row_index, :, 0]):
            if c_value != 0:  # not black
                firstBlack = False
                continue
            if not firstBlack:
                firstBlack = True
                coordinates.append([c_index, row_index, ])

    for i in range(col_count):
        col_index = col_interval // 2 + i * col_interval
        firstBlack = False
        for r_index, r_value in enumerate(img_np[:, col_index, 0]):
            if r_value != 0:  # not black
                firstBlack = False
                continue
            if not firstBlack:
                firstBlack = True
                coordinates.append([col_index, r_index])

    tmp_coordinates = []
    for each in coordinates:
        _x, _y = each
        if _x < patch_size // 2 or _x > w - patch_size // 2 - 1 \
                or _y < patch_size // 2 or _y > h - patch_size // 2 - 1:
            continue
        tmp_coordinates.append(each)
    """大于MaxNode的点, 随机舍弃, 而不会 丢一整行或某几行的数据."""
    coordinates = tmp_coordinates
    random.shuffle(coordinates)

    """生成 (S, 2)坐标矩阵, 此处rescale到[0, 255]"""
    coorLen = len(coordinates)
    if coorLen == 0:
        coordinates.append([1, 1])
        coordinates.append([1, 1])
        coorLen = 2
    tmp_result = np.array(coordinates, dtype=np.float)

    """根据 tmp result 生成 Adjacent Matrix, 此处不能晚于 filling, 因为有0,0 的干扰"""
    AdjMatrix = np.zeros((maxPointFilled, maxPointFilled), dtype=np.float)

    eucDistance = cdist(tmp_result, tmp_result)
    reverseEucDistance: np.ndarray = 1 - eucDistance / (255 * np.sqrt(2))
    reverseEucDistance -= np.eye(coorLen)  # eye with be added in the graph normalization.

    """以距离为权重"""
    reverseEucDistance[np.where(reverseEucDistance < 0.8)] = 0
    finalDistance = reverseEucDistance

    if coorLen > maxPointFilled:
        AdjMatrix[:coorLen, :coorLen] = finalDistance[:maxPointFilled, :maxPointFilled]
    else:
        AdjMatrix[:coorLen, :coorLen] = finalDistance[:, :]

    """此处的result 及 AdjMatrix"""
    for i, coor in enumerate(coordinates):
        if i == maxPointFilled:
            break
        _tmp_patch_x, _tmp_patch_y = coor
        _tmp_patch = img_np[
                     _tmp_patch_y - patch_size // 2 + 1:_tmp_patch_y + patch_size // 2 + 1,
                     _tmp_patch_x - patch_size // 2 + 1:_tmp_patch_x + patch_size // 2 + 1, 0]
        result_patches[i] = patch_trans(cv2.resize(_tmp_patch, (128, 128)))

    if coorLen < maxPointFilled:  # 满足maxPointFilled训练要求
        fill = np.zeros((maxPointFilled - coorLen, 2), dtype=np.float)
        return result_patches.numpy(), AdjMatrix
    else:
        return result_patches.numpy(), AdjMatrix


def get_node_coordinates_graph_label(img_np: np.ndarray,
                                     row_count: int, col_count: int,
                                     maxPointFilled: int,
                                     mask_prob: float,
                                     max_pixel_value: int) -> [np.ndarray,
                                                               np.ndarray]:
    """
    :param img_np:
    :param row_count:
    :param col_count:
    :return: list[[x, y], [c, r]]
    """
    h, w, _ = img_np.shape
    nodes = []

    row_interval = h // row_count
    col_interval = w // col_count

    for i in range(row_count):
        row_index = row_interval // 2 + i * row_interval
        firstBlack = False
        for c_index, c_value in enumerate(img_np[row_index, :, 0]):
            if c_value > 50:  # not black, not label, is background
                firstBlack = False
                continue
            if not firstBlack:
                firstBlack = True
                nodes.append([c_index, row_index, c_value])

    for i in range(col_count):
        col_index = col_interval // 2 + i * col_interval
        firstBlack = False
        for r_index, r_value in enumerate(img_np[:, col_index, 0]):
            if r_value > 50:  # not black, not label, is background
                firstBlack = False
                continue
            if not firstBlack:
                firstBlack = True
                nodes.append([col_index, r_index, r_value])

    """大于MaxNode的点, 随机舍弃, 而不会 丢一整行或某几行的数据."""

    random.shuffle(nodes)
    coordinates = [[x[0], x[1]] for x in nodes]
    labels = [[x[2]] for x in nodes]

    labels = np.stack(labels)

    """生成 (S, 2)坐标矩阵, 此处rescale到[0, 255]"""
    coorLen = len(coordinates)
    tmp_result = np.array(coordinates, dtype=np.float)
    tmp_result -= tmp_result.min()
    tmp_result /= tmp_result.max()
    tmp_result *= max_pixel_value  # out of range if not -1


    """根据 tmp result 生成 Adjacent Matrix, 此处不能晚于 filling, 因为有0,0 的干扰"""
    AdjMatrix = np.zeros((maxPointFilled, maxPointFilled), dtype=np.float)

    eucDistance = cdist(tmp_result, tmp_result)
    reverseEucDistance: np.ndarray = 1 - eucDistance / eucDistance.max()
    reverseEucDistance -= np.eye(coorLen)  # eye with be added in the graph normalization.

    reverseEucDistance[np.where(reverseEucDistance < 0.6)] = 0

    if coorLen > maxPointFilled:
        AdjMatrix[:coorLen, :coorLen] = reverseEucDistance[:maxPointFilled, :maxPointFilled]
    else:
        AdjMatrix[:coorLen, :coorLen] = reverseEucDistance[:, :]

    if mask_prob != 0:
        for _ in range(int(coorLen * mask_prob)):
            if random.randint(0, 1):
                index = random.randint(0, min(coorLen - 1, maxPointFilled - 1))
                AdjMatrix[index, :] = 0
                AdjMatrix[:, index] = 0

    """此处的result 及 AdjMatrix"""
    if coorLen < maxPointFilled:
        fill = np.zeros((maxPointFilled - coorLen, 2), dtype=np.float)
        fill_labels = np.zeros((maxPointFilled - coorLen, 1), dtype="uint8")
        return np.concatenate([tmp_result, fill]), AdjMatrix, np.concatenate([labels, fill_labels])
    else:
        return tmp_result[:maxPointFilled], AdjMatrix, labels[:maxPointFilled]
