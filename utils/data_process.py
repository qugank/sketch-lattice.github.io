import numpy as np
import cv2
import random
from rdp import rdp
from interval import Interval, IntervalSet
# from scipy.spatial.distance import cdist
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
    # 返回可能处理过的sketch
    # print(int(start_x), int(start_y), int(h), int(w))
    return [int(start_x), int(start_y), int(h), int(w)], sketch[:]


def canvas_size_ours(sketch):
    """
    获得canvas大小
    :param sketch:
    :return: return w, h, sketch[:]
    """
    if sketch.dtype != np.int:
        # print("type of data changing.")
        sketch = sketch.round().astype("int16")  # 有可能有小数.
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
    # print("three ")
    # print(sketch)
    # print("-" * 70)

    [start_x, start_y, h, w], sketch = canvas_size_google(sketch=sketch)
    canvas = np.ones((h, w, 3), dtype='uint8') * 255
    color = (0, 0, 0)
    pen_now = np.array([start_x, start_y])
    first_zero = False
    for stroke in sketch:
        delta_x_y: np.ndarray = stroke[0:0 + 2]
        delta_x_y = delta_x_y.astype(np.int)
        state = stroke[2:]
        if first_zero:  # 首个零是偏移量, 不画
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
    # print("-" * 70)
    # print("five ")
    # print(sketch)
    # print("-" * 70)
    if sketch.dtype != np.int:
        sketch = np.around(sketch).astype(np.int)
    w, h = canvas_size_ours(sketch)  # 获得长宽
    canvas = np.ones((h, w, 3), dtype="uint8") * 255

    color = (random.randint(0, 255),
             random.randint(0, 255),
             random.randint(0, 255))
    for index, strokes in enumerate(sketch):  # Drawing.
        if int(sketch[index][3]) == 1:  # 表示笔触结束，跳转笔触不用画 [0,1,0]
            color = (random.randint(0, 255),
                     random.randint(0, 255),
                     random.randint(0, 255))
            continue
        # 最后一个元素的两种判断方法, 当前笔触是的状态码是结束的, 或者当前笔触是sketch的最后一笔.
        if index == len(sketch) - 1:  # 已经是最后一个count了，立即退出循环 [0,0,1]
            break  # 在 index + 1 之前退出
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
    # cv2.imwrite(f"./{windowname}.png", canvas)
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
        result[count][:2] = our_sketch[count + 1][:2] - our_sketch[count][:2]  # 写成偏移量的形式
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
        if int(sketch[count][3]) == 1:  # 表示笔触结束，跳转笔触不用画
            continue
        if int(sketch[count][4]) == 1:  # 已经是最后一个count了，立即退出循环
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
            # print(temblock, "\n")
            flag = idx + 1
    result[-1:, 2:] = np.array([[0, 0, 1]])  # 最后将状态给置换了就行
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
    Y, X = np.where(img_np_single <= 760)  # max = 765, 留有一些余地
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
    # print(sketch)
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

    """大于MaxNode的点, 随机舍弃, 而不会 丢一整行或某几行的数据."""
    random.shuffle(coordinates)

    """生成 (S, 2)坐标矩阵, 此处rescale到[0, 255]"""
    coorLen = len(coordinates)
    # print(coorLen)
    if coorLen == 0:
        coordinates.append([1, 1])
        coordinates.append([1, 1])
        coorLen = 2
    tmp_result = np.array(coordinates, dtype=np.float)
    # tmp_result -= tmp_result.min()  # ? 此处是否要归一到0
    # tmp_result /= tmp_result.max()
    # tmp_result *= max_pixel_value  # out of range if not -1

    #     # tmp_result = tmp_result / tmp_result.max() * 255  # 此处是按照比例放大的

    """根据 tmp result 生成 Adjacent Matrix, 此处不能晚于 filling, 因为有0,0 的干扰"""
    AdjMatrix = np.zeros((maxPointFilled, maxPointFilled), dtype=np.float)

    eucDistance = cdist(tmp_result, tmp_result)
    reverseEucDistance: np.ndarray = 1 - eucDistance / (255 * np.sqrt(2))
    reverseEucDistance -= np.eye(coorLen)  # eye with be added in the graph normalization.

    """保留最近点"""
    # finalDistance = np.zeros_like(reverseEucDistance)
    # max_index = np.argmax(reverseEucDistance, axis=1)
    # row_index = np.array(list(range(reverseEucDistance.shape[0])))
    # sliceIndex = (row_index, max_index)
    # sliceIndex_T = (max_index, row_index)
    # # 注意 此处一定要生成对角阵
    # finalDistance[sliceIndex] = reverseEucDistance[sliceIndex]
    # finalDistance[sliceIndex_T] = reverseEucDistance[sliceIndex]

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
    # print(coorLen)
    if coorLen == 0:
        coordinates.append([1, 1])
        coordinates.append([1, 1])
        coorLen = 2
    tmp_result = np.array(coordinates, dtype=np.float)
    # tmp_result -= tmp_result.min()  # ? 此处是否要归一到0
    # tmp_result /= tmp_result.max()
    # tmp_result *= max_pixel_value  # out of range if not -1

    #     # tmp_result = tmp_result / tmp_result.max() * 255  # 此处是按照比例放大的

    """根据 tmp result 生成 Adjacent Matrix, 此处不能晚于 filling, 因为有0,0 的干扰"""
    AdjMatrix = np.zeros((maxPointFilled, maxPointFilled), dtype=np.float)

    eucDistance = cdist(tmp_result, tmp_result)
    reverseEucDistance: np.ndarray = 1 - eucDistance / (255 * np.sqrt(2))
    reverseEucDistance -= np.eye(coorLen)  # eye with be added in the graph normalization.

    """保留最近点"""
    # finalDistance = np.zeros_like(reverseEucDistance)
    # max_index = np.argmax(reverseEucDistance, axis=1)
    # row_index = np.array(list(range(reverseEucDistance.shape[0])))
    # sliceIndex = (row_index, max_index)
    # sliceIndex_T = (max_index, row_index)
    # # 注意 此处一定要生成对角阵
    # finalDistance[sliceIndex] = reverseEucDistance[sliceIndex]
    # finalDistance[sliceIndex_T] = reverseEucDistance[sliceIndex]

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
    tmp_result -= tmp_result.min()  # ? 此处是否要归一到0
    tmp_result /= tmp_result.max()
    tmp_result *= max_pixel_value  # out of range if not -1

    #     # tmp_result = tmp_result / tmp_result.max() * 255  # 此处是按照比例放大的

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
    if coorLen < maxPointFilled:  # 满足maxPointFilled训练要求
        fill = np.zeros((maxPointFilled - coorLen, 2), dtype=np.float)
        fill_labels = np.zeros((maxPointFilled - coorLen, 1), dtype="uint8")
        return np.concatenate([tmp_result, fill]), AdjMatrix, np.concatenate([labels, fill_labels])
    else:
        return tmp_result[:maxPointFilled], AdjMatrix, labels[:maxPointFilled]


if __name__ == '__main__':
    # dataset = np.load("./shoev2_all_edges.npz", allow_pickle=True, encoding="latin1")
    # dataset = np.load("./chair_dataset/chairv2_all_edges.npz", allow_pickle=True, encoding="latin1")
    # dataset = np.load("./shoes_datasets/shoev2_all_edges.npz", allow_pickle=True, encoding="latin1")

    dataset = np.load("../dataset/butterfly.npz", allow_pickle=True, encoding="latin1")

    dataset_train = dataset["train"]
    print(list(dataset.keys()))
    nodesSum = 0

    sumTime = 0
    for index, each_sketch in enumerate(dataset_train):
        sample_g: np.ndarray = each_sketch
        s = time.time()
        print("start", )
        canvas = draw_three(sample_g, thickness=4, )
        print("canvas finished", time.time() - s)
        s = time.time()
        result_points, A = get_node_coordinates_graph(canvas, 8, 8, maxPointFilled=80,
                                                      mask_prob=0., max_pixel_value=255)
        sumTime += time.time() - s
        print("A finished", time.time() - s)

        # result_points = result_points.astype("int16")
        # for p in result_points:
        #     if p[0] == p[1] == 0:
        #         continue
        #     cv2.line(canvas, tuple(p), tuple(p), color=(0, 255, 0), thickness=5)
        # cv2.imwrite("test.jpg", canvas)

        if index > 100:
            break
        # canvas = padding_image(canvas, padding=64)
        # result_points = get_node_coordinates(canvas, 8, 8)
        # print(result_points)
        result_points = result_points.astype("int")
        h, w, _ = canvas.shape
        for p in result_points:
            cv2.line(canvas, tuple(p), tuple(p), color=(0, 0, 255), thickness=5)
        cv2.imshow("resuilt 2", canvas)
        key = cv2.waitKeyEx()
        if key == 27:
            exit(0)
    print(sumTime)
    exit(0)

"""
PPPPPPPAAAAAAAATTTTTTTTTCCCCCCCCCCCCHHHHHHHHH  PART
PPPPPPPAAAAAAAATTTTTTTTTCCCCCCCCCCCCHHHHHHHHH  PART
PPPPPPPAAAAAAAATTTTTTTTTCCCCCCCCCCCCHHHHHHHHH  PART
PPPPPPPAAAAAAAATTTTTTTTTCCCCCCCCCCCCHHHHHHHHH  PART
PPPPPPPAAAAAAAATTTTTTTTTCCCCCCCCCCCCHHHHHHHHH  PART
PPPPPPPAAAAAAAATTTTTTTTTCCCCCCCCCCCCHHHHHHHHH  PART
PPPPPPPAAAAAAAATTTTTTTTTCCCCCCCCCCCCHHHHHHHHH  PART
PPPPPPPAAAAAAAATTTTTTTTTCCCCCCCCCCCCHHHHHHHHH  PART
"""
#
# def point_in_box(pointXY,
#                  x, y, size, closed=True):
#     """
#     此处用于返回一个点是否在区域内
#     :param x: 区域左上角坐标x
#     :param y: 区域左上角坐标y
#     :param size: 长度, 默认正方形
#     :return: 如果在内部, 则返回该点, 反之返回None
#     """
#     if (pointXY[0] in Interval(x, x + size, closed=closed)) and (
#             pointXY[1] in Interval(y, y + size, closed=closed)):
#         return [True, pointXY]
#     else:
#         return [False, None]
#
#
# def new_point(point1, point2, pos, mode=0):
#     """
#     此处, 根据两点构成的直线, 计算pos(X或者Y坐标)下, 产生的新的坐标点.
#     :param point1: 任意一点
#     :param point2: 任意不同于point1的一点
#     :param pos: 任意一处坐标, X 或者 Y
#     :param mode: 0 为 横坐标X固定, 计算Y坐标; 其他为 纵坐标Y固定, 计算X坐标
#     :return: 新产生的坐标
#     """
#     assert not (point1[0] == point2[0] and point1[1] == point2[1])  # 若两点相同, 则不构成直线. 斜率不存在.
#     if mode == 0:
#         if point1[0] == point2[0]:
#             return [False, None]
#         y = int(point1[1] + (point2[1] - point1[1]) * (pos - point1[0]) / (point2[0] - point1[0]))
#         return [True, np.array([pos, y], dtype="int16")]
#     else:
#         if point1[1] == point2[1]:
#             return [False, None]
#         x = int(point1[0] + (point2[0] - point1[0]) * (pos - point1[1]) / (point2[1] - point1[1]))
#         return [True, np.array([x, pos], dtype="int16")]
#
#
# def point_on_line(point, line_point1, line_point2, closed=True):
#     """
#     ***注意, 此处的交点是 **用这条直线求出的交点**
#     此处判断 **交点**是否在线段上
#     即, 带球点的横坐标以及纵坐标是否在直线范围内.
#     :param point: 待求点
#     :param line_point1: 线段一个端点
#     :param line_point2: 线段另一个端点
#     :param closed: 默认在线段两边的 不算交点.
#     :return:
#     """
#     if point[0] in Interval(int(line_point1[0]), int(line_point2[0]), closed=closed) and \
#             point[1] in Interval(int(line_point1[1]), int(line_point2[1]), closed=closed):
#         return True
#     else:
#         return False
#
#
# def distance(point1, point2):
#     """
#     模方
#     :param point1:
#     :param point2:
#     :return:
#     """
#     return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
#
#
# def scale_sketch_ours(sketch, size=(256, 256)):
#     """
#     将输入的sketch缩放到指定size大小.
#     此函数仅支持绝对坐标. 相对坐标请调用scale_sketch_google
#     注意, 此处并没有删除留白. 删除留白请调用remove_white_space_sketch()
#     :param sketch: n*5 ours [[x, y, p1, p2, p3], [...], ...]
#     :param size:
#     :return:
#     """
#     sketch[:, 0] = sketch[:, 0] / np.max(sketch[:, 0]) * size[0]
#     sketch[:, 1] = sketch[:, 1] / np.max(sketch[:, 1]) * size[1]
#     if sketch.dtype != np.int:
#         print("type of data changing.")
#         sketch = sketch.round().astype("int16")  # 有可能有小数.
#     return sketch
#
#
# def add_white_space_sketch(sketch: np.ndarray, padding=30):
#     """
#     为sketch添加留白, 实际上就是将绝对坐标[:2] 统一添加white
#     注意: sketch中 是没有办法表达右下侧留白的(因为实际上 sketch不像pix图, 只有在canvas上画图的时候才能体现)
#     :param sketch: n*5 ours
#     :param padding: 单边宽度
#     :return: sketch
#     """
#     sketch[:, :2] += np.array([padding, padding], dtype=sketch.dtype)  # 添加white, 保持原有的数据格式
#     return sketch
#
#
# def add_white_space_image(img_np: np.ndarray, padding=30):
#     """
#     为pix的image添加留白, pix图可以直接添加四周的留白. 即 padding.
#     注意, 此处的留白全是255. 如果是1. 的形式
#     :param img_np:
#     :param padding:
#     :return:
#     """
#     if img_np.max() <= 1 or "float" in str(img_np.dtype):
#         img_np *= 255
#         img_np = img_np.astype("uint8")
#     return cv2.copyMakeBorder(img_np, padding, padding, padding, padding,
#                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
#
#
# def npz_ours2google_generate(our_set, max_len=180):
#     """
#     :param our_set:  sketches n*5
#     :return: "Success"
#     """
#     ###################
#     # 生成训练要求的npz
#     # PS:test和valid相同
#     our_test = []
#     our_valid = []
#     our_train = []
#     # for index, photo in enumerate(our_set[f"train_photos"]):
#     #     # print(photo.shape)
#     #     # print(type(photo))  # float64
#     #     photo = photo * 255
#     #     photo = photo.astype("uint8")  # 原来是float64 没办法压缩成uint8
#     #     photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
#     #     photo = cv2.resize(photo, (48, 48), cv2.INTER_LINEAR)
#     #     cv2.imwrite(f"./shoe_photos/train/{index}.png", photo)
#     #     print(index)
#     #     # exit(0)
#     #     pass
#     # for index, photo in enumerate(our_set["test_photos"]):
#     #     pass
#     # exit(0)
#
#     type = "strokes"
#
#     # train
#     for index, sketch in enumerate(our_set[f"train_{type}"]):
#         sketch = five_rdp_with_discard(sketch, epsilon=2, discard=5)
#         if sketch.shape[0] > max_len:
#             print(f"len of {index} larger then {max_len}")
#             continue
#         if sketch.shape == (0, 5):
#             continue
#         else:
#             five2png_save(f"./shoe_{type}_png/train/48x48/{index}.png", sketch)
#         print(index)
#         sketch = trans_ours2google(sketch)
#         our_train.append(sketch)
#     our_train = np.array(our_train)
#
#     # test
#     for index, sketch in enumerate(our_set[f"test_{type}"]):
#         sketch = five_rdp_with_discard(sketch, epsilon=2, discard=5)
#         if sketch.shape[0] > max_len:
#             print(f"len of {index} larger then {max_len}")
#             continue
#         if sketch.shape != (0, 5):
#             five2png_save(f"./shoe_{type}_png/test/48x48/{index}.png", sketch)
#         else:
#             continue
#         sketch = trans_ours2google(sketch)
#         our_test.append(sketch)
#         print(index)
#     our_test = np.array(our_test)
#
#     # valid
#     for index, sketch in enumerate(our_set[f"test_{type}"]):
#         sketch = five_rdp_with_discard(sketch, epsilon=2, discard=5)
#         if sketch.shape[0] > max_len:
#             print(f"len of {index} larger then {max_len}")
#             continue
#         if sketch.shape != (0, 5):
#             five2png_save(f"./shoe_{type}_png/valid/48x48/{index}.png", sketch)
#         else:
#             continue
#         sketch = trans_ours2google(sketch)
#         our_valid.append(sketch)
#         print(index)
#     our_valid = np.array(our_valid)
#
#     # save now
#     print(our_train.shape,
#           our_test.shape,
#           our_valid.shape)
#     np.savez(f"./shoe_{type}_ours2google.npz",
#              train=our_train,
#              test=our_test,
#              valid=our_valid)
#     print("generation end")
#     # generation end
#     ####################
#     return
#
#
# def get_patch(sketch, img: np.ndarray,
#               x: int = 128, y: int = 30, size: int = 64, padding=0):
#     """
#     此处针对图片的crop不是简单的坐标对应, 因为sketch canvas的大小和img的大小极有可能不一样.
#     所以
#     1. 获取sketch canvas画布大小; sketch是**无留白的**
#     4. 对图片进行边缘裁剪(bounding), 对图片按照sketch canvas的尺寸进行resize.
#     2. sketchpatch 和 imgpatch 的留白.
#     3. 生成patch
#     :param sketch: 需要取patch的sketch. n*5 ours
#     :param x: patch左上角坐标
#     :param y: patch左上角坐标
#     :param size: patch大小
#     :return: 返回值有两个部分, 分别是sketch-patch, img-patch
#     """
#     # 大小必须要是4的倍数
#     assert size >= 0 and isinstance(size, int) and size % 4 == 0
#     # 1
#     # sketch = remove_white_space_sketch(sketch)  # 去除留白
#     # sketch = scale_sketch_ours(sketch, size=(256, 256))  # scale 放大
#     sketch_w, sketch_h = canvas_size_ours(sketch)  # 获得画布大小
#     # print(f"sketch宽高: {sketch_w}, {sketch_h}")
#     canvas = np.ones((sketch_h + padding, sketch_w + padding, 3), dtype="uint8") * 255
#     # 2
#     result_sketch_raw = []  # 单纯按顺序保存笔画, 最终结果需要清洗.
#     # 3
#     for index, eachStroke in enumerate(sketch):
#         if int(eachStroke[3]) == 1:
#             continue
#         if int(eachStroke[4]) == 1:
#             break
#         pre_point = sketch[index][0:2]
#         nxt_point = sketch[index + 1][0:2]
#         # 对笔画的过滤操作
#         """
#         逻辑:
#         patch的边框是固定的, 所以以边框为Interval的区间值.
#         pre_point = [x1, y1]
#         nxt_point = [x2, y2]
#         以水平方向为例, 区间为[H1, H2]
#         有以下三种通用情况:
#             1. 水平方向线全部落在区间外;
#                 1.1. 横跨区间
#                 2.2. 都在区间一侧
#             2. 单侧落在区间内;
#             3. 两点都落在区间内;
#         只有同时满足 水平方向和数值方向都有区域内笔画的时候, 才画.
#         注意: 就算都落在区间内, patch中也不一定有笔画, 因为使用的是lineto, 所以有可能出现区域相交, 但是线段不包含
#
#         !!!解决方法
#         这个任务实际上是在一个坐标系中, 任取两个点, 计算这条无限长的直线在与一个区域的四个边界的焦点,
#         得到6(2+4)个点之后, 取在范围内的两个点即可: 注意 取更靠内的两个点
#         """
#         # 原图
#         cv2.line(canvas, tuple(pre_point), tuple(nxt_point),  # 如果 是相同的点, 则会画一个点
#                  (0, 0, 0), thickness=2)
#         cv2.line(canvas, tuple(pre_point), tuple(pre_point),  # 画一个点
#                  (0, 0, 255), thickness=2)
#
#         """
#         首先实现 全在内部的stroke 画出来
#         """
#         if point_in_box(pre_point, x, y, size)[0] and point_in_box(nxt_point, x, y, size)[0]:
#             cv2.line(canvas, tuple(pre_point), tuple(nxt_point),  # 如果 是相同的点, 则会画一个点
#                      (0, 97, 255), thickness=2)
#             """获得sketch_patch"""
#             result_sketch_raw.append(pre_point)
#             result_sketch_raw.append(nxt_point)
#
#         """
#         两个点都在外部. 按照直接求四个交点然后确认区域内两点时, 可能会有 线段所在直线延长与区域相交, 这种笔画是不能要的.
#         而如果线段会跨过区域, 则会产生交线.
#         根据以上情况, 有效线段有两个先决条件:
#             1. 两个点都在外部(这是一种情况)
#             2. 两个点跨过区域.(即, 排除两个点在矩形一侧的情况)
#         无效的线 包括一种情况: 都在外部; 两个点不在同侧. 这种线会进入这个if, 但是不会画出线, 因为得到的4点不在区域上.
#         """
#
#         """
#         两个点全在外部的情况"""
#         if all((not point_in_box(pre_point, x, y, size)[0], not point_in_box(nxt_point, x, y, size)[0])):
#             # 两个点不在同一侧; 此处叫做 快速矩形判别法, 速度比较快.
#             if not any(
#                     ((pre_point[0] < x and nxt_point[0] < x),
#                      (pre_point[1] < y and nxt_point[1] < y),
#                      (pre_point[0] > x + size and nxt_point[0] > x + size),
#                      (pre_point[1] > y + size and nxt_point[1] > y + size))
#             ):
#                 pointList = []
#                 for index, eachPos in enumerate([x, y, x + size, y + size]):
#                     pointList.append(new_point(pre_point, nxt_point, eachPos, mode=index % 2))
#                 # 生成四个点  有可能会有None
#                 pointList = [list(_x[1]) for _x in list(filter(lambda _x: _x[0], pointList))]
#                 # 将在区域内的点过滤出来
#                 resultPointList = []
#                 for index, each in enumerate(pointList):
#                     if point_in_box(each, x, y, size)[0]:
#                         resultPointList.append(each)
#                 cv2.line(canvas, tuple(pre_point), tuple(nxt_point),
#                          (0, 0, 255), thickness=2)  # 此处不是结果, 只是画出满足以上条件的线而已.
#
#                 # 内部一定会有两个交点, 正好跟边角只有一个交点略过.
#                 if len(resultPointList) == 2:
#                     cv2.line(canvas, tuple(pre_point), tuple(nxt_point),
#                              (255, 0, 0), thickness=2)  # 蓝色的原始线  此处不是结果 仅仅为了可视化
#
#                     # 此处保证笔画的方向性.(n*5中, 即顺序)
#                     if distance(pre_point, resultPointList[0]) <= distance(pre_point, resultPointList[1]):
#                         cv2.line(canvas, tuple(resultPointList[0]), tuple(resultPointList[1]),
#                                  (0, 255, 0), thickness=2)
#                         """获得sketch_patch"""
#                         result_sketch_raw.append(resultPointList[0])
#                         result_sketch_raw.append(resultPointList[1])
#                     else:
#                         cv2.line(canvas, tuple(resultPointList[1]), tuple(resultPointList[0]),
#                                  (0, 255, 0), thickness=2)
#                         """获得sketch_patch"""
#                         result_sketch_raw.append(resultPointList[1])
#                         result_sketch_raw.append(resultPointList[0])
#         """
#         一个点在内部 一个点在外部"""
#         if point_in_box(pre_point, x, y, size)[0] ^ point_in_box(nxt_point, x, y, size)[0]:
#             resultPoint = None
#             for index, eachPos in enumerate([x, y, x + size, y + size]):
#                 if new_point(pre_point, nxt_point, eachPos, mode=index % 2)[0]:
#                     point = new_point(pre_point, nxt_point, eachPos, mode=index % 2)[1]
#                     """
#                     判断是否在线段上并且在方框内, 这种情况和矩形只会有一个交点, 所以可以直接break
#                     注意:仅仅判断是否在线段上是不够的, 因为这种一条在内部, 一条在外部的线 最多可以获得两个在线段上的点,
#                         但是在box内部的只有一条.
#                     """
#                     if point_on_line(point, pre_point, nxt_point) and point_in_box(point, x, y, size)[0]:
#                         resultPoint = point
#                         break
#             # 如果是pre点 在 区域内, 那么就是pre-point; 反之, 是 point-nxt
#             if point_in_box(pre_point, x, y, size)[0]:
#                 assert resultPoint is not None
#                 cv2.line(canvas, tuple(pre_point), tuple(resultPoint),
#                          (0, 255, 0), thickness=2)
#                 """获得sketch_patch"""
#                 result_sketch_raw.append(pre_point)
#                 result_sketch_raw.append(resultPoint)
#             else:
#                 assert resultPoint is not None
#                 cv2.line(canvas, tuple(resultPoint), tuple(nxt_point),
#                          (0, 255, 0), thickness=2)
#                 """获得sketch_patch"""
#                 result_sketch_raw.append(resultPoint)
#                 result_sketch_raw.append(nxt_point)
#             pass
#         """
#         一笔一笔可视化.
#         """
#         # cv2.imshow("each stroke", canvas)
#         # if cv2.waitKeyEx() == 27:
#         #     exit(0)
#     """
#     对result_sketch_raw进行解析处理, 生成最终的sketch-patch
#     解析处理的数据结构
#     """
#     result_sketch_raw = np.array(result_sketch_raw)
#     # print(result_sketch_raw, len(result_sketch_raw),
#     #       result_sketch_raw.shape)
#     resultSketch = []
#     state_drawing = np.array([1, 0, 0], dtype="int16")
#     state_strokeFinish = np.array([0, 1, 0], dtype="int16")
#     state_allFinish = np.array([0, 0, 1], dtype="int16")
#     for index, each in enumerate(result_sketch_raw):
#         if index == 0:  # 第一个, 直接append进去
#             resultSketch.append(np.concatenate((each, state_drawing)))
#             continue
#         if (index + 1) % 2 == 0:  # 偶数直接加
#             resultSketch.append(np.concatenate((each, state_drawing)))
#         else:  # 奇数先判断resultSketch的最后一个, 然后再考虑情况
#             if (resultSketch[-1] == np.concatenate((each, state_drawing))).all():  # 是连笔
#                 # print("连笔")
#                 continue  #
#             else:  # 不是连笔
#                 resultSketch[-1][2:] = state_strokeFinish
#                 resultSketch.append(np.concatenate((each, state_drawing)))
#         if index == len(result_sketch_raw) - 1:  # 最后一笔
#             # print("last")
#             resultSketch[-1][2:] = state_allFinish
#     if not resultSketch:  # 如果sketch 是空, 直接退出即可.
#         return []
#     resultSketch = np.array(resultSketch)
#     resultSketch[:, 0] -= x  # 将patch 脱离原有的图
#     resultSketch[:, 1] -= y
#     # 由此, patch 生成完毕
#
#     img_cropped = img[y:y + size, x:x + size, :]
#     if np.min(img_cropped) >= 250:  # 可以近似看做全白图
#         return []
#     return [resultSketch, img_cropped]
#
#
# if __name__ == '__main__' and False:
#     import time
#     import glob
#
#     ########
#     # sample show
#     # sample select from dataset.
#     # load all dataset
#     data = np.load("chair_dataset/chair_total_data_with_edge.npz",
#                    encoding='latin1', allow_pickle=True)
#     data = np.load("shoes_datasets/shoev2_total_data_with_edge_nobias_100.npz ",
#                    encoding='latin1', allow_pickle=True)
#     savePath = "./patch_dataset_test/"
#     sample_strokes = data["test_strokes"]
#     sample_photos = data["test_photos"]
#     sample_edges = data["test_edges"]
#     print(len(sample_strokes), len(sample_photos))
#     for num, each in enumerate(sample_strokes):
#         sample_stroke = sample_strokes[num]
#         sample_photo = sample_photos[num]
#         sample_edge = five_rdp_with_discard(sample_edges[num], epsilon=2, discard=5)  # use rdp
#         # print("edge", sample_edges.shape,
#         #       "photos", sample_photos.shape,
#         #       "stroke", sample_strokes.shape)  # edge (279, 5) photos (100, 100, 3) stroke (68, 5)
#         """
#         add preprocess here
#         Test Block
#         """
#         sample_edge = remove_white_space_sketch(sample_edge)
#         sample_edge = scale_sketch_ours(sample_edge, size=(226, 226))
#         sample_edge = add_white_space_sketch(sample_edge, padding=15)
#
#         sample_stroke = remove_white_space_sketch(sample_stroke)  # 去除留白
#         sample_stroke = scale_sketch_ours(sample_stroke, size=(226, 226))  # scale 放大
#         sample_stroke = add_white_space_sketch(sample_stroke, padding=15)
#
#         sample_photo = remove_white_space_image(sample_photo)
#         sample_photo = cv2.resize(sample_photo, (226, 226))
#         sample_photo = add_white_space_image(sample_photo, padding=15)
#
#         cv2.imshow("chair photo", sample_photo)
#         draw_five(sample_stroke, "stroke", time=1)
#         draw_five(sample_edge, "edge", time=1)
#         exit(0) if 27 == cv2.waitKeyEx() else ""
#
#         for count in range(16):
#             print(num, count)
#             x_rand = np.random.randint(0, 256 - 64)
#             y_rand = np.random.randint(0, 256 - 64)
#             patch_pair_sketch = get_patch(sample_stroke, sample_photo, x=x_rand, y=y_rand, size=64, padding=15)
#             patch_pair_edge = get_patch(sample_edge, sample_photo, x=x_rand, y=y_rand, size=64, padding=15)
#             continue
#             if patch_pair_sketch and patch_pair_edge:  # 两者都满足 1. skroke != [], 2. image != white\
#                 draw_five(patch_pair_sketch[0], time=0, windowname="sketch")
#                 draw_five(patch_pair_edge[0], time=0, windowname="edge")
#                 cv2.imshow("test", patch_pair_sketch[1])
#                 if 27 == cv2.waitKeyEx():
#                     exit(0)
#                 continue
#                 np.save(f"{savePath}/sketch_{num}_{x_rand}_{y_rand}.npy", patch_pair_sketch[0])
#                 np.save(f"{savePath}/edge_{num}_{x_rand}_{y_rand}.npy", patch_pair_edge[0])
#                 cv2.imwrite(f"{savePath}/img_{num}_{x_rand}_{y_rand}.png", patch_pair_sketch[1])
