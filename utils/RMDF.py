import numpy as np
import math
import copy


class RMDF:
    def __init__(self, depth=10, ascent_rate=20, start=np.array([0, 0]), end=np.array([1, 0])):
        control_points_num = depth * 2 + 1
        self.control_points = [[] for n in range(control_points_num)]
        self.control_points_copy = [[] for n in range(control_points_num)]
        self.anchor = [[] for n in range(control_points_num)]
        self.depth = depth
        self.start = start
        self.end = end
        self.ascent_rate = ascent_rate

    def gen_anchor(self):
        """
        根据相似三角形求解控制点坐标（引自TSAGen源码，感觉根本就没这么复杂，可能是故意这么写的）
        """
        start = self.start
        end = self.end
        self.control_points[0].append([[start[0], end[0]], start, end])
        for d in range(self.depth):
            for e in self.control_points[d]:
                start = e[1]
                end = e[2]
                length = self.__length(start, end)
                pmid = self.__mid(start, end)
                h = np.random.normal(0, length/self.ascent_rate)

                zeta = math.atan(h / (length/2))
                l2 = math.sqrt(h * h + (length/2) * (length/2))
                T = np.matrix([[math.cos(zeta), -math.sin(zeta)], [math.sin(zeta), math.cos(zeta)]])
                a = np.matrix([[pmid[0] - start[0]], [pmid[1] - start[1]]])
                b = np.matmul(T, a) * (l2/length * 2)
                p = np.array([start[0] + b[0, 0], start[1] + b[1, 0]])

                self.control_points[d + 1].append([[start[0], p[0]], start, p])
                self.control_points[d + 1].append([[p[0], end[0]], p, end])
            # ll = ll/2
        self.anchor = self.control_points.copy()
        # self.__std_anchor()
    
    def clear_all(self):
        self.__clear(self.depth+1)
        self.gen_anchor()

    def gen(self, forking_depth, length):
        self.__clear(forking_depth)
        self.__forking(forking_depth)
        self.__std()
        x_ = np.arange(0, 1, 1/length)
        y = np.array([self.__expression(x, 10) for x in x_])
        return y

    def __std_anchor(self):
        point_list = list(map(lambda x: x[2], self.anchor[10]))
        y_value_list = list(map(lambda x: x[1], point_list))
        max_y = np.max(y_value_list)
        min_y = np.min(y_value_list)
        height = max_y-min_y
        for i in range(len(self.anchor[10])):
            self.anchor[10][i][2][1] = self.anchor[10][i][2][1]/height

    def __std(self):
        point_list = list(map(lambda x: x[2], self.control_points_copy[10]))
        y_value_list = list(map(lambda x: x[1], point_list))
        max_y = np.max(y_value_list)
        min_y = np.min(y_value_list)
        height = max_y-min_y
        for i in range(len(self.control_points_copy[10])):
            self.control_points_copy[10][i][2][1] = self.control_points_copy[10][i][2][1]/height

    def __expression(self, x, depth):
        expression = self.control_points_copy[depth]
        for e in expression:
            if x >= e[0][0] and x <= e[0][1]:
                p1 = e[1]
                p2 = e[2]
                k = (p2[1]-p1[1])/(p2[0]-p1[0])
                b = p1[1]-k*p1[0]
                return k*x+b

    def __forking(self, forking_depth):
        shared_depth = self.depth - forking_depth
        for d in range(shared_depth, self.depth):
            for e in self.control_points[d]:
                start = e[1]
                end = e[2]
                l = self.__length(start, end)
                pmid = self.__mid(start, end)
                h = np.random.normal(0, l/self.ascent_rate)

                zeta = math.atan(h / (l/2))
                l2 = math.sqrt(h * h + (l/2) * (l/2))
                T = np.matrix([[math.cos(zeta), -math.sin(zeta)], [math.sin(zeta), math.cos(zeta)]])
                a = np.matrix([[pmid[0] - start[0]], [pmid[1] - start[1]]])
                b = np.matmul(T, a) * (l2/l * 2)
                p = np.array([start[0] + b[0, 0], start[1] + b[1, 0]])

                self.control_points[d + 1].append([[start[0], p[0]], start, p])
                self.control_points[d + 1].append([[p[0], end[0]], p, end])
        self.control_points_copy = copy.deepcopy(self.control_points)

    def __clear(self, forking_depth):
        # clear the latest forking_depth layer.
        shared_depth = self.depth - forking_depth
        for i in range(shared_depth, self.depth):
            self.control_points[i + 1] = []

    @staticmethod
    def __length(p1, p2):
        """
        计算两点之间距离

        :param p1: 端点坐标，np.array.
        :param p2: 端点坐标，np.array.
        :return: 两点之间的距离.
        """
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def __mid(p1, p2):
        """
        计算中点坐标

        :param p1: 端点坐标，np.array.
        :param p2: 端点坐标，np.array.
        :return: np.array，中点坐标.
        """
        # mid point of line(p1,p2)
        x = (p2[0] + p1[0])/2
        y = (p2[1] + p1[1])/2
        return np.array([x, y])
