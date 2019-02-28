import os
import cv2
import torch
import numpy as np
import torch.utils.data as data

train_size = 9800
test_size = 200
image_size = 75
size = 5
question_size = 11
# 6 for one-hot vector of color,
# 2 for question type,
# 3 for question subtype
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

nb_questions = 10

colors = [
    (0, 0, 255),  # 红
    (0, 255, 0),  # 绿
    (255, 0, 0),  # 蓝
    (0, 156, 255),  # 橙
    (128, 128, 128),  # 灰
    (0, 255, 255)  # 黄
]


class SortOfClevr(data.Dataset):
    def __init__(self, root, train=True):
        super().__init__()
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root)
        if not os.listdir(root) or not os.path.exists(os.path.join(self.root, 'test.pkl')) or \
                not os.path.exists(os.path.join(self.root, 'train.pkl')):
            print("generating data")
            self.generate()
        if train:
            self.data, self.r_qst, self.r_ans, \
            self.nor_qst, self.nor_ans = torch.load(self.root + '/train.pkl')
        else:
            self.data, self.r_qst, self.r_ans, \
            self.nor_qst, self.nor_ans = torch.load(self.root + '/test.pkl')

    def __getitem__(self, item):
        return self.data[item], self.r_qst[item], self.r_ans[item], \
               self.nor_qst[item], self.nor_ans[item]

    def __len__(self):
        return self.data.shape[0]

    def generate(self):
        print('making data...')
        train_data, train_rel_qst, train_rel_ans, train_norel_qst, train_norel_ans = build_dataset(train_size)
        test_data, test_rel_qst, test_rel_ans, test_norel_qst, test_norel_ans = build_dataset(test_size)
        print('saving data...')
        with open(os.path.join(self.root, 'train.pkl'), "wb")as f:
            torch.save((train_data, train_rel_qst, train_rel_ans, train_norel_qst, train_norel_ans), f)
        with open(os.path.join(self.root, 'test.pkl'), "wb")as f:
            torch.save((test_data, test_rel_qst, test_rel_ans, test_norel_qst, test_norel_ans), f)
        print("Finish!")


def center_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0 + size, image_size - size, 2)
        if len(objects) > 0:
            for name, c, shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center


def build_dataset(total):
    images = []
    relational_questions = []
    non_relational_questions = []
    relational_answers = []
    non_relational_answers = []
    for num in range(total):
        objects = []
        image = np.ones((image_size, image_size, 3), dtype=np.float32) * 255
        for color_id, color in enumerate(colors):
            center = center_generate(objects)
            if np.random.random() < 0.5:
                start = (center[0] - size, center[1] - size)
                end = (center[0] + size, center[1] + size)
                cv2.rectangle(image, start, end, color, -1)
                objects.append((color_id, center, 'r'))
            else:
                center_ = (center[0], center[1])
                cv2.circle(image, center_, size, color, -1)
                objects.append((color_id, center, 'c'))
        image = image / 255.
        image = np.swapaxes(image, 0, 2)
        image = np.expand_dims(image, 0).repeat(nb_questions, 0)
        images.append(image)
        """Non-relational questions"""
        for _ in range(nb_questions):
            question = np.zeros(question_size)
            color = np.random.randint(0, 5)
            question[color] = 1
            question[6] = 1
            subtype = np.random.randint(0, 2)
            question[subtype + 8] = 1
            non_relational_questions.append(question)
            """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
            if subtype == 0:
                """query shape->rectangle/circle"""
                if objects[color][2] == 'r':
                    answer = 2
                else:
                    answer = 3

            elif subtype == 1:
                """query horizontal position->yes/no"""
                if objects[color][1][0] < image_size / 2:
                    answer = 0
                else:
                    answer = 1

            elif subtype == 2:
                """query vertical position->yes/no"""
                if objects[color][1][1] < image_size / 2:
                    answer = 0
                else:
                    answer = 1
            non_relational_answers.append(answer)

        """Relational questions"""
        for i in range(nb_questions):
            question = np.zeros(question_size)
            color = np.random.randint(0, 5)
            question[color] = 1
            question[7] = 1
            subtype = np.random.randint(0, 2)
            question[subtype + 8] = 1
            relational_questions.append(question)
            if subtype == 0:
                """closest-to->rectangle/circle"""
                cur_object = objects[color][1]
                dist_list = [((cur_object - obj[1]) ** 2).sum() for obj in objects]
                dist_list[dist_list.index(0)] = 999
                closest = dist_list.index(min(dist_list))
                if objects[closest][2] == 'r':
                    answer = 2
                else:
                    answer = 3
            elif subtype == 1:
                """furthest-from->rectangle/circle"""
                cur_object = objects[color][1]
                dist_list = [((cur_object - obj[1]) ** 2).sum() for obj in objects]
                furthest = dist_list.index(max(dist_list))
                if objects[furthest][2] == 'r':
                    answer = 2
                else:
                    answer = 3
            elif subtype == 2:
                """count->1~6"""
                cur_object = objects[color][2]
                count = -1
                for obj in objects:
                    if obj[2] == cur_object:
                        count += 1
                answer = count + 4
            relational_answers.append(answer)
    images = np.concatenate(images, 0).astype(np.float32)
    non_relational_questions = np.stack(non_relational_questions, 0).astype(np.float32)
    relational_questions = np.stack(relational_questions, 0).astype(np.float32)
    non_relational_answers = np.array(non_relational_answers, dtype=np.int64)
    relational_answers = np.array(relational_answers, dtype=np.int64)
    return torch.from_numpy(images), torch.from_numpy(relational_questions), \
           torch.from_numpy(relational_answers), torch.from_numpy(non_relational_questions), \
           torch.from_numpy(non_relational_answers)
