import cv2
import torch
from dataset import build_dataset
from models import RelationalNet


def translate(question, answer):
    colors = ['red ', 'green ', 'blue ', 'orange ', 'gray ', 'yellow ']
    answer_sheet = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6']
    query = ''
    query += colors[question.tolist()[0:6].index(1)]
    if question[6] == 1:
        if question[8] == 1:
            query += 'shape?'
        if question[9] == 1:
            query += 'left?'
        if question[10] == 1:
            query += 'up?'
    if question[7] == 1:
        if question[8] == 1:
            query += 'closest shape?'
        if question[9] == 1:
            query += 'furthest shape?'
        if question[10] == 1:
            query += 'count?'
    ans = answer_sheet[answer]
    return query, ans


if __name__ == '__main__':
    net = RelationalNet()
    net.load_state_dict(torch.load('./model/model.pth', 'cpu'))
    net.eval()
    image, rel_questions, rel_answers, norel_questions, norel_answers = build_dataset(1)
    image = image[0].unsqueeze(0)
    questions = torch.cat((rel_questions, norel_questions), 0)
    answers = torch.cat((rel_answers, norel_answers), 0)
    for question, answer in zip(questions, answers):
        pre = net(image, question).argmax(1).item()
        query, ans = translate(question, answer)
        print("Ground Truth:", query, '==>', ans)
        query, ans = translate(question, pre)
        print("Relational Net:", query, '==>', ans)
    cv2.imshow('Image', cv2.resize(image.squeeze().permute(1, 2, 0).numpy(), (512, 512)))
    cv2.waitKey(0)
