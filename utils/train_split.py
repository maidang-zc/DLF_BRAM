import os
import csv
import random
import pickle

def pathlist(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader]

def train_test_split_bg(bg_face_path, flag=1):
    trainpath = f"./data/Bag/split/train{flag}.csv"
    testpath = f"./data/Bag/split/test{flag}.csv"
    with open(trainpath, 'r') as f:
        trainlines = f.readlines()
    with open(testpath, 'r') as f:
        testlines = f.readlines()
    train = [i.split(',')[0] for i in trainlines]
    test = [i.split(',')[0] for i in testlines]
    return train, test


def train_test_split_MU3d(MU3d_face_path, flag=1):
    user = os.listdir(MU3d_face_path)
    random.shuffle(user)
    part1 = [ui for ui in user if int(ui.split('_')[0][2:]) < 10]
    part2 = [ui for ui in user if 10 <= int(ui.split('_')[0][2:]) < 19]
    part3 = [ui for ui in user if int(ui.split('_')[0][2:]) >= 19]

    if flag == 1:
        train = part1 + part2
        test = part3
    elif flag == 2:
        train = part1 + part3
        test = part2
    elif flag == 3:
        train = part2 + part3
        test = part1
    else:
        raise ValueError("flag must be 1, 2, or 3")
    return train, test

def train_test_split_Dolos(Dolos_face_path, flag=1):
    trainpath = f"./data/Dolos/Training_Protocols/train_fold{flag}.csv"
    testpath = f"./data/Dolos/Training_Protocols/test_fold{flag}.csv"
    dirlist = os.listdir(Dolos_face_path)
    trainlist = pathlist(trainpath)
    testlist = pathlist(testpath)

    dirset = set(dirlist)
    trainset = set(trainlist)
    testset = set(testlist)
    train_dir = trainset & dirset
    test_dir = testset & dirset
    return list(train_dir), list(test_dir)
