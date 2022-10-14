"""
添加
"""
import random

classes = ("O", "A", "B", "C")

if __name__ == '__main__':
    with open("train.txt") as f, open("train.txt", "w") as fw:
        for l in f:
            cls = random.sample(classes, 1)[0]
            if l.strip() != "":
                new_line = "{}\t{}\n".format(l.strip("\n"), cls)
            else:
                new_line = "\n"
            fw.write(new_line)
