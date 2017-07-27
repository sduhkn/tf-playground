import time
from asyncio import coroutine
import re


def follow(thefile, target):
    target.send(None)
    while True:
        line = thefile.readline()
        print("do sth")
        time.sleep(2)
        if not line:

            break
        target.send(line)
    thefile.close()


def printer():
    while True:
        line = (yield 1)
        print(line)


def grep(pattern, target):
    next(target)
    p = re.compile(pattern)
    while True:
        line = (yield 1)
        if p.match(line):
            target.send(line)

def printer1():

    counter = 0
    print(11)
    while True:
        string = (yield 1)
        print('[{0}] {1}'.format(counter, string))
        counter += 1

if __name__ == '__main__':
    # f = open("test.txt")
    # follow(f, grep("python", printer()))
    p = printer1()
    next(p)
    try:
        for i in range(10):
            p.send(i)
            if i == 11:
                p.close()
    except StopIteration:
        print("停止")
    else:
        print("wow")