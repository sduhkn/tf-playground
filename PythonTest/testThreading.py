import threading
import time

l = [1,2,3,4,5,6]
thread_number = 2
segment = len(l) // thread_number
def action(data, i):
    time.sleep(1)
    thread_name = threading.currentThread().getName()
    for i in range(10):
        print("{0}: {1}".format(thread_name, i))
    print(thread_name + "the data is {}".format(data))
thread_list = []
for i in range(thread_number):
    start = i * segment
    end = start+segment
    t = threading.Thread(target=action, args=(l[start:end],i))
    t.start()
    thread_list.append(t)

for i in range(thread_number):
    thread_list[i].join(2000)
print("main thread start")