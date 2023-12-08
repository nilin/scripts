import time
import os
import threading
from functools import partial

def wait_for_end(start_time=0):
    input('Press Enter to stop.\n')
    end_time = time.time()
    return end_time - start_time

def print_time(running,start_time=0):
    while running[0]:
        time.sleep(.005)
        dt=time.time() - start_time
        print(f"   {dt:.2f}", end='\r')

def save_time(dt):
    fn=input('Save as or append to (leave blank to skip): ')
    if fn=='':
        return
    name=os.path.join('stopwatchtimes',fn+'.txt')
    os.makedirs(os.path.dirname(name), exist_ok=True)
    with open(name, 'a') as f:
        f.write(str(dt)+'\n')

if __name__ == '__main__':

    input('Press Enter to start.')
    start_time = time.time()
    running=[True]

    threading.Thread(target=partial(print_time,running,start_time)).start()
    dt=wait_for_end(start_time)
    running[0]=False
    time.sleep(.05)

    save_time(dt)
