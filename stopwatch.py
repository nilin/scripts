import time
import datetime
import os
import threading
import sys
class Timer:

    def __init__(self):
        self.start_time = 0
        self.intervals=[]
        self.running=False

    def start(self):
        self.start_time = time.time()
        self.running=True

    def stop(self):
        self.running=False
        self.intervals.append(time.time() - self.start_time)

    def print_time(self):
        while self.running:
            dt=time.time() - self.start_time
            print(f"   {dt:.2f}", end='\r')
            time.sleep(.01)
        sys.exit()

    def get_time_till_Enter(self):
        return self.stop()

    def save_time(self,fn):
        name=os.path.join('stopwatchtimes',fn+'.txt')
        os.makedirs(os.path.dirname(name), exist_ok=True)
        for dt in self.intervals:
            with open(name, 'a') as f:
                f.write(str(dt)+'\n')
        print(f'Saved splits to \n{name}')
        return name

    def print_total_time(self):
        T=sum(self.intervals)
        print(f'\nTotal time\n'+str(datetime.timedelta(seconds=T)))

if __name__ == '__main__':

    timer=Timer()

    while True:
        timer.print_total_time()
        fn=input("\nPress Enter to start,\n"+\
                 "q+Enter to quit,\n"+\
                 "or enter [name] to save splits.\n")
        if fn=='q':
            break
        elif fn!='':
            name=timer.save_time(fn)
            timer.print_total_time()
            break

        timer.start()
        threading.Thread(target=timer.print_time).start()
        input('Press Enter to pause.\n')
        timer.stop()

