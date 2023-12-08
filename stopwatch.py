import time
import os
import threading
import sys
class Timer:

    def start(self):
        self.start_time = time.time()
        self.running=True

    def stop(self):
        self.running=False
        self.dt=time.time() - self.start_time

    def print_time(self):
        while self.running:
            dt=time.time() - self.start_time
            print(f"   {dt:.2f}", end='\r')
            time.sleep(.001)
        sys.exit()

    def get_time_till_Enter(self):
        return self.stop()

    @staticmethod
    def save_time(intervals,fn):
        name=os.path.join('stopwatchtimes',fn+'.txt')
        os.makedirs(os.path.dirname(name), exist_ok=True)
        for dt in intervals:
            with open(name, 'a') as f:
                f.write(str(dt)+'\n')
        print(f'Saved splits to \n{name}')

if __name__ == '__main__':

    timer=Timer()
    intervals=[]

    while True:
        fn=input("\nPress Enter to start,\n"+\
                 "q+Enter to quit,\n"+\
                 "or enter [name] to save splits.\n")
        if fn=='q':
            break
        elif fn!='':
            Timer.save_time(intervals,fn)
            break

        timer.start()
        threading.Thread(target=timer.print_time).start()
        input('Press Enter to pause.\n')
        timer.stop()
        intervals.append(timer.dt)

