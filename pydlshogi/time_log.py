import time

class TimeLog():
    measure_list = {}

    def __init__(self,name):
        self.name = name
        self.start_time = None
        self.cnt = 0
        self.sum_time = 0
        self.error_cnt = 0
        if not name in TimeLog.measure_list:
            TimeLog.measure_list[name] = self

    def start(self):
        if self.start_time is not None:
            self.error_cnt += 1

        self.start_time = time.time()

    def end(self):
        if self.start_time is None:
            self.error_cnt += 1
        else:
            self.sum_time += time.time() - self.start_time
            self.cnt += 1
            self.start_time = None
               
    @staticmethod
    def debug():
        str = ''
        for name,timelog in TimeLog.measure_list.items():
            str += f'{name}'
            str += f',{timelog.cnt}'
            str += f',{timelog.sum_time}'
            str += f',{0 if timelog.cnt == 0 else timelog.sum_time / timelog.cnt}'
            str += f',{timelog.error_cnt}'
            str += '\n'
        return str
