'''
Measurement infrastructure used for runners.
'''

import time
import threading
import psutil
import os
from typing import Callable, Tuple, List
from dataclasses import dataclass, field

@dataclass
class BenchmarkResult:
    '''
    Contains collected metrics for a single benchmark run
    '''
    label: str #Execution mode
    algorithm: str #Which alg is being measured
    size: int #Workload magnitude 
    num_workers: int #Number of parallel processes or Ray tasks
    elapsed_sec: float #Wall clock exec time
    peak_memory_mb: float #Highest RSS mem reading during the run in MB
    avg_cpu_pct: float #Average cpu utilisation across all cores during run
    speedup: float = field(default=0.0) #elapsed_sec of single process baseline / this runs elapsed sec. set post-hoc; 0.0 means "not yet computed"

class ResourceMonitor:
    '''
    Polls CPU utilisation and process RSS mem on a background thread while a benchmark task runs on the main thread.
    '''

    POLL_INTERVAL: float = 0.1 #Secs between samples. 100ms balances accuracy vs overhead.

    def __init__(self) -> None:
        #Track the current process so mem readings are scoped to our program
        self._proc: psutil.Process = psutil.Process(os.getpid())

        #Threading event used to signal the background thread to stop
        self._stop: threading.Event = threading.Event()

        self._cpu_samples: List[float] = [] #One reading per poll interval
        self._mem_samples: List[float] = [] #RSS in MB per poll interval

        #daemon=True makes sure thread does not prevent interpreter shutdown if main prog exits unexpectdly
        self._thread: threading.Thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        '''
        Clear any previous stop signal and launch the polling thread
        '''
        self._stop.clear()
        self._thread.start()

    def stop(self) -> Tuple[float, float]:
        '''
        Signal the polling thread to stop, wait for it to finish, and return summary stats

        Returns (avg_cpu_pct, peak_mem_mb)
        '''
        self._stop.set()
        self._thread.join()  #Wait until the thread has fully exited

        avg_cpu:  float = sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else 0.0
        peak_mem: float = max(self._mem_samples, default=0.0)
        return avg_cpu, peak_mem
    
    def _run(self) -> None:
        '''
        Main loop executed on the background thread. Appends one cpu % and one RSS mem sample per POLL_INTERVAL.
        '''

        while not self._stop.is_set():
            try:
                #cpu_percent(interval=None) returns usage since the last call (or since process start on the first call) — non-blocking.
                self._cpu_samples.append(psutil.cpu_percent(interval=None))

                #rss = Resident Set Size: physical RAM currently used by the process
                self._mem_samples.append(self._proc.memory_info().rss / 1024 / 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                #Process ended before the thread was stopped — exit gracefully
                break
            time.sleep(self.POLL_INTERVAL)

def run_benchmark(label: str, algorithm: str, size: int, num_workers: int, fn: Callable[[], None]) -> BenchmarkResult:
    '''
    Execute fn() once, measure wall clock time, peak mem, and avg cpu util
    '''

    #Start resource monitoring before launching a task
    monitor = ResourceMonitor()
    monitor.start()

    t0 = time.perf_counter()
    fn() #Excecute workload
    elapsed = time.perf_counter() - t0

    #Stop monitoring and collect summary stats
    avg_cpu, peak_mem = monitor.stop()

    return BenchmarkResult(
        label=label,
        algorithm=algorithm,
        size=size,
        num_workers=num_workers,
        elapsed_sec=round(elapsed, 4),
        peak_memory_mb=round(peak_mem, 2),
        avg_cpu_pct=round(avg_cpu, 2),
    )

