#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 22:16:43 2021

@author: dawei
"""

import sysfs_paths as sysfs
import subprocess
from timeit import default_timer as timer


def getAvailFreqs(cluster):
    """
    obtain the available frequency for a cpu. Return unit in khz by default!
    """
    # read cpu freq from sysfs_paths.py
    freqs = open(sysfs.fn_cluster_freq_range.format(cluster)).read().strip().split(' ')
    return [int(f.strip()) for f in freqs]

def getClusterFreq(cluster_num):
    """
    read the current cluster freq. cluster_num must be 0 (little) or 4 (big)
    """
    with open(sysfs.fn_cluster_freq_read.format(cluster_num), 'r') as f:
        return int(f.read().strip())

def setUserSpace(clusters=None):
    """
    set the system governor as 'userspace'. This is necessary before you can change the cluster/cpu freq to customized values
    """
    print("Setting userspace")
    clusters = [0, 4]
    for i in clusters:
        with open(sysfs.fn_cluster_gov.format(i), 'w') as f:
            f.write('userspace')

def setClusterFreq(cluster_num, frequency):
    """
    set customized freq for a cluster. Accepts frequency in khz as int or string
    """
    with open(sysfs.fn_cluster_freq_set.format(cluster_num), 'w') as f:
        f.write(str(frequency))	
        

       
print('available freq for little cluster:', getAvailFreqs(0))
print('available freq for little cluster:', getAvailFreqs(4))
setUserSpace()
setClusterFreq(4, 2000000)   # big cluster
setClusterFreq(0, 200000)
# print current freq for the big cluster
print('current freq for big cluster:', getClusterFreq(4))
# execution of your benchmark    
start=timer()
# run the benchmark
command = "taskset --all-tasks 0x10 /home/odroid/HW2_files/TPBench.exe"   # 0x20: core 5
proc_ben = subprocess.call(command.split())

time_count = timer() - start
print("Runtime is:", time_count)