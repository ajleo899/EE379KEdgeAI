#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 22:16:43 2021

@author: dawei
"""

import sysfs_paths as sysfs
import subprocess
import telnetlib as tel
import time
from timeit import default_timer as timer


def getTelnetPower(SP2_tel, last_power):
    """
    read power values using telnet.
    """
	# Get the latest data available from the telnet connection without blocking
    tel_dat = str(SP2_tel.read_very_eager()) 
    print('telnet reading:', tel_dat)
    # find latest power measurement in the data
    findex = tel_dat.rfind('\n')
    findex2 = tel_dat[:findex].rfind('\n')
    findex2 = findex2 if findex2 != -1 else 0
    ln = tel_dat[findex2:findex].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power

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
#SP2_tel = tel.Telnet("192.168.4.1")
#power = 0.0
print("Start Time: ", time.time())#time_stamp
command = "taskset --all-tasks 0xF0 ./parsec_files/bodytrack ./parsec_files/sequenceB_261 4 260 3000 8 3 4 0"   # 0x20: core 5

proc_ben = subprocess.call(command.split())
print('End Time: ', time.time())
#power = getTelnetPower(SP2_tel, power)
#print("Total power for Benchmark: ", power)
time_count = timer() - start
print("Runtime is:", time_count)