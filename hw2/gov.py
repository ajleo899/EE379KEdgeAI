#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 23:49:12 2021

@author: dawei
"""

import time
import sysfs_paths as sysfs
import psutil
import math


def getTemps():
    """
    obtain the temp values from sysfs_paths.py
    """
    templ = []
    # get temp from temp zones 0-3 (the big cores)
    for i in range(4):
        temp = float(file(sysfs.fn_thermal_sensor.format(i),'r').readline().strip())/1000
        templ.append(temp)
	# Note: on the 5422, cpu temperatures 5 and 7 (big cores 1 and 3, counting from 0) appear to be swapped. Therefore, swap them back.
    t1 = templ[1]
    templ[1] = templ[3]
    templ[3] = t1
    return templ

def getAvailFreqs(cpu_num):
	cluster = (cpu_num//4) * 4
	freqs = open(sysfs.fn_cluster_freq_range.format(cluster)).read().strip().split(' ')
	return [int(f.strip()) for f in freqs]

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

        
def getClusterFreq(cluster_num):
	with open(sysfs.fn_cluster_freq_read.format(cluster_num), 'r') as f:
		return int(f.read().strip())

def getCpuLoad():
	"""
	Returns the cpu load as a value from the interval [0.0, 1.0]
	"""
	loads = [x/100 for x in psutil.cpu_percent(interval=None, percpu=True)]   # 2p here!!!
	return loads


if __name__ == "__main__":
    cluster = 4
    REFRESH_PERIOD = 0.2
    #MAX_TEMP, MIN_TEMP = 70, 60
    CPU_THRESHOLD = 0.8
    MAX_TEMP = 70
    P = 0.1
    I = 0.1
    headroom_integral = 0

    print("Starting userspace ondemand with thermal limit on clusters {}.".format(cluster))
    setUserSpace(clusters=cluster)   
    avail_freqs = getAvailFreqs(cluster)
    print('avail freq:', avail_freqs)
    setClusterFreq(4, min(avail_freqs))
    print('current freq:', getClusterFreq(cluster))
    
    while True:
        last_time = time.time()
        usages = getCpuLoad() # usage for each core
        max_core_temp = max(getTemps())
        headroom = MAX_TEMP - max_core_temp
        max_allowed = 0
        # get current frequency's index in list of available frequencies
        curr_freq = getClusterFreq(cluster)
        idx = avail_freqs.index(curr_freq)
        if headroom<=0:
            try:
                max_allowed = avail_freqs[idx-1]
                headroom_integral = 0
            except:
                pass
        else:
            if max(usages) < CPU_THRESHOLD:
                try:
                    steps = math.floor(headroom*P + headroom_integral*I)
                    max_allowed = avail_freqs[idx+steps]
                    headroom_integral = headroom_integral + headroom
                except:
                    pass
            else:
                try:
                    max_allowed = avail_freqs[idx-1]
                    headroom_integral = 0
                except:
                    pass

        setClusterFreq(cluster, max_allowed)    
        
        print('current freq:', getClusterFreq(cluster))
        time.sleep( max(0, REFRESH_PERIOD - ( time.time() - last_time ) ) )
