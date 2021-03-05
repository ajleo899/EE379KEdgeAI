#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 23:49:12 2021

@author: dawei
"""

import time
import sysfs_paths as sysfs


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



if __name__ == "__main__":
    cluster = 4
    REFRESH_PERIOD = 0.2
    MAX_TEMP, MIN_TEMP = 70, 60
    print("Starting userspace ondemand with thermal limit on clusters {}.".format(cluster))
    setUserSpace(clusters=cluster)   
    avail_freqs = getAvailFreqs(cluster)
    print('avail freq:', avail_freqs)
    setClusterFreq(4, min(avail_freqs))
    print('current freq:', getClusterFreq(cluster))
    
    while True:
        last_time = time.time()
        if cluster == 4:
				# get the first four temp which correspond to the big cluster cores
            T = getTemps()
            curr_freq = getClusterFreq(cluster)
            idx = avail_freqs.index(curr_freq)
            if max(T) >= MAX_TEMP:
                try:
                    setClusterFreq(4, avail_freqs[idx - 1])
                except:
                    pass
            else:
                try:
                    setClusterFreq(4, avail_freqs[idx + 1])
                except:
                    pass
        print('current freq:', getClusterFreq(cluster))
        time.sleep( max(0, REFRESH_PERIOD - ( time.time() - last_time ) ) )
