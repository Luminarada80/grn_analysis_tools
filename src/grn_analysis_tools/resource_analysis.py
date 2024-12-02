import os
import re

def parse_wall_clock_time(line):
    # Extract the time part after the last mention of 'time'
    time_part = line.split("):")[-1].strip()
    
    # Split the time part by colons to get hours, minutes, and seconds if present
    time_parts = time_part.split(":")
    
    # Initialize hours, minutes, seconds to 0
    hours, minutes, seconds = 0, 0, 0
    
    # Clean up and parse each part
    if len(time_parts) == 3:  # h:mm:ss or h:mm:ss.ss
        hours = float(re.sub(r'[^\d.]', '', time_parts[0]))  # Remove non-numeric characters
        minutes = float(re.sub(r'[^\d.]', '', time_parts[1]))
        seconds = float(re.sub(r'[^\d.]', '', time_parts[2]))

    elif len(time_parts) == 2:  # m:ss or m:ss.ss
        minutes = float(re.sub(r'[^\d.]', '', time_parts[0]))
        seconds = float(re.sub(r'[^\d.]', '', time_parts[1]))

    # Calculate total time in seconds
    total_seconds = seconds + (minutes * 60) + (hours * 3600)
    hours = total_seconds * 0.0002778
    
    return hours

def parse_time_module_output(log_dir: str, sample_list: list):
    sample_resource_dict = {}

    for sample_log_dir in os.listdir(log_dir):
        # Find each sample in the LOGS directory
        if sample_log_dir in sample_list:
            
            # Initialize pipeline_step_dict once per sample_log_dir
            sample_resource_dict[sample_log_dir] = {}
            
            # Find each step log file for the sample
            for file in os.listdir(f'{log_dir}/{sample_log_dir}'):
                
                if file.endswith(".log"):
                    pipeline_step = file.split(".")[0]
                    sample_resource_dict[sample_log_dir][pipeline_step] = {
                        "user_time": 0,
                        "system_time": 0,
                        "percent_cpu": 0,
                        "wall_clock_time": 0,
                        "max_ram": 0
                    }

                    # Extract each relevant resource statistic for the sample step and save it in a dictionary
                    with open(f'{log_dir}/{sample_log_dir}/{file}', 'r') as log_file:
                        for line in log_file:
                            if 'User time' in line:
                                sample_resource_dict[sample_log_dir][pipeline_step]["user_time"] = float(line.split(":")[-1])
                            if 'System time' in line:
                                sample_resource_dict[sample_log_dir][pipeline_step]["system_time"] = float(line.split(":")[-1])
                            if 'Percent of CPU' in line:
                                sample_resource_dict[sample_log_dir][pipeline_step]["percent_cpu"] = float(line.split(":")[-1].split("%")[-2])
                            if 'wall clock' in line:
                                sample_resource_dict[sample_log_dir][pipeline_step]["wall_clock_time"] = parse_wall_clock_time(line)
                            if 'Maximum resident set size' in line:
                                kb_per_gb = 1048576
                                sample_resource_dict[sample_log_dir][pipeline_step]["max_ram"] = (float(line.split(":")[-1]) / kb_per_gb)

    return sample_resource_dict

