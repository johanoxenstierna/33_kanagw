
'''These are dimensions for backr pic. Has a huge impact on cpu-time'''
MAP_DIMS = (1280, 720)  #(233, 141)small  # NEEDED FOR ASSERTIONS
# MAP_DIMS = (2560, 1440)  #(233, 141)small
# MAP_DIMS = (3840, 2160)  #(233, 141)small

COMPLEXITY = 0  # OBS REMEMBER SET P.FRAMES in waves_helper

FRAMES_START = 0
FRAMES_STOP = 500

FRAMES_TOT = FRAMES_STOP - FRAMES_START

A_F = 1
A_K = 0
# if COMPLEXITY == 1:
#     A_K = 1

'''Note Z is moving away from screen (numpy convention). 
Hence, increasing z means increasing rows in k0. BUT y is going up'''
NUM_X = 50  # MUST CORRESPOND SOMEHOW WITH O1 PICTURES
NUM_Z = 1  # 30  # 20 HAS IMPACT ON WAVE  80/40 takes 100min  100/50 takes 166min

O0_TO_SHOW = ['waves']
QUERY_STNS = 0