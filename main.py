"""
Matplotlib animation of projectiles, waves and clouds

Only create 1 stns and Y map (or whatever TF its called).

YT still needed!

-*Foam f twist after h (probably just shift)

-zorder.
-increase c
-rotate_around
-scale
-tune foam in 1 and 2
-MORERandom b foam flying into air.
"""

import matplotlib.pyplot as plt
from sys import platform
if platform == 'win32':
    plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

import numpy as np
import random
random.seed(7)  # ONLY HERE
np.random.seed(7)  # ONLY HERE
import time
import pickle
import matplotlib.animation as animation
from src import gen_objects
from src.ani_helpers import *
import P as P

WRITE = 0
FPS = 20

Writer = animation.writers['ffmpeg']
writer = Writer(fps=FPS, bitrate=3600)  #, extra_args=['-vcodec', 'h264_nvenc'])

fig, ax_b = plt.subplots(figsize=(6, 3), frameon=False)
fig.subplots_adjust(bottom=0)
fig.subplots_adjust(top=1)
fig.subplots_adjust(right=1)
fig.subplots_adjust(left=0)

axs0 = []
axs1 = []

g = gen_objects.GenObjects()
g.gen_backgr(ax_b, axs0, axs1)

o0 = g.gen_o0()
g.gen_O1(o0)

# with open('./O0', 'wb') as f:
#     pickle.dump(O0, f)

plt.gca().invert_yaxis()

brkpoint = ''
'''VIEWER ==========================================='''
print("Done prep")

def init():
    return axs0 #+ axs1


def animate(i):

    prints = "i: " + str(i) + "  len_axs0: " + str(len(axs0)) + "  len_axs1: " + str(len(axs1))

        # for o1_id, o1 in o0.O1.items():  # this loop is super fast
        #
        #     if i in o1.gi['init_frames']:
        #
        #         if o1.drawn == 0:
        #             prints += "  adding f"
        #             exceeds_frame_max, how_many = o1.check_frame_max(i, o1.gi['frames_tot'])
        #             if exceeds_frame_max == True:
        #                 print("EXCEEDS MAX. This means objects at end of animation will go faster. \n")
        #                 o1.gi['frames_tot'] = how_many
        #
        #             # o1.dyn_gen()
        #             o1.drawn = 1  # this variable can serve multiple purposes (see below, and in set_clock)
        #             o1.set_frame_ss(i, o1.gi['frames_tot'])  # uses AbstractSSS
        #
        #             ''' EVIL BUG HERE. An o1 cannot be allowed to init new O2 children if old children
        #             are still being drawn!!! THIS MEANS o1 FRAMES_TOT MUST > O2 FRAMES TOT
        #             UPDATE: Try releasing o1 once max frame stop of its sps reached. '''
        #
        #         else:
        #             prints += "  no free o1"

    for o1_id, o1 in o0.O1.items():  # this is where most of the CPU time goes

        if o1.drawn != 0:  # Its not just boolean!
            o1.set_clock(i)
            drawBool, index_removed = o1.ani_update_step(ax_b, axs0, axs1, object_type='o1')

            if drawBool == 0:  # dont draw
                continue
            elif drawBool == 1:  # continue drawing
                set_O1(o1, ax_b, axs0)
            elif drawBool == 2:  # remove
                # prints += "  removing o1"
                decrement_all_index_axs0(index_removed, o0)

    print(prints)

    return axs0  # + axs1  # 0 for dynamic objects, 1 for background


sec_vid = ((P.FRAMES_STOP - P.FRAMES_START) / FPS)
min_vid = ((P.FRAMES_STOP - P.FRAMES_START) / FPS) / 60
print("len of vid: " + str(sec_vid) + " s" + "    " + str(min_vid) + " min")

start_t = time.time()
ani = animation.FuncAnimation(fig, animate, frames=range(P.FRAMES_START, P.FRAMES_STOP),
                              blit=True, interval=1, init_func=init,
                              repeat=True)  # interval only affects live ani. blitting seems to make it crash

if WRITE == 0:
    plt.show()
else:
    ani.save('./vids/vid_' + str(WRITE) + '_' + str(P.NUM_X) + '_' + str(P.NUM_Z) + '.mp4', writer=writer)

    # ani.save('./vids/vid_' + str(WRITE) + '.mov',
    #          codec="png",
    #          dpi=100,
    #          fps=40,
    #          bitrate=3600)

    # THIS ONE!!!
    # ani.save('./vids/vid_' + str(WRITE) + '.mov',
    #          codec="png",
    #          dpi=100,
    #          fps=FPS,
    #          bitrate=3600,
    #          savefig_kwargs={"transparent": True, "facecolor": "none"})

tot_time = round((time.time() - start_t) / 60, 4)
print("minutes to make animation: " + str(tot_time) + " |  min_gen/min_vid: " + str(tot_time / min_vid))  #
