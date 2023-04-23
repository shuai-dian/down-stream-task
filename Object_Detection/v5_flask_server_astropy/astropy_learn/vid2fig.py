from moviepy.editor import *

clip = (VideoFileClip("111.mp4"))
clip.write_gif("22.gif",fps=20)