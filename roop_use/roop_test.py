import os
import sys
#先引入roop_main的绝对路径
sys.path.append('/wangjin_4_fix/dawanqufusai/unidiffusion_text_to_image-master/roop_main')


from roop_main.roop import core
#执行roop，传入三个路径，第一个是换脸对象的源文件，第二个是被换脸的目标文件，第三个是输出路径，输出的文件名是“源文件名-目标文件名.jpg”
core.run('./resources/girl1_example.jpeg','/wangjin_4_fix/dawanqufusai/unidiffusion_text_to_image-master/temp/temp_images/50-004.jpg','./roop_test')

