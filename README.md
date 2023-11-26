## Sketch Video Synthesis

<a href='https://arxiv.org/abs/'><img src='https://img.shields.io/badge/ArXiv-2303.09535-red'></a> 
<a href='https://sketchvideo.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>


### Gettting Start
if optimize the example, only (1)(5)

##### (1) build up the environment: 

the total training need projects of layer **neural layer atlas** and **diffvg**

```
<!-- install all and train from begining -->
sh scripts/install.sh
<!-- install diffvg(optimize the example) -->
sh scripts/install_diffvg.sh
```

##### (2) download Dataset or take your own data(less than 70 frames,and extract masks), put on the folder <data>:

```
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-Full-Resolution.zip
 
unzip DAVIS-2017-Unsupervised-trainval-Full-Resolution.zip
```

or using the examples data and extract the masks skip this step.

##### (3) process/crop the data:

```
sh scripts/process_dataset.sh
```

##### (4) build up atlas:
```
sh scripts/operate_atlas.sh <video_name>
```

##### (5)compute clipavideo:
```
sh scripts/operate_clipavideo.sh <video_name>
```

Look at arguments.txt to see more arguments
