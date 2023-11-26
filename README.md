## Sketch Video Synthesis

[Yudian Zheng](https://yudianzheng2018@gmail.com/), [Xiaodong Cun](http://vinthony.github.io/), [Menghan Xia](https://menghanxia.github.io/), [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/)

<a href='https://arxiv.org/abs/'><img src='https://img.shields.io/badge/ArXiv-2303.09535-red'></a> 
<a href='https://sketchvideo.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://www.youtube.com/watch?v=cVQERKS3Iqg'><img src='https://img.shields.io/badge/Youtube-Video-blue'></a>

<table class="center">
  <td><img src="gif/teaser.gif"></td>
</tr>
</table >

------------------
## 💡 Abstract

#### Understanding semantic intricacies and high-level concepts is essential in image sketch generation, and this challenge becomes even more formidable when applied to the domain of videos. To address this, we propose a novel optimization-based framework for sketching videos represented by the frame-wise Bézier Curves. In detail, we first propose a cross-frame stroke initialization approach to warm up the location and the width of each curve. Then, we optimize the locations of these curves by utilizing a semantic loss based on CLIP features and a newly designed consistency loss using the self-decomposed 2D atlas network. Built upon these design elements, the resulting sketch video showcases impressive visual abstraction and temporal coherence. Furthermore, by transforming a video into SVG lines through the sketching process, our method unlocks applications in sketch-based video editing and video doodling, enabled through video composition, as exemplified in the teaser.
------------------

### Getting Start
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

