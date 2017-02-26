## Randomized Visual Phrases for Object Search

Code for visual object search method described in the paper [Randomized Visual Phrases for Object Search](http://eeeweba.ntu.edu.sg/computervision/Research%20Papers/2012/Randomized%20Visual%20Phrases%20for%20Object%20Search.pdf), by Yuning Jiang, Jingjing Meng and Junsong Yuan (Nanyang Technological University).


## Prerequisites

1. The code is written in Python 2.7.12. Therefore you need to have Python 2 installed on your computer before you can run the code.
2. Classes and methods from OpenCV 2.12 are used in the code. OpenCV 2.12 can be downloaded from http://opencv.org/.


## Format

To compile and run _locateObj.py_ make sure the directory tree looks like the following:

```Markdown
- RVP/
    - Model/
        - kmeans
        - stoplist
        - ...
    - Groundhog day/
        - I_02001.jpg (must be 5 digits)
        - I_02002.jpg
        - ...
    - Results/
    - locateObj.py
    - featureExtractor.py
    - ...
```
## Steps to run _locateObj.py_

1. Run _locateObj.py_ and you'll see the test image.
2. Select the object you are interested with using your mouse.
Once you select the object, a green bounding box appears.
3. Press the _escape_ button on your keyboard.
4. Be patient and wait for the final result, together with the immediate results.
This process takes a few seconds, since the program shows each picture for at least 2 seconds.

Some clarifications:
- The folder named "Model" stores pre-computed bag of visual words, stop list, word frequency and kmeans object. They are all computed with part of the Groundhog data set with functions written in _featureExtractor.py_.
- You can watch the final results in the folder named "Results".


## Reference

1. Gabriella Csurka, Christopher R. Dance, Lixin Fan, Jutta Willamowski, and Cédric Bray. Visual Categorization with Bags of Keypoints. _IEEE European Conference on Computer Vision (ECCV'04)_, 2004 ([pdf](http://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/csurka-eccv-04.pdf)).
2. Yuning Jiang, Jingjing Meng, and Junsong Yuan. Randomized Visual Phrases for Object Search. _IEEE Computer Vision and Pattern Recognition (CVPR'12)_, 2012 ([pdf](http://eeeweba.ntu.edu.sg/computervision/Research%20Papers/2012/Randomized%20Visual%20Phrases%20for%20Object%20Search.pdf)).
3. J. Sivic and A. Zisserman. Efficient visual search of videos cast as text retrieval. _IEEE Trans. on Pattern Analysis and Machine Intelligence_, 2009 ([pdf](http://people.ee.duke.edu/~lcarin/Video_Search_PAMI.pdf)).
