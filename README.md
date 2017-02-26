## Randomized Visual Phrases for Object Search

Code for visual object search method described in the paper [Randomized Visual Phrases for Object Search](http://eeeweba.ntu.edu.sg/computervision/Research%20Papers/2012/Randomized%20Visual%20Phrases%20for%20Object%20Search.pdf), by Yuning Jiang, Jingjing Meng and Junsong Yuan (Nanyang Technological University).


## Prerequisites
1. The code is written in Python 2.7.12. Therefore you need to have Python 2 installed on your computer before you can run the code.
2. Classes and methods from OpenCV 2.12 are used in the code. OpenCV 2.12 can be downloaded from http://opencv.org/.


## Format

To compile and run _locateObj.py_ make sure the directory tree looks like the following:

```Markdown
- RVP/
    - model/
        - kmeans
        - stoplist
        - ...
    - images//
        - I_00061.jpg (must be 5 digits)
        - I_00062.jpg
    - locateObj.py
    - featureExtractor.py
    - ...
```

## Reference

1. Gabriella Csurka, Christopher R. Dance, Lixin Fan, Jutta Willamowski, and Cédric Bray. Visual Categorization with Bags of Keypoints. _IEEE European Conference on Computer Vision (ECCV'04)_, 2004 ([pdf](http://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/csurka-eccv-04.pdf)).
2. Yuning Jiang, Jingjing Meng, and Junsong Yuan. Randomized Visual Phrases for Object Search. _IEEE Computer Vision and Pattern Recognition (CVPR'12)_, 2012 ([pdf](http://eeeweba.ntu.edu.sg/computervision/Research%20Papers/2012/Randomized%20Visual%20Phrases%20for%20Object%20Search.pdf)).
3. J. Sivic and A. Zisserman. Efficient visual search of videos cast as text retrieval. _IEEE Trans. on Pattern Analysis and Machine Intelligence_, 2009 ([pdf](http://people.ee.duke.edu/~lcarin/Video_Search_PAMI.pdf)).
You can use the [editor on GitHub](https://github.com/weilheim/Random-Visual-Phrase/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
