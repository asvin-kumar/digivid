# Asvin's solutions for HW 3 of Digital Video ECE 381K

## Description
HW 3 is a programming focussed assignment in Dr. Alan Bovik's course on Digital Video at UT Austin.  
This HW focuses on motion, motion estimation algorithms and natural scene statistics.  

## Questions
See Digital_Video_HW3_updated.pdf for the questions. 
* Q1 is about motion estimation using full search.
* Q2 is about motion estimation using logarithmic search.
* Q3 is about the Fleet Jepson method. This is not a programming question.
* Q4 is about natural scene statistics.

### Q1
Q1 uses the Middlebury dataset found at https://vision.middlebury.edu/flow/data/.  
Loading this data using `q1/load_middlebuy.m`

### Q4
Q4 uses the UVG dataset found at https://ultravideo.fi/#testsequences.  
This script conmputes statistics on frame 10 from SunBath_3840x2160_50fps_420_8bit_YUV_RAW.7z.    
 
## Solutions  
Solution to questions 1, 2, 3, 4 are in folders `q1/`, `q2/`, `q3/` and `q4/`. 
Each of these folders has an .ipynb for interacting with the code and a .pdf export of the same file.
Please note: I moved these solution files from the base folder into folders for each question to clean things up. Some paths might need to be fixed in order to get the scripts working.

## Other useful repos
1. https://github.com/001honi/video-processing/tree/main/homework-1 
2. https://github.com/gautamo/BlockMatching  
3. https://github.com/KarelZhang/optical-flow-evaluation  

## Note
If you're a student taking this course in the future, please give it a try on your own!  
It really is a fun programming exercise.
