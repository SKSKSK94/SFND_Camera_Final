# SFND_Final

## Final Project : Track an Object in 3D Space

### Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions][cmakelink]
  [cmakelink]: http://gnuwin32.sourceforge.net/packages/make.htm
  
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: install Xcode command line tools to get make
  * Windows: Click here for installation instructions
* OpenCV >= 4.1
  * This must be compiled from source using the -D OPENCV_ENABLE_NONFREE=ON cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found here
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - install Xcode command line tools
  * Windows: recommend using MinGW
***
### Basic Build Instructions
1. Clone this repo.
2. Make a build directory in the top level project directory: mkdir build && cd build
3. Compile: cmake .. && make
4. Run it: ./3D_object_tracking.
***
### FP.1 Match 3D Objects
Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences. Code is functional and returns the specified output, where each bounding box is assigned the match candidate with the highest number of occurrences.
  
    void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
    {
      // initializing vector to save number of matches
      int currFrameBoxSizes = currFrame.boundingBoxes.size(), prevFrameBoxSizes = prevFrame.boundingBoxes.size();
      std::vector<std::vector<int>> numMatches; // (curr, prev)
      std::vector<int> temp(prevFrame.boundingBoxes.size(), 0);
      for(int idx = 0; idx < currFrameBoxSizes; ++idx)
      {        
          numMatches.push_back(temp);        
      }

      // find which bounding box match keypoints are included and count the numbers
      for (auto itr = matches.begin(); itr != matches.end(); ++itr)
      {
          int boxIdx1 = 0, currBoxIdx = -1; // find which bounding box match keypoint is included 
          for (auto itr2 = currFrame.boundingBoxes.begin(); itr2 != currFrame.boundingBoxes.end(); ++itr2, ++boxIdx1)
          {
              if(itr2->roi.contains(currFrame.keypoints[itr->trainIdx].pt))
              {
                  currBoxIdx = boxIdx1;
              }
          }
          if(currBoxIdx < 0) continue;

          int boxIdx2 = 0, prevBoxIdx = -1; // find which bounding box match keypoint is included 
          for (auto itr2 = prevFrame.boundingBoxes.begin(); itr2 != prevFrame.boundingBoxes.end(); ++itr2, ++boxIdx2)
          {
              if(itr2->roi.contains(prevFrame.keypoints[itr->queryIdx].pt))
              {
                  prevBoxIdx = boxIdx2;
              }
          }
          if(prevBoxIdx < 0) continue;

          numMatches[currBoxIdx][prevBoxIdx]++;
      }

      // finding best combination of bounding boxes b/w current and previous frame based on number of assigned match keypoints
      std::vector<int> numMatchesSave(currFrame.boundingBoxes.size(), 0);
      int outterIdx = 0;
      for(auto itr = numMatches.begin(); itr != numMatches.end(); ++itr, ++outterIdx)
      {
          int maxValue = -1, maxIdx = -1, innerIdx = 0;
          for(auto elem : *itr)
          {
              if(maxValue < elem){
                  maxValue = elem;
                  maxIdx = innerIdx;
              }
              ++innerIdx;
          }
          numMatchesSave[outterIdx] = maxIdx;
      }
      for(int prevIdx = 0; prevIdx < prevFrameBoxSizes; ++prevIdx)
      {
          int maxValue = -1;
          for(int currIdx = 0; currIdx < currFrameBoxSizes; ++currIdx)
          {
              if (prevIdx == numMatchesSave[currIdx])
              {
                  if (numMatches[currIdx][prevIdx] > maxValue)
                  {
                      maxValue = numMatches[currIdx][prevIdx];
                      bbBestMatches[prevIdx] = currIdx;
                  }
              }
          }
       }
    }
  
