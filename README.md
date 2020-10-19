# SFND_Final

## Final Project : Track an Object in 3D Space

### Dependencies for Running Locally
* cmake >= 2.8  
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
* OpenCV >= 4.1
  * This must be compiled from source using the -D OPENCV_ENABLE_NONFREE=ON cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found here
* gcc/g++ >= 5.4
***
### Basic Build Instructions
1. Clone this repo.
2. Make a build directory in the top level project directory: <code> mkdir build && cd build </code>
3. Compile: <code> cmake .. && make </code>
4. Run it: <code> ./3D_object_tracking. </code>
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
***
### FP.2 Compute Lidar-based TTC
Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame. Code is functional and returns the specified output. Also, the code is able to deal with outlier Lidar points in a statistically robust way to avoid severe estimation errors.

    void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
    {
        // auxiliary variables
        double dT = 1/frameRate;        // time between two measurements in seconds
        double laneWidth = 4.0; // assumed width of the ego lane
        double removeFrontRatio = 0.2;
        double removeEndRatio = 0.4;

        double meanXPrev = 0.0, meanXCurr = 0.0;
        int xPrevSize = 0, xCurrSize = 0;
        std::vector<double> xPrev, xCurr;
        for (auto itr = lidarPointsPrev.begin(); itr != lidarPointsPrev.end(); ++itr)
        {
            if (abs(itr->y) > laneWidth/2.0) continue;
            xPrev.push_back(itr->x);
        }
        std::sort(xPrev.begin(), xPrev.end());
        int sizeVec = xPrev.size();
        auto xPrevStart = xPrev.begin() + removeFrontRatio*sizeVec;
        auto xPrevEnd = xPrev.end() - removeEndRatio*sizeVec;
        for(auto itr = xPrevStart; itr != xPrevEnd; ++itr)
        {
            ++xPrevSize;
            meanXPrev += *itr;
        }
        meanXPrev /= xPrevSize;

        for (auto itr = lidarPointsCurr.begin(); itr != lidarPointsCurr.end(); ++itr)
        {   
            if (abs(itr->y) > laneWidth/2.0) continue;
            xCurr.push_back(itr->x);

        }
        std::sort(xCurr.begin(), xCurr.end());
        sizeVec = xCurr.size();
        auto xCurrStart = xCurr.begin() + removeFrontRatio*sizeVec;
        auto xCurrEnd = xCurr.end() - removeEndRatio*sizeVec;
        for(auto itr = xCurrStart; itr != xCurrEnd; ++itr)
        {
            ++xCurrSize;
            meanXCurr += *itr;
        }
        meanXCurr /= xCurrSize;

        // compute TTC from both measurements
        TTC = meanXCurr * dT / (meanXPrev - meanXCurr);    
    }
***
### FP.3 Associate Keypoint Correspondences with Bounding Boxes
Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box. Code performs as described and adds the keypoint correspondences to the "kptMatches" property of the respective bounding boxes. Also, outlier matches have been removed based on the euclidean distance between them in relation to all the matches in the bounding box.
 
    // (prev, curr) = (source, ref) = (query, train)
    // associate a given bounding box with the keypoints it contains
    void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
    {
        //**** method  = remove 20% of the high value euclidean distances among keypoints ****//
        double outlierFilterRatio = 0.2; // remove 20% of the high value euclidean distances among keypoints
        std::multiset<double> euclideanDistSets;

        // calculate euclidean distance b/w keypoints and insert it to multiset(search tree)
        for(auto itr = kptMatches.begin(); itr != kptMatches.end(); ++itr)
        {
            auto currKpt = kptsCurr[itr->trainIdx];
            if(boundingBox.roi.contains(currKpt.pt))
            {
                auto prevKpt = kptsPrev[itr->queryIdx];
                double l2Norm = cv::norm(currKpt.pt - prevKpt.pt);
                euclideanDistSets.insert(l2Norm);
            }
        }

        int numOffset = (int)(std::round(outlierFilterRatio * euclideanDistSets.size()));
        auto rItr = euclideanDistSets.rend(); // reverse iterator( rItr++ = <-  //  rItr-- = ->)
        std::advance(rItr, numOffset);
        double filterThreshold = *rItr;

        std::multiset<double> euclideanDistSets2;
        for(auto itr = kptMatches.begin(); itr != kptMatches.end(); ++itr)
        {
            auto currKpt = kptsCurr[itr->trainIdx];
            if(boundingBox.roi.contains(currKpt.pt))
            {
                auto prevKpt = kptsPrev[itr->queryIdx];
                double l2Norm = cv::norm(currKpt.pt - prevKpt.pt);
                if (l2Norm <= filterThreshold)
                {
                    boundingBox.keypoints.push_back(currKpt);
                    boundingBox.kptMatches.push_back(*itr);
                }
            }
        }
        //**** method  = remove 20% of the high value euclidean distances among keypoints ****//    
    }
***
### FP.4 Compute Camera-based TTC
Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame. Code is functional and returns the specified output. Also, the code is able to deal with outlier correspondences in a statistically robust way to avoid severe estimation errors.
    
    
    // Compute time-to-collision (TTC) based on keypoint correspondences in successive images
    void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                          std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
    {
        // compute distance ratios between all matched keypoints
        vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
        for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
        { // outer kpt. loop

            // get current keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpOuterCurr = kptsCurr[it1->trainIdx];
            cv::KeyPoint kpOuterPrev = kptsPrev[it1->queryIdx];

            for (auto it2 = it1 + 1; it2 != kptMatches.end(); ++it2)
            // for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
            { // inner kpt.-loop

                double minDist = 100.0; // min. required distance

                // get next keypoint and its matched partner in the prev. frame
                cv::KeyPoint kpInnerCurr = kptsCurr[it2->trainIdx];
                cv::KeyPoint kpInnerPrev = kptsPrev[it2->queryIdx];

                // compute distances and distance ratios
                double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
                double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

                if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
                { // avoid division by zero

                    double distRatio = distCurr / distPrev;
                    distRatios.push_back(distRatio);
                }
            } // eof inner loop over all matched kpts
        }     // eof outer loop over all matched kpts

        // only continue if list of distance ratios is not empty
        if (distRatios.size() == 0)
        {
            TTC = NAN;
            return;
        }

        // find median distance
        double medianDistRatio;
        std::sort(distRatios.begin(), distRatios.end()); // ascending sorting
        if(distRatios.size() % 2 == 1)
        {
            auto itr = distRatios.begin() + (distRatios.size()-1)/2;
            medianDistRatio = *itr;
        }
        else
        {
            auto itr = distRatios.begin() + distRatios.size()/2 - 1;
            auto nextItr = itr + 1;
            medianDistRatio = (*itr + *nextItr)/2;
        }

        double dT = 1 / frameRate;
        TTC = -dT / (1 - medianDistRatio);
    }
***
### FP.5 Performance Evaluation 1
Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened. Several examples (2-3) have been identified and described in detail. The assertion that the TTC is off has been based on manually estimating the distance to the rear of the preceding vehicle from a top view perspective of the Lidar points

One example of TTC of Lidar in case of <Detector, Descriptor> = <Shi-Tomashi, FREAK> is as follows.
As you can see bold time in the columns of TTC of Lidar of below table, a sudden increase in TTC has occured. In the formula for calculating the TTC of Lidar, 
**TTC = meanXCurr * dT / (meanXPrev - meanXCurr)**, a small value change in x in a continuous frame, **(meanXPrev - meanXCurr)**, caused a sudden increase in TTC.
This is because distance to preceding car from previous data frame, **meanXPrev**, might have been influenced by some point cloud outliers, which result in shorter distance than the actual tailgate. Some outliers can be removed by sorting the x-values in ascending order, removing the front part and back part and calculating the mean for the rest(This can be found in the code in **FP.2**) . By doing so, some way off can be removed.

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
| SHITOMASI|         FREAK|          0000|           nans|           nans|
| SHITOMASI|         FREAK|          0001|       12.6940s|       13.8195s|
| SHITOMASI|         FREAK|          0002|       12.0310s|       13.3121s|
| SHITOMASI|         FREAK|          0003|       **17.3639s**|       12.8840s|
| SHITOMASI|         FREAK|          0004|       **16.0990s**|       12.7464s|
| SHITOMASI|         FREAK|          0005|       13.3004s|       13.4423s|
| SHITOMASI|         FREAK|          0006|       13.1047s|       14.0207s|
| SHITOMASI|         FREAK|          0007|       12.1979s|       17.7217s|
| SHITOMASI|         FREAK|          0008|       13.3858s|       12.9116s|
| SHITOMASI|         FREAK|          0009|       13.4396s|       12.2294s|
| SHITOMASI|         FREAK|          0010|       12.1049s|       13.5328s|
| SHITOMASI|         FREAK|          0011|       12.2210s|       11.6809s|
| SHITOMASI|         FREAK|          0012|        9.2248s|       11.5191s|
| SHITOMASI|         FREAK|          0013|        9.5029s|       13.0041s|
| SHITOMASI|         FREAK|          0014|        9.5302s|       11.6442s|
| SHITOMASI|         FREAK|          0015|        8.1272s|       11.0141s|
| SHITOMASI|         FREAK|          0016|        9.9349s|       11.6037s|
| SHITOMASI|         FREAK|          0017|       **10.5262s**|       11.0073s|
| SHITOMASI|         FREAK|          0018|        8.4862s|        7.8567s|

