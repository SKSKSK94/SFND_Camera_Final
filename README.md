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

![LidarTTC-1](https://user-images.githubusercontent.com/73100569/96540485-d9213480-12d8-11eb-8542-feada2263b36.png)
![LidarTTC-2](https://user-images.githubusercontent.com/73100569/96540484-d9213480-12d8-11eb-8000-57d7f2d0b645.png)

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

![CameraTTC-1](https://user-images.githubusercontent.com/73100569/96540483-d8889e00-12d8-11eb-80de-f33a1903eb2b.png)
![CameraTTC-2](https://user-images.githubusercontent.com/73100569/96540482-d7f00780-12d8-11eb-9261-8e8f322bea40.png)
![CameraTTC-3](https://user-images.githubusercontent.com/73100569/96540480-d6264400-12d8-11eb-9cf9-86135d55a17e.jpg)

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
As you can see bold time in the column of TTC of Lidar of below table, a sudden increase in TTC has occured. In the formula for calculating the TTC of Lidar, 
**TTC = meanXCurr * dT / (meanXPrev - meanXCurr)**, a small value change in x in a continuous frame, **(meanXPrev - meanXCurr)**, caused a sudden increase in TTC.
This is because distance to preceding car from previous data frame, **meanXPrev**, might have been influenced by some point cloud outliers, which result in shorter distance than the actual tailgate. Some of these outliers can be removed by sorting the x-values in ascending order, removing the front part and back part and calculating the mean for the rest(This can be found in the code in **FP.2**). By doing so, some way off can be removed but the rest of the way off(**bold type time in the TTC of Lidar column**) seems to be because the overall point cloud datas are skewed to one side considering top view image of point cloud within interesting bounding box.

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

***
### FP.6 Performance Evaluation 2
Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons. All detector / descriptor combinations implemented in previous chapters have been compared with regard to the TTC estimate on a frame-by-frame basis. To facilitate comparison, a spreadsheet and graph should be used to represent the different TTCs.

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|    HARRIS|         BRISK|          0000|           nans|           nans|
|    HARRIS|         BRISK|          0001|       12.6940s|        9.6030s|
|    HARRIS|         BRISK|          0002|       12.0310s|       12.5101s|
|    HARRIS|         BRISK|          0003|       17.3639s|       12.0789s|
|    HARRIS|         BRISK|          0004|       16.0990s|       25.5028s|
|    HARRIS|         BRISK|          0005|       13.3004s|       13.4692s|
|    HARRIS|         BRISK|          0006|       13.1047s|       17.5125s|
|    HARRIS|         BRISK|          0007|       12.1979s|       31.6313s|
|    HARRIS|         BRISK|          0008|       13.3858s|          -infs|
|    HARRIS|         BRISK|          0009|       13.4396s|       10.1682s|
|    HARRIS|         BRISK|          0010|       12.1049s|       11.1726s|
|    HARRIS|         BRISK|          0011|       12.2210s|           nans|
|    HARRIS|         BRISK|          0012|        9.2248s|           nans|
|    HARRIS|         BRISK|          0013|        9.5029s|           nans|
|    HARRIS|         BRISK|          0014|        9.5302s|           nans|
|    HARRIS|         BRISK|          0015|        8.1272s|           nans|
|    HARRIS|         BRISK|          0016|        9.9349s|      -244.9500s|
|    HARRIS|         BRISK|          0017|       10.5262s|           nans|
|    HARRIS|         BRISK|          0018|        8.4862s|           nans|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|    HARRIS|         BRIEF|          0000|           nans|           nans|
|    HARRIS|         BRIEF|          0001|       12.6940s|       56.0386s|
|    HARRIS|         BRIEF|          0002|       12.0310s|       12.5101s|
|    HARRIS|         BRIEF|          0003|       17.3639s|       12.7918s|
|    HARRIS|         BRIEF|          0004|       16.0990s|       40.2406s|
|    HARRIS|         BRIEF|          0005|       13.3004s|          -infs|
|    HARRIS|         BRIEF|          0006|       13.1047s|       17.5125s|
|    HARRIS|         BRIEF|          0007|       12.1979s|          -infs|
|    HARRIS|         BRIEF|          0008|       13.3858s|          -infs|
|    HARRIS|         BRIEF|          0009|       13.4396s|       36.6298s|
|    HARRIS|         BRIEF|          0010|       12.1049s|       11.4720s|
|    HARRIS|         BRIEF|          0011|       12.2210s|           nans|
|    HARRIS|         BRIEF|          0012|        9.2248s|           nans|
|    HARRIS|         BRIEF|          0013|        9.5029s|           nans|
|    HARRIS|         BRIEF|          0014|        9.5302s|           nans|
|    HARRIS|         BRIEF|          0015|        8.1272s|           nans|
|    HARRIS|         BRIEF|          0016|        9.9349s|      -244.9500s|
|    HARRIS|         BRIEF|          0017|       10.5262s|           nans|
|    HARRIS|         BRIEF|          0018|        8.4862s|           nans|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|    HARRIS|           ORB|          0000|           nans|           nans|
|    HARRIS|           ORB|          0001|       12.6940s|       56.0386s|
|    HARRIS|           ORB|          0002|       12.0310s|       12.5101s|
|    HARRIS|           ORB|          0003|       17.3639s|       12.7918s|
|    HARRIS|           ORB|          0004|       16.0990s|       40.2406s|
|    HARRIS|           ORB|          0005|       13.3004s|          -infs|
|    HARRIS|           ORB|          0006|       13.1047s|       17.5125s|
|    HARRIS|           ORB|          0007|       12.1979s|       14.3741s|
|    HARRIS|           ORB|          0008|       13.3858s|          -infs|
|    HARRIS|           ORB|          0009|       13.4396s|       36.6298s|
|    HARRIS|           ORB|          0010|       12.1049s|       11.4720s|
|    HARRIS|           ORB|          0011|       12.2210s|           nans|
|    HARRIS|           ORB|          0012|        9.2248s|           nans|
|    HARRIS|           ORB|          0013|        9.5029s|           nans|
|    HARRIS|           ORB|          0014|        9.5302s|           nans|
|    HARRIS|           ORB|          0015|        8.1272s|           nans|
|    HARRIS|           ORB|          0016|        9.9349s|      -244.9500s|
|    HARRIS|           ORB|          0017|       10.5262s|           nans|
|    HARRIS|           ORB|          0018|        8.4862s|           nans|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|    HARRIS|         FREAK|          0000|           nans|           nans|
|    HARRIS|         FREAK|          0001|       12.6940s|       56.0386s|
|    HARRIS|         FREAK|          0002|       12.0310s|       12.5101s|
|    HARRIS|         FREAK|          0003|       17.3639s|       12.0789s|
|    HARRIS|         FREAK|          0004|       16.0990s|       40.2406s|
|    HARRIS|         FREAK|          0005|       13.3004s|       13.4692s|
|    HARRIS|         FREAK|          0006|       13.1047s|       30.2418s|
|    HARRIS|         FREAK|          0007|       12.1979s|       12.4733s|
|    HARRIS|         FREAK|          0008|       13.3858s|          -infs|
|    HARRIS|         FREAK|          0009|       13.4396s|       11.4287s|
|    HARRIS|         FREAK|          0010|       12.1049s|       11.4720s|
|    HARRIS|         FREAK|          0011|       12.2210s|           nans|
|    HARRIS|         FREAK|          0012|        9.2248s|           nans|
|    HARRIS|         FREAK|          0013|        9.5029s|           nans|
|    HARRIS|         FREAK|          0014|        9.5302s|           nans|
|    HARRIS|         FREAK|          0015|        8.1272s|           nans|
|    HARRIS|         FREAK|          0016|        9.9349s|       -8.5509s|
|    HARRIS|         FREAK|          0017|       10.5262s|           nans|
|    HARRIS|         FREAK|          0018|        8.4862s|           nans|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|    HARRIS|          SIFT|          0000|           nans|           nans|
|    HARRIS|          SIFT|          0001|       12.6940s|       56.0386s|
|    HARRIS|          SIFT|          0002|       12.0310s|       12.5101s|
|    HARRIS|          SIFT|          0003|       17.3639s|       12.8630s|
|    HARRIS|          SIFT|          0004|       16.0990s|       40.2406s|
|    HARRIS|          SIFT|          0005|       13.3004s|          -infs|
|    HARRIS|          SIFT|          0006|       13.1047s|       17.5125s|
|    HARRIS|          SIFT|          0007|       12.1979s|       14.3741s|
|    HARRIS|          SIFT|          0008|       13.3858s|          -infs|
|    HARRIS|          SIFT|          0009|       13.4396s|       36.6298s|
|    HARRIS|          SIFT|          0010|       12.1049s|       11.4720s|
|    HARRIS|          SIFT|          0011|       12.2210s|           nans|
|    HARRIS|          SIFT|          0012|        9.2248s|           nans|
|    HARRIS|          SIFT|          0013|        9.5029s|           nans|
|    HARRIS|          SIFT|          0014|        9.5302s|           nans|
|    HARRIS|          SIFT|          0015|        8.1272s|           nans|
|    HARRIS|          SIFT|          0016|        9.9349s|      -244.9500s|
|    HARRIS|          SIFT|          0017|       10.5262s|           nans|
|    HARRIS|          SIFT|          0018|        8.4862s|           nans|

-----------------------------------------------------------------------------

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
| SHITOMASI|         BRISK|          0000|           nans|           nans|
| SHITOMASI|         BRISK|          0001|       12.6940s|       13.8983s|
| SHITOMASI|         BRISK|          0002|       12.0310s|       12.9008s|
| SHITOMASI|         BRISK|          0003|       17.3639s|       13.0941s|
| SHITOMASI|         BRISK|          0004|       16.0990s|       12.8070s|
| SHITOMASI|         BRISK|          0005|       13.3004s|       13.5029s|
| SHITOMASI|         BRISK|          0006|       13.1047s|       14.5050s|
| SHITOMASI|         BRISK|          0007|       12.1979s|       17.3368s|
| SHITOMASI|         BRISK|          0008|       13.3858s|       13.4822s|
| SHITOMASI|         BRISK|          0009|       13.4396s|       12.3665s|
| SHITOMASI|         BRISK|          0010|       12.1049s|       14.7902s|
| SHITOMASI|         BRISK|          0011|       12.2210s|       11.7727s|
| SHITOMASI|         BRISK|          0012|        9.2248s|       11.8924s|
| SHITOMASI|         BRISK|          0013|        9.5029s|       12.4032s|
| SHITOMASI|         BRISK|          0014|        9.5302s|       11.6245s|
| SHITOMASI|         BRISK|          0015|        8.1272s|       12.2008s|
| SHITOMASI|         BRISK|          0016|        9.9349s|       11.0562s|
| SHITOMASI|         BRISK|          0017|       10.5262s|       10.8961s|
| SHITOMASI|         BRISK|          0018|        8.4862s|        7.4844s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
| SHITOMASI|         BRIEF|          0000|           nans|           nans|
| SHITOMASI|         BRIEF|          0001|       12.6940s|       14.6343s|
| SHITOMASI|         BRIEF|          0002|       12.0310s|       13.0553s|
| SHITOMASI|         BRIEF|          0003|       17.3639s|       14.8637s|
| SHITOMASI|         BRIEF|          0004|       16.0990s|       13.5840s|
| SHITOMASI|         BRIEF|          0005|       13.3004s|       14.7159s|
| SHITOMASI|         BRIEF|          0006|       13.1047s|       14.1989s|
| SHITOMASI|         BRIEF|          0007|       12.1979s|       17.3469s|
| SHITOMASI|         BRIEF|          0008|       13.3858s|       13.3620s|
| SHITOMASI|         BRIEF|          0009|       13.4396s|       12.1524s|
| SHITOMASI|         BRIEF|          0010|       12.1049s|       13.9940s|
| SHITOMASI|         BRIEF|          0011|       12.2210s|       12.2139s|
| SHITOMASI|         BRIEF|          0012|        9.2248s|       12.2462s|
| SHITOMASI|         BRIEF|          0013|        9.5029s|       11.8895s|
| SHITOMASI|         BRIEF|          0014|        9.5302s|       12.2951s|
| SHITOMASI|         BRIEF|          0015|        8.1272s|       13.4113s|
| SHITOMASI|         BRIEF|          0016|        9.9349s|       12.7817s|
| SHITOMASI|         BRIEF|          0017|       10.5262s|       11.8612s|
| SHITOMASI|         BRIEF|          0018|        8.4862s|        8.3820s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
| SHITOMASI|           ORB|          0000|           nans|           nans|
| SHITOMASI|           ORB|          0001|       12.6940s|       14.3373s|
| SHITOMASI|           ORB|          0002|       12.0310s|       13.3226s|
| SHITOMASI|           ORB|          0003|       17.3639s|       14.1781s|
| SHITOMASI|           ORB|          0004|       16.0990s|       12.3351s|
| SHITOMASI|           ORB|          0005|       13.3004s|       14.1356s|
| SHITOMASI|           ORB|          0006|       13.1047s|       13.9533s|
| SHITOMASI|           ORB|          0007|       12.1979s|       16.8540s|
| SHITOMASI|           ORB|          0008|       13.3858s|       12.9575s|
| SHITOMASI|           ORB|          0009|       13.4396s|       12.3189s|
| SHITOMASI|           ORB|          0010|       12.1049s|       13.7999s|
| SHITOMASI|           ORB|          0011|       12.2210s|       12.2490s|
| SHITOMASI|           ORB|          0012|        9.2248s|       12.2870s|
| SHITOMASI|           ORB|          0013|        9.5029s|       12.5998s|
| SHITOMASI|           ORB|          0014|        9.5302s|       12.2294s|
| SHITOMASI|           ORB|          0015|        8.1272s|       11.4583s|
| SHITOMASI|           ORB|          0016|        9.9349s|       12.0362s|
| SHITOMASI|           ORB|          0017|       10.5262s|       11.4707s|
| SHITOMASI|           ORB|          0018|        8.4862s|        8.2341s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
| SHITOMASI|         FREAK|          0000|           nans|           nans|
| SHITOMASI|         FREAK|          0001|       12.6940s|       13.8195s|
| SHITOMASI|         FREAK|          0002|       12.0310s|       13.3121s|
| SHITOMASI|         FREAK|          0003|       17.3639s|       12.8840s|
| SHITOMASI|         FREAK|          0004|       16.0990s|       12.7464s|
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
| SHITOMASI|         FREAK|          0017|       10.5262s|       11.0073s|
| SHITOMASI|         FREAK|          0018|        8.4862s|        7.8567s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
| SHITOMASI|          SIFT|          0000|           nans|           nans|
| SHITOMASI|          SIFT|          0001|       12.6940s|       14.6343s|
| SHITOMASI|          SIFT|          0002|       12.0310s|       13.3226s|
| SHITOMASI|          SIFT|          0003|       17.3639s|       14.3874s|
| SHITOMASI|          SIFT|          0004|       16.0990s|       13.0851s|
| SHITOMASI|          SIFT|          0005|       13.3004s|       13.9639s|
| SHITOMASI|          SIFT|          0006|       13.1047s|       13.9909s|
| SHITOMASI|          SIFT|          0007|       12.1979s|       17.0305s|
| SHITOMASI|          SIFT|          0008|       13.3858s|       13.5459s|
| SHITOMASI|          SIFT|          0009|       13.4396s|       12.4892s|
| SHITOMASI|          SIFT|          0010|       12.1049s|       13.8777s|
| SHITOMASI|          SIFT|          0011|       12.2210s|       12.3866s|
| SHITOMASI|          SIFT|          0012|        9.2248s|       12.2870s|
| SHITOMASI|          SIFT|          0013|        9.5029s|       12.6071s|
| SHITOMASI|          SIFT|          0014|        9.5302s|       11.6442s|
| SHITOMASI|          SIFT|          0015|        8.1272s|       14.1824s|
| SHITOMASI|          SIFT|          0016|        9.9349s|       12.3278s|
| SHITOMASI|          SIFT|          0017|       10.5262s|       11.4781s|
| SHITOMASI|          SIFT|          0018|        8.4862s|        8.6423s|

-----------------------------------------------------------------------------

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|      FAST|         BRISK|          0000|           nans|           nans|
|      FAST|         BRISK|          0001|       12.6940s|       12.7349s|
|      FAST|         BRISK|          0002|       12.0310s|       11.9539s|
|      FAST|         BRISK|          0003|       17.3639s|       14.7877s|
|      FAST|         BRISK|          0004|       16.0990s|       14.1341s|
|      FAST|         BRISK|          0005|       13.3004s|       41.5667s|
|      FAST|         BRISK|          0006|       13.1047s|       14.1659s|
|      FAST|         BRISK|          0007|       12.1979s|       16.9250s|
|      FAST|         BRISK|          0008|       13.3858s|       12.1201s|
|      FAST|         BRISK|          0009|       13.4396s|       14.3463s|
|      FAST|         BRISK|          0010|       12.1049s|       12.9845s|
|      FAST|         BRISK|          0011|       12.2210s|       13.6706s|
|      FAST|         BRISK|          0012|        9.2248s|       12.6032s|
|      FAST|         BRISK|          0013|        9.5029s|       12.4660s|
|      FAST|         BRISK|          0014|        9.5302s|       12.8125s|
|      FAST|         BRISK|          0015|        8.1272s|       12.0753s|
|      FAST|         BRISK|          0016|        9.9349s|       12.6506s|
|      FAST|         BRISK|          0017|       10.5262s|       10.6880s|
|      FAST|         BRISK|          0018|        8.4862s|       13.8043s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|      FAST|         BRIEF|          0000|           nans|           nans|
|      FAST|         BRIEF|          0001|       12.6940s|       12.8228s|
|      FAST|         BRIEF|          0002|       12.0310s|       12.5159s|
|      FAST|         BRIEF|          0003|       17.3639s|       17.4453s|
|      FAST|         BRIEF|          0004|       16.0990s|       13.4961s|
|      FAST|         BRIEF|          0005|       13.3004s|       30.3494s|
|      FAST|         BRIEF|          0006|       13.1047s|       13.4083s|
|      FAST|         BRIEF|          0007|       12.1979s|       14.1140s|
|      FAST|         BRIEF|          0008|       13.3858s|       12.6147s|
|      FAST|         BRIEF|          0009|       13.4396s|       14.4561s|
|      FAST|         BRIEF|          0010|       12.1049s|       12.6461s|
|      FAST|         BRIEF|          0011|       12.2210s|       14.1554s|
|      FAST|         BRIEF|          0012|        9.2248s|       12.1206s|
|      FAST|         BRIEF|          0013|        9.5029s|       12.0299s|
|      FAST|         BRIEF|          0014|        9.5302s|       12.8223s|
|      FAST|         BRIEF|          0015|        8.1272s|       12.3427s|
|      FAST|         BRIEF|          0016|        9.9349s|       12.6845s|
|      FAST|         BRIEF|          0017|       10.5262s|       10.4941s|
|      FAST|         BRIEF|          0018|        8.4862s|       12.2107s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|      FAST|           ORB|          0000|           nans|           nans|
|      FAST|           ORB|          0001|       12.6940s|       12.5923s|
|      FAST|           ORB|          0002|       12.0310s|       12.5002s|
|      FAST|           ORB|          0003|       17.3639s|       18.3711s|
|      FAST|           ORB|          0004|       16.0990s|       13.4235s|
|      FAST|           ORB|          0005|       13.3004s|       48.4755s|
|      FAST|           ORB|          0006|       13.1047s|       13.2487s|
|      FAST|           ORB|          0007|       12.1979s|       14.5185s|
|      FAST|           ORB|          0008|       13.3858s|       12.6536s|
|      FAST|           ORB|          0009|       13.4396s|       14.4561s|
|      FAST|           ORB|          0010|       12.1049s|       12.8734s|
|      FAST|           ORB|          0011|       12.2210s|       14.6373s|
|      FAST|           ORB|          0012|        9.2248s|       12.4836s|
|      FAST|           ORB|          0013|        9.5029s|       12.7008s|
|      FAST|           ORB|          0014|        9.5302s|       12.6587s|
|      FAST|           ORB|          0015|        8.1272s|       12.0541s|
|      FAST|           ORB|          0016|        9.9349s|       12.6413s|
|      FAST|           ORB|          0017|       10.5262s|       11.2503s|
|      FAST|           ORB|          0018|        8.4862s|       14.1744s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|      FAST|         FREAK|          0000|           nans|           nans|
|      FAST|         FREAK|          0001|       12.6940s|       12.4196s|
|      FAST|         FREAK|          0002|       12.0310s|       13.8461s|
|      FAST|         FREAK|          0003|       17.3639s|       16.1950s|
|      FAST|         FREAK|          0004|       16.0990s|       14.7054s|
|      FAST|         FREAK|          0005|       13.3004s|       39.5129s|
|      FAST|         FREAK|          0006|       13.1047s|       12.5303s|
|      FAST|         FREAK|          0007|       12.1979s|       15.0592s|
|      FAST|         FREAK|          0008|       13.3858s|       12.4368s|
|      FAST|         FREAK|          0009|       13.4396s|       13.7589s|
|      FAST|         FREAK|          0010|       12.1049s|       12.4930s|
|      FAST|         FREAK|          0011|       12.2210s|       13.4364s|
|      FAST|         FREAK|          0012|        9.2248s|       12.7616s|
|      FAST|         FREAK|          0013|        9.5029s|       12.5265s|
|      FAST|         FREAK|          0014|        9.5302s|       12.6069s|
|      FAST|         FREAK|          0015|        8.1272s|       11.8620s|
|      FAST|         FREAK|          0016|        9.9349s|       13.0069s|
|      FAST|         FREAK|          0017|       10.5262s|       11.5339s|
|      FAST|         FREAK|          0018|        8.4862s|       11.7957s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|      FAST|          SIFT|          0000|           nans|           nans|
|      FAST|          SIFT|          0001|       12.6940s|       12.7505s|
|      FAST|          SIFT|          0002|       12.0310s|       11.8668s|
|      FAST|          SIFT|          0003|       17.3639s|       20.9723s|
|      FAST|          SIFT|          0004|       16.0990s|       14.3697s|
|      FAST|          SIFT|          0005|       13.3004s|       55.9611s|
|      FAST|          SIFT|          0006|       13.1047s|       13.9917s|
|      FAST|          SIFT|          0007|       12.1979s|       14.0397s|
|      FAST|          SIFT|          0008|       13.3858s|       12.3263s|
|      FAST|          SIFT|          0009|       13.4396s|       14.4348s|
|      FAST|          SIFT|          0010|       12.1049s|       12.6126s|
|      FAST|          SIFT|          0011|       12.2210s|       14.8780s|
|      FAST|          SIFT|          0012|        9.2248s|       11.5057s|
|      FAST|          SIFT|          0013|        9.5029s|       12.3379s|
|      FAST|          SIFT|          0014|        9.5302s|       12.3278s|
|      FAST|          SIFT|          0015|        8.1272s|       11.6734s|
|      FAST|          SIFT|          0016|        9.9349s|       12.3677s|
|      FAST|          SIFT|          0017|       10.5262s|       11.6657s|
|      FAST|          SIFT|          0018|        8.4862s|       11.7451s|

-----------------------------------------------------------------------------

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|     BRISK|         BRISK|          0000|           nans|           nans|
|     BRISK|         BRISK|          0001|       12.6940s|       17.5941s|
|     BRISK|         BRISK|          0002|       12.0310s|       26.8505s|
|     BRISK|         BRISK|          0003|       17.3639s|       19.2280s|
|     BRISK|         BRISK|          0004|       16.0990s|       18.1887s|
|     BRISK|         BRISK|          0005|       13.3004s|       29.0754s|
|     BRISK|         BRISK|          0006|       13.1047s|       24.3089s|
|     BRISK|         BRISK|          0007|       12.1979s|       29.4660s|
|     BRISK|         BRISK|          0008|       13.3858s|       16.5358s|
|     BRISK|         BRISK|          0009|       13.4396s|       18.9180s|
|     BRISK|         BRISK|          0010|       12.1049s|       15.9976s|
|     BRISK|         BRISK|          0011|       12.2210s|       15.0965s|
|     BRISK|         BRISK|          0012|        9.2248s|       12.7072s|
|     BRISK|         BRISK|          0013|        9.5029s|       12.2005s|
|     BRISK|         BRISK|          0014|        9.5302s|       15.9117s|
|     BRISK|         BRISK|          0015|        8.1272s|       13.9605s|
|     BRISK|         BRISK|          0016|        9.9349s|       13.4019s|
|     BRISK|         BRISK|          0017|       10.5262s|       11.5163s|
|     BRISK|         BRISK|          0018|        8.4862s|       14.3220s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|     BRISK|         BRIEF|          0000|           nans|           nans|
|     BRISK|         BRIEF|          0001|       12.6940s|       14.9055s|
|     BRISK|         BRIEF|          0002|       12.0310s|       23.0425s|
|     BRISK|         BRIEF|          0003|       17.3639s|       19.1297s|
|     BRISK|         BRIEF|          0004|       16.0990s|       22.6850s|
|     BRISK|         BRIEF|          0005|       13.3004s|       19.3827s|
|     BRISK|         BRIEF|          0006|       13.1047s|       28.8309s|
|     BRISK|         BRIEF|          0007|       12.1979s|       25.1930s|
|     BRISK|         BRIEF|          0008|       13.3858s|       19.3001s|
|     BRISK|         BRIEF|          0009|       13.4396s|       22.0864s|
|     BRISK|         BRIEF|          0010|       12.1049s|       16.2644s|
|     BRISK|         BRIEF|          0011|       12.2210s|       14.9321s|
|     BRISK|         BRIEF|          0012|        9.2248s|       16.8288s|
|     BRISK|         BRIEF|          0013|        9.5029s|       13.0474s|
|     BRISK|         BRIEF|          0014|        9.5302s|       16.1756s|
|     BRISK|         BRIEF|          0015|        8.1272s|       13.7789s|
|     BRISK|         BRIEF|          0016|        9.9349s|       14.0256s|
|     BRISK|         BRIEF|          0017|       10.5262s|       13.1816s|
|     BRISK|         BRIEF|          0018|        8.4862s|       13.6771s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|     BRISK|           ORB|          0000|           nans|           nans|
|     BRISK|           ORB|          0001|       12.6940s|       18.7747s|
|     BRISK|           ORB|          0002|       12.0310s|       22.9000s|
|     BRISK|           ORB|          0003|       17.3639s|       20.2640s|
|     BRISK|           ORB|          0004|       16.0990s|       20.6342s|
|     BRISK|           ORB|          0005|       13.3004s|       22.9633s|
|     BRISK|           ORB|          0006|       13.1047s|       22.9839s|
|     BRISK|           ORB|          0007|       12.1979s|       24.8693s|
|     BRISK|           ORB|          0008|       13.3858s|       18.1632s|
|     BRISK|           ORB|          0009|       13.4396s|       20.1061s|
|     BRISK|           ORB|          0010|       12.1049s|       15.4474s|
|     BRISK|           ORB|          0011|       12.2210s|       16.4451s|
|     BRISK|           ORB|          0012|        9.2248s|       13.7577s|
|     BRISK|           ORB|          0013|        9.5029s|       12.9999s|
|     BRISK|           ORB|          0014|        9.5302s|       15.9064s|
|     BRISK|           ORB|          0015|        8.1272s|       15.2110s|
|     BRISK|           ORB|          0016|        9.9349s|       13.8079s|
|     BRISK|           ORB|          0017|       10.5262s|       11.4080s|
|     BRISK|           ORB|          0018|        8.4862s|       14.7506s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|     BRISK|         FREAK|          0000|           nans|           nans|
|     BRISK|         FREAK|          0001|       12.6940s|       17.5975s|
|     BRISK|         FREAK|          0002|       12.0310s|       24.3133s|
|     BRISK|         FREAK|          0003|       17.3639s|       18.9989s|
|     BRISK|         FREAK|          0004|       16.0990s|       15.3470s|
|     BRISK|         FREAK|          0005|       13.3004s|       30.1017s|
|     BRISK|         FREAK|          0006|       13.1047s|       19.5104s|
|     BRISK|         FREAK|          0007|       12.1979s|       23.4160s|
|     BRISK|         FREAK|          0008|       13.3858s|       17.1189s|
|     BRISK|         FREAK|          0009|       13.4396s|       18.6577s|
|     BRISK|         FREAK|          0010|       12.1049s|       14.7642s|
|     BRISK|         FREAK|          0011|       12.2210s|       14.8411s|
|     BRISK|         FREAK|          0012|        9.2248s|       12.3387s|
|     BRISK|         FREAK|          0013|        9.5029s|       12.9704s|
|     BRISK|         FREAK|          0014|        9.5302s|       16.1370s|
|     BRISK|         FREAK|          0015|        8.1272s|       14.9891s|
|     BRISK|         FREAK|          0016|        9.9349s|       12.3828s|
|     BRISK|         FREAK|          0017|       10.5262s|       10.7686s|
|     BRISK|         FREAK|          0018|        8.4862s|       13.8477s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|     BRISK|          SIFT|          0000|           nans|           nans|
|     BRISK|          SIFT|          0001|       12.6940s|       17.2788s|
|     BRISK|          SIFT|          0002|       12.0310s|       18.7891s|
|     BRISK|          SIFT|          0003|       17.3639s|       20.1856s|
|     BRISK|          SIFT|          0004|       16.0990s|       14.8334s|
|     BRISK|          SIFT|          0005|       13.3004s|       28.2877s|
|     BRISK|          SIFT|          0006|       13.1047s|       17.6358s|
|     BRISK|          SIFT|          0007|       12.1979s|       15.0635s|
|     BRISK|          SIFT|          0008|       13.3858s|       15.2211s|
|     BRISK|          SIFT|          0009|       13.4396s|       15.1894s|
|     BRISK|          SIFT|          0010|       12.1049s|       14.6158s|
|     BRISK|          SIFT|          0011|       12.2210s|       11.9260s|
|     BRISK|          SIFT|          0012|        9.2248s|       11.9286s|
|     BRISK|          SIFT|          0013|        9.5029s|       12.4028s|
|     BRISK|          SIFT|          0014|        9.5302s|       13.3204s|
|     BRISK|          SIFT|          0015|        8.1272s|       13.4952s|
|     BRISK|          SIFT|          0016|        9.9349s|       11.2916s|
|     BRISK|          SIFT|          0017|       10.5262s|       11.0207s|
|     BRISK|          SIFT|          0018|        8.4862s|       11.6379s|

-----------------------------------------------------------------------------

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|       ORB|         BRISK|          0000|           nans|           nans|
|       ORB|         BRISK|          0001|       12.6940s|       16.7826s|
|       ORB|         BRISK|          0002|       12.0310s|       10.1192s|
|       ORB|         BRISK|          0003|       17.3639s|       24.2494s|
|       ORB|         BRISK|          0004|       16.0990s|       19.3375s|
|       ORB|         BRISK|          0005|       13.3004s|       45.5018s|
|       ORB|         BRISK|          0006|       13.1047s|       32.0020s|
|       ORB|         BRISK|          0007|       12.1979s|          -infs|
|       ORB|         BRISK|          0008|       13.3858s|       12.7867s|
|       ORB|         BRISK|          0009|       13.4396s|       21.5135s|
|       ORB|         BRISK|          0010|       12.1049s|          -infs|
|       ORB|         BRISK|          0011|       12.2210s|        8.4254s|
|       ORB|         BRISK|          0012|        9.2248s|          -infs|
|       ORB|         BRISK|          0013|        9.5029s|       11.6384s|
|       ORB|         BRISK|          0014|        9.5302s|       27.8458s|
|       ORB|         BRISK|          0015|        8.1272s|       13.6463s|
|       ORB|         BRISK|          0016|        9.9349s|       35.0129s|
|       ORB|         BRISK|          0017|       10.5262s|       20.3758s|
|       ORB|         BRISK|          0018|        8.4862s|       42.9951s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|       ORB|         BRIEF|          0000|           nans|           nans|
|       ORB|         BRIEF|          0001|       12.6940s|       24.1759s|
|       ORB|         BRIEF|          0002|       12.0310s|          -infs|
|       ORB|         BRIEF|          0003|       17.3639s|       44.7040s|
|       ORB|         BRIEF|          0004|       16.0990s|       17.5321s|
|       ORB|         BRIEF|          0005|       13.3004s|       35.4197s|
|       ORB|         BRIEF|          0006|       13.1047s|       31.7015s|
|       ORB|         BRIEF|          0007|       12.1979s|          -infs|
|       ORB|         BRIEF|          0008|       13.3858s|       18.3178s|
|       ORB|         BRIEF|          0009|       13.4396s|      195.2898s|
|       ORB|         BRIEF|          0010|       12.1049s|       52.1582s|
|       ORB|         BRIEF|          0011|       12.2210s|       25.2823s|
|       ORB|         BRIEF|          0012|        9.2248s|       16.2227s|
|       ORB|         BRIEF|          0013|        9.5029s|       13.4321s|
|       ORB|         BRIEF|          0014|        9.5302s|       60.6236s|
|       ORB|         BRIEF|          0015|        8.1272s|       45.5112s|
|       ORB|         BRIEF|          0016|        9.9349s|       24.5366s|
|       ORB|         BRIEF|          0017|       10.5262s|       15.0441s|
|       ORB|         BRIEF|          0018|        8.4862s|       28.2631s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|       ORB|           ORB|          0000|           nans|           nans|
|       ORB|           ORB|          0001|       12.6940s|       37.6140s|
|       ORB|           ORB|          0002|       12.0310s|       10.1802s|
|       ORB|           ORB|          0003|       17.3639s|       22.3594s|
|       ORB|           ORB|          0004|       16.0990s|       21.8457s|
|       ORB|           ORB|          0005|       13.3004s|      163.2342s|
|       ORB|           ORB|          0006|       13.1047s|          -infs|
|       ORB|           ORB|          0007|       12.1979s|          -infs|
|       ORB|           ORB|          0008|       13.3858s|       15.7290s|
|       ORB|           ORB|          0009|       13.4396s|       21.1185s|
|       ORB|           ORB|          0010|       12.1049s|          -infs|
|       ORB|           ORB|          0011|       12.2210s|       10.0197s|
|       ORB|           ORB|          0012|        9.2248s|          -infs|
|       ORB|           ORB|          0013|        9.5029s|          -infs|
|       ORB|           ORB|          0014|        9.5302s|          -infs|
|       ORB|           ORB|          0015|        8.1272s|      144.1523s|
|       ORB|           ORB|          0016|        9.9349s|       22.2594s|
|       ORB|           ORB|          0017|       10.5262s|       11.2808s|
|       ORB|           ORB|          0018|        8.4862s|       82.6523s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|       ORB|         FREAK|          0000|           nans|           nans|
|       ORB|         FREAK|          0001|       12.6940s|       15.8586s|
|       ORB|         FREAK|          0002|       12.0310s|       17.2527s|
|       ORB|         FREAK|          0003|       17.3639s|       19.8566s|
|       ORB|         FREAK|          0004|       16.0990s|       12.6081s|
|       ORB|         FREAK|          0005|       13.3004s|       45.5018s|
|       ORB|         FREAK|          0006|       13.1047s|       57.0977s|
|       ORB|         FREAK|          0007|       12.1979s|          -infs|
|       ORB|         FREAK|          0008|       13.3858s|        9.7842s|
|       ORB|         FREAK|          0009|       13.4396s|       15.9762s|
|       ORB|         FREAK|          0010|       12.1049s|       65.6540s|
|       ORB|         FREAK|          0011|       12.2210s|        8.1489s|
|       ORB|         FREAK|          0012|        9.2248s|       50.7145s|
|       ORB|         FREAK|          0013|        9.5029s|        9.3572s|
|       ORB|         FREAK|          0014|        9.5302s|       54.2226s|
|       ORB|         FREAK|          0015|        8.1272s|        8.5350s|
|       ORB|         FREAK|          0016|        9.9349s|       10.4631s|
|       ORB|         FREAK|          0017|       10.5262s|        6.9482s|
|       ORB|         FREAK|          0018|        8.4862s|       24.1855s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|       ORB|          SIFT|          0000|           nans|           nans|
|       ORB|          SIFT|          0001|       12.6940s|       14.4902s|
|       ORB|          SIFT|          0002|       12.0310s|        9.5420s|
|       ORB|          SIFT|          0003|       17.3639s|       15.8576s|
|       ORB|          SIFT|          0004|       16.0990s|       23.2249s|
|       ORB|          SIFT|          0005|       13.3004s|       29.8659s|
|       ORB|          SIFT|          0006|       13.1047s|       16.4949s|
|       ORB|          SIFT|          0007|       12.1979s|      906.5233s|
|       ORB|          SIFT|          0008|       13.3858s|       12.9210s|
|       ORB|          SIFT|          0009|       13.4396s|       13.7532s|
|       ORB|          SIFT|          0010|       12.1049s|          -infs|
|       ORB|          SIFT|          0011|       12.2210s|        7.9560s|
|       ORB|          SIFT|          0012|        9.2248s|       16.8999s|
|       ORB|          SIFT|          0013|        9.5029s|       10.6768s|
|       ORB|          SIFT|          0014|        9.5302s|       18.5566s|
|       ORB|          SIFT|          0015|        8.1272s|          -infs|
|       ORB|          SIFT|          0016|        9.9349s|       11.4315s|
|       ORB|          SIFT|          0017|       10.5262s|       11.1814s|
|       ORB|          SIFT|          0018|        8.4862s|       23.3647s|

-----------------------------------------------------------------------------

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|     AKAZE|         BRISK|          0000|           nans|           nans|
|     AKAZE|         BRISK|          0001|       12.6940s|       15.9469s|
|     AKAZE|         BRISK|          0002|       12.0310s|       17.8169s|
|     AKAZE|         BRISK|          0003|       17.3639s|       16.2678s|
|     AKAZE|         BRISK|          0004|       16.0990s|       15.3951s|
|     AKAZE|         BRISK|          0005|       13.3004s|       17.1252s|
|     AKAZE|         BRISK|          0006|       13.1047s|       19.1867s|
|     AKAZE|         BRISK|          0007|       12.1979s|       21.7614s|
|     AKAZE|         BRISK|          0008|       13.3858s|       17.1293s|
|     AKAZE|         BRISK|          0009|       13.4396s|       17.2759s|
|     AKAZE|         BRISK|          0010|       12.1049s|       13.0903s|
|     AKAZE|         BRISK|          0011|       12.2210s|       14.2321s|
|     AKAZE|         BRISK|          0012|        9.2248s|       12.6941s|
|     AKAZE|         BRISK|          0013|        9.5029s|       11.8704s|
|     AKAZE|         BRISK|          0014|        9.5302s|       11.8377s|
|     AKAZE|         BRISK|          0015|        8.1272s|       14.9621s|
|     AKAZE|         BRISK|          0016|        9.9349s|       12.6351s|
|     AKAZE|         BRISK|          0017|       10.5262s|       10.8491s|
|     AKAZE|         BRISK|          0018|        8.4862s|        9.9450s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|     AKAZE|         BRIEF|          0000|           nans|           nans|
|     AKAZE|         BRIEF|          0001|       12.6940s|       20.8162s|
|     AKAZE|         BRIEF|          0002|       12.0310s|       17.6241s|
|     AKAZE|         BRIEF|          0003|       17.3639s|       15.8624s|
|     AKAZE|         BRIEF|          0004|       16.0990s|       15.6698s|
|     AKAZE|         BRIEF|          0005|       13.3004s|       17.6416s|
|     AKAZE|         BRIEF|          0006|       13.1047s|       17.5862s|
|     AKAZE|         BRIEF|          0007|       12.1979s|       20.4121s|
|     AKAZE|         BRIEF|          0008|       13.3858s|       17.0796s|
|     AKAZE|         BRIEF|          0009|       13.4396s|       18.4164s|
|     AKAZE|         BRIEF|          0010|       12.1049s|       12.4503s|
|     AKAZE|         BRIEF|          0011|       12.2210s|       13.9969s|
|     AKAZE|         BRIEF|          0012|        9.2248s|       13.9496s|
|     AKAZE|         BRIEF|          0013|        9.5029s|       11.5248s|
|     AKAZE|         BRIEF|          0014|        9.5302s|       11.6523s|
|     AKAZE|         BRIEF|          0015|        8.1272s|       12.0581s|
|     AKAZE|         BRIEF|          0016|        9.9349s|       12.0309s|
|     AKAZE|         BRIEF|          0017|       10.5262s|       10.8945s|
|     AKAZE|         BRIEF|          0018|        8.4862s|       10.5249s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|     AKAZE|           ORB|          0000|           nans|           nans|
|     AKAZE|           ORB|          0001|       12.6940s|       16.0811s|
|     AKAZE|           ORB|          0002|       12.0310s|       17.3441s|
|     AKAZE|           ORB|          0003|       17.3639s|       15.2794s|
|     AKAZE|           ORB|          0004|       16.0990s|       15.9317s|
|     AKAZE|           ORB|          0005|       13.3004s|       17.3218s|
|     AKAZE|           ORB|          0006|       13.1047s|       17.3384s|
|     AKAZE|           ORB|          0007|       12.1979s|       19.9245s|
|     AKAZE|           ORB|          0008|       13.3858s|       16.7307s|
|     AKAZE|           ORB|          0009|       13.4396s|       17.8242s|
|     AKAZE|           ORB|          0010|       12.1049s|       12.7257s|
|     AKAZE|           ORB|          0011|       12.2210s|       14.0523s|
|     AKAZE|           ORB|          0012|        9.2248s|       14.9698s|
|     AKAZE|           ORB|          0013|        9.5029s|       11.8578s|
|     AKAZE|           ORB|          0014|        9.5302s|       12.2273s|
|     AKAZE|           ORB|          0015|        8.1272s|       14.7306s|
|     AKAZE|           ORB|          0016|        9.9349s|       11.5271s|
|     AKAZE|           ORB|          0017|       10.5262s|       10.6337s|
|     AKAZE|           ORB|          0018|        8.4862s|       10.6225s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|     AKAZE|         FREAK|          0000|           nans|           nans|
|     AKAZE|         FREAK|          0001|       12.6940s|       15.7402s|
|     AKAZE|         FREAK|          0002|       12.0310s|       17.6571s|
|     AKAZE|         FREAK|          0003|       17.3639s|       16.5485s|
|     AKAZE|         FREAK|          0004|       16.0990s|       15.2564s|
|     AKAZE|         FREAK|          0005|       13.3004s|       17.7390s|
|     AKAZE|         FREAK|          0006|       13.1047s|       17.8397s|
|     AKAZE|         FREAK|          0007|       12.1979s|       20.8023s|
|     AKAZE|         FREAK|          0008|       13.3858s|       16.9250s|
|     AKAZE|         FREAK|          0009|       13.4396s|       18.1219s|
|     AKAZE|         FREAK|          0010|       12.1049s|       13.7871s|
|     AKAZE|         FREAK|          0011|       12.2210s|       14.2653s|
|     AKAZE|         FREAK|          0012|        9.2248s|       14.7644s|
|     AKAZE|         FREAK|          0013|        9.5029s|       11.6331s|
|     AKAZE|         FREAK|          0014|        9.5302s|       11.3156s|
|     AKAZE|         FREAK|          0015|        8.1272s|       12.6697s|
|     AKAZE|         FREAK|          0016|        9.9349s|       12.9769s|
|     AKAZE|         FREAK|          0017|       10.5262s|       10.9091s|
|     AKAZE|         FREAK|          0018|        8.4862s|        9.8091s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|     AKAZE|         AKAZE|          0000|           nans|           nans|
|     AKAZE|         AKAZE|          0001|       12.6940s|       16.8159s|
|     AKAZE|         AKAZE|          0002|       12.0310s|       17.7118s|
|     AKAZE|         AKAZE|          0003|       17.3639s|       15.5471s|
|     AKAZE|         AKAZE|          0004|       16.0990s|       17.0062s|
|     AKAZE|         AKAZE|          0005|       13.3004s|       17.7416s|
|     AKAZE|         AKAZE|          0006|       13.1047s|       16.8094s|
|     AKAZE|         AKAZE|          0007|       12.1979s|       18.9116s|
|     AKAZE|         AKAZE|          0008|       13.3858s|       16.8142s|
|     AKAZE|         AKAZE|          0009|       13.4396s|       16.9813s|
|     AKAZE|         AKAZE|          0010|       12.1049s|       13.2078s|
|     AKAZE|         AKAZE|          0011|       12.2210s|       13.3483s|
|     AKAZE|         AKAZE|          0012|        9.2248s|       13.2811s|
|     AKAZE|         AKAZE|          0013|        9.5029s|       12.0634s|
|     AKAZE|         AKAZE|          0014|        9.5302s|       11.9308s|
|     AKAZE|         AKAZE|          0015|        8.1272s|       12.5649s|
|     AKAZE|         AKAZE|          0016|        9.9349s|       11.7735s|
|     AKAZE|         AKAZE|          0017|       10.5262s|       10.6148s|
|     AKAZE|         AKAZE|          0018|        8.4862s|       10.2548s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|     AKAZE|          SIFT|          0000|           nans|           nans|
|     AKAZE|          SIFT|          0001|       12.6940s|       16.9605s|
|     AKAZE|          SIFT|          0002|       12.0310s|       17.2272s|
|     AKAZE|          SIFT|          0003|       17.3639s|       15.5654s|
|     AKAZE|          SIFT|          0004|       16.0990s|       17.2530s|
|     AKAZE|          SIFT|          0005|       13.3004s|       18.0108s|
|     AKAZE|          SIFT|          0006|       13.1047s|       17.4215s|
|     AKAZE|          SIFT|          0007|       12.1979s|       19.5604s|
|     AKAZE|          SIFT|          0008|       13.3858s|       16.4822s|
|     AKAZE|          SIFT|          0009|       13.4396s|       17.0641s|
|     AKAZE|          SIFT|          0010|       12.1049s|       13.0747s|
|     AKAZE|          SIFT|          0011|       12.2210s|       13.3573s|
|     AKAZE|          SIFT|          0012|        9.2248s|       13.2902s|
|     AKAZE|          SIFT|          0013|        9.5029s|       12.1820s|
|     AKAZE|          SIFT|          0014|        9.5302s|       11.8017s|
|     AKAZE|          SIFT|          0015|        8.1272s|       12.7103s|
|     AKAZE|          SIFT|          0016|        9.9349s|       11.2770s|
|     AKAZE|          SIFT|          0017|       10.5262s|       10.5009s|
|     AKAZE|          SIFT|          0018|        8.4862s|        9.9836s|

-----------------------------------------------------------------------------

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|      SIFT|         BRISK|          0000|           nans|           nans|
|      SIFT|         BRISK|          0001|       12.6940s|       15.5906s|
|      SIFT|         BRISK|          0002|       12.0310s|       17.8582s|
|      SIFT|         BRISK|          0003|       17.3639s|       22.1008s|
|      SIFT|         BRISK|          0004|       16.0990s|       22.7348s|
|      SIFT|         BRISK|          0005|       13.3004s|       17.5141s|
|      SIFT|         BRISK|          0006|       13.1047s|       20.9884s|
|      SIFT|         BRISK|          0007|       12.1979s|       21.0563s|
|      SIFT|         BRISK|          0008|       13.3858s|       18.5753s|
|      SIFT|         BRISK|          0009|       13.4396s|       15.7640s|
|      SIFT|         BRISK|          0010|       12.1049s|       11.4007s|
|      SIFT|         BRISK|          0011|       12.2210s|       15.3544s|
|      SIFT|         BRISK|          0012|        9.2248s|       11.4263s|
|      SIFT|         BRISK|          0013|        9.5029s|       11.2556s|
|      SIFT|         BRISK|          0014|        9.5302s|       11.7701s|
|      SIFT|         BRISK|          0015|        8.1272s|        9.9716s|
|      SIFT|         BRISK|          0016|        9.9349s|        9.7441s|
|      SIFT|         BRISK|          0017|       10.5262s|       10.9129s|
|      SIFT|         BRISK|          0018|        8.4862s|       12.8614s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|      SIFT|         BRIEF|          0000|           nans|           nans|
|      SIFT|         BRIEF|          0001|       12.6940s|       16.5194s|
|      SIFT|         BRIEF|          0002|       12.0310s|       13.8293s|
|      SIFT|         BRIEF|          0003|       17.3639s|       17.4897s|
|      SIFT|         BRIEF|          0004|       16.0990s|       22.7348s|
|      SIFT|         BRIEF|          0005|       13.3004s|       17.5891s|
|      SIFT|         BRIEF|          0006|       13.1047s|       15.7286s|
|      SIFT|         BRIEF|          0007|       12.1979s|       18.1654s|
|      SIFT|         BRIEF|          0008|       13.3858s|       17.4125s|
|      SIFT|         BRIEF|          0009|       13.4396s|       13.0992s|
|      SIFT|         BRIEF|          0010|       12.1049s|       10.2220s|
|      SIFT|         BRIEF|          0011|       12.2210s|       13.8759s|
|      SIFT|         BRIEF|          0012|        9.2248s|       10.6914s|
|      SIFT|         BRIEF|          0013|        9.5029s|       11.2580s|
|      SIFT|         BRIEF|          0014|        9.5302s|       11.7298s|
|      SIFT|         BRIEF|          0015|        8.1272s|       10.9479s|
|      SIFT|         BRIEF|          0016|        9.9349s|        9.5675s|
|      SIFT|         BRIEF|          0017|       10.5262s|       11.2464s|
|      SIFT|         BRIEF|          0018|        8.4862s|       12.5151s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|      SIFT|         FREAK|          0000|           nans|           nans|
|      SIFT|         FREAK|          0001|       12.6940s|       16.3965s|
|      SIFT|         FREAK|          0002|       12.0310s|       18.3606s|
|      SIFT|         FREAK|          0003|       17.3639s|       22.2537s|
|      SIFT|         FREAK|          0004|       16.0990s|       21.8187s|
|      SIFT|         FREAK|          0005|       13.3004s|       21.8638s|
|      SIFT|         FREAK|          0006|       13.1047s|       17.3353s|
|      SIFT|         FREAK|          0007|       12.1979s|       19.9302s|
|      SIFT|         FREAK|          0008|       13.3858s|       17.0116s|
|      SIFT|         FREAK|          0009|       13.4396s|       15.4507s|
|      SIFT|         FREAK|          0010|       12.1049s|       11.4295s|
|      SIFT|         FREAK|          0011|       12.2210s|       15.1196s|
|      SIFT|         FREAK|          0012|        9.2248s|       11.4393s|
|      SIFT|         FREAK|          0013|        9.5029s|       10.7286s|
|      SIFT|         FREAK|          0014|        9.5302s|       11.7025s|
|      SIFT|         FREAK|          0015|        8.1272s|       10.5617s|
|      SIFT|         FREAK|          0016|        9.9349s|       10.1457s|
|      SIFT|         FREAK|          0017|       10.5262s|       10.4312s|
|      SIFT|         FREAK|          0018|        8.4862s|       12.0489s|

|Detector type|Descriptor type|Image number|TTC of Lidar|TTC of Camera|
|:--------:|:--------:|:--------:|:--------:|:--------:|
|      SIFT|          SIFT|          0000|           nans|           nans|
|      SIFT|          SIFT|          0001|       12.6940s|       14.8482s|
|      SIFT|          SIFT|          0002|       12.0310s|       17.0145s|
|      SIFT|          SIFT|          0003|       17.3639s|       17.0855s|
|      SIFT|          SIFT|          0004|       16.0990s|       23.8457s|
|      SIFT|          SIFT|          0005|       13.3004s|       16.6151s|
|      SIFT|          SIFT|          0006|       13.1047s|       18.4668s|
|      SIFT|          SIFT|          0007|       12.1979s|       16.7137s|
|      SIFT|          SIFT|          0008|       13.3858s|       18.1649s|
|      SIFT|          SIFT|          0009|       13.4396s|       14.4615s|
|      SIFT|          SIFT|          0010|       12.1049s|       11.3295s|
|      SIFT|          SIFT|          0011|       12.2210s|       14.5464s|
|      SIFT|          SIFT|          0012|        9.2248s|       12.6781s|
|      SIFT|          SIFT|          0013|        9.5029s|       11.6395s|
|      SIFT|          SIFT|          0014|        9.5302s|       11.5741s|
|      SIFT|          SIFT|          0015|        8.1272s|       10.9398s|
|      SIFT|          SIFT|          0016|        9.9349s|        9.7848s|
|      SIFT|          SIFT|          0017|       10.5262s|       10.2891s|
|      SIFT|          SIFT|          0018|        8.4862s|       11.1022s|

-----------------------------------------------------------------------------
