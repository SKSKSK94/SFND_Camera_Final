
#include <iostream>
#include <set>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255)); // white background

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object (Random Number Generator)
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {   
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        int lineType = 2;
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), lineType);

        // augment object with some key data
        char str1[200], str2[200];
        std::sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250/2, bottom+50/2), cv::FONT_ITALIC, 1, currColor);
        std::sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250/2, bottom+125/2), cv::FONT_ITALIC, 1, currColor);  
        
        // char str1[200], str2[200], str3[200];
        // std::sprintf(str1, "id=%d, #pts=%d", itr1->boxID, (int)(itr1->lidarPoints.size()));
        // cv::putText(topviewImg, str1, cv::Point(left-30, bottom+50), cv::FONT_ITALIC, 1, currColor);
        // std::sprintf(str2, "xmin=%2.2f m", xwmin);
        // cv::putText(topviewImg, str2, cv::Point(left-30, bottom+85), cv::FONT_ITALIC, 1, currColor);
        // std::sprintf(str3, "width=%2.2f m", ywmax-ywmin);
        // cv::putText(topviewImg, str3, cv::Point(left-30, bottom+120), cv::FONT_ITALIC, 1, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

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

    // STUDENT TASK (replacement for meanDistRatio)
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


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    //******** method 1 ********//
    // // find closest distance to Lidar points within ego lane
    // double minXPrev = 1e9, minXCurr = 1e9;
    // for (auto itr = lidarPointsPrev.begin(); itr != lidarPointsPrev.end(); ++itr)
    // {
    //     if (abs(itr->y) > laneWidth/2.0) continue;
    //     minXPrev = minXPrev > itr->x ? itr->x : minXPrev;
    // }

    // for (auto itr = lidarPointsCurr.begin(); itr != lidarPointsCurr.end(); ++itr)
    // {   
    //     if (abs(itr->y) > laneWidth/2.0) continue;
    //     minXCurr = minXCurr > itr->x ? itr->x : minXCurr;
    // }
    // TTC = minXCurr * dT / (minXPrev - minXCurr);
    //******** method 1 ********//

    //******** method 2 ********//
    // double meanXPrev = 0.0, meanXCurr = 0.0;
    // int xPrevSize = 0, xCurrSize = 0;
    // for (auto itr = lidarPointsPrev.begin(); itr != lidarPointsPrev.end(); ++itr)
    // {
    //     if (abs(itr->y) > laneWidth/2.0) continue;
    //     meanXPrev += itr->x;
    //     ++xPrevSize;
    // }
    // meanXPrev /= xPrevSize;

    // for (auto itr = lidarPointsCurr.begin(); itr != lidarPointsCurr.end(); ++itr)
    // {   
    //     if (abs(itr->y) > laneWidth/2.0) continue;
    //     meanXCurr += itr->x;
    //     ++xCurrSize;
    // }
    // meanXCurr /= xCurrSize;

    // // compute TTC from both measurements
    // TTC = meanXCurr * dT / (meanXPrev - meanXCurr);
    
    //******** method 2 ********//

    //******** method 3 ********//
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

    //******** method 3 ********//


    
}

// (prev, curr) = (source, ref) = (query, train) // bbBestMatches = (prev, curr)
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

    // for(auto itr = numMatches.begin(); itr != numMatches.end(); ++itr)
    // {
    //     for(auto elem : *itr)
    //     {
    //         std::cout << elem << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // finding number of matches
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

    // finding best combination of bounding boxes b/w current and previous frame based on number of matches
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

    // for(auto itr = numMatches.begin(); itr != numMatches.end(); ++itr)
    // {
    //     for(auto elem : *itr)
    //     {
    //         std::cout << elem << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << std::endl;
    // for (auto elem : numMatchesSave)
    // {
    //     std::cout << elem << " ";
    // }
    // std::cout << std::endl;

    // for (auto elem : bbBestMatches)
    // {
    //     std::cout << "(" << elem.first << ", " << elem.second << ") ";
    // }
    // std::cout << std::endl;
}
