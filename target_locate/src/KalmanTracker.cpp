///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.cpp: KalmanTracker Class Implementation Declaration

#include "KalmanTracker.h"


int KalmanTracker::kf_count = 0;


// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat)
{
	int stateNum = 9;
	int measureNum = 3; // x, y, z
	kf = KalmanFilter(stateNum, measureNum, 0);

	measurement = Mat::zeros(measureNum, 1, CV_32F);

	kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) << // A
        1, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1);

	setIdentity(kf.measurementMatrix);                      // H
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2));     // Q
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1)); // R
	setIdentity(kf.errorCovPost, Scalar::all(1));           // Pk

	// initialize state vector with bounding box in [cx,cy,s,r] style
	// kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	// kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	// kf.statePost.at<float>(2, 0) = stateMat.area();
	// kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;

    kf.statePost.at<float>(0, 0) = stateMat.x;
	kf.statePost.at<float>(1, 0) = stateMat.y;
	kf.statePost.at<float>(2, 0) = stateMat.z;
}


// Predict the estimated bounding box.
StateType KalmanTracker::predict()
{
	// predict
	Mat p = kf.predict();

	if (time_since_update > 0)
		hit_streak = 0;
	time_since_update += 1;
    age += 1;

	// StateType predictBox = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));
    StateType predictPoint(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0));

	history.push_back(predictPoint);
	return history.back();
}


// Update the state vector with observed bounding box.
void KalmanTracker::update(StateType stateMat)
{   
    age = 0;
	time_since_update = 0;
	history.clear();
	hits += 1;
	hit_streak += 1;

	// measurement
	// measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	// measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	// measurement.at<float>(2, 0) = stateMat.area();
	// measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

    measurement.at<float>(0, 0) = stateMat.x;
	measurement.at<float>(1, 0) = stateMat.y;
	measurement.at<float>(2, 0) = stateMat.z;

	// update
	kf.correct(measurement);
}


// Return the current state vector
StateType KalmanTracker::get_state()
{
	Mat s = kf.statePost;
	// return get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
    return StateType(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0));
}