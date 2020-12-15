/*
Support Vector Machine.
Solves for a separating hyperplane between samples of two classes.
SVM is solved using the sequential minization optimization (smo) algorithm.

Greg Smith
gregjksmith@gmail.com
*/

#pragma once
#include <WeakLearner.h>

//equality tolerance, two samples are considered equal if ther vary by EQUALITY_TOLERANCE
#define EQUALITY_TOLERANCE 0.01f

//SVM hyperparameter, which allows for soft boundaries.
#define C 0.05f

//number of non-changing iterations, until the algorithm stops.
#define MAX_PASSES 10

class Svm : public WeakLearner
{
public:
	Svm();
	virtual ~Svm();
	virtual float label(Sample* x);
	virtual void train(std::vector<Sample*>& samples, float* sampleWeights, int classIndex);

protected:
	virtual void exportInternal(std::string& params);
	virtual void importInternal(std::string& params);

private:
	float* _w; //hyperplane normal	
	float _b; //hyperplane bias
	int _n; //vector size of each training sample.

	/*
	Computes the inner product / dot product between two samples x0 and x1. 
	*/
	float innerProduct(Sample* x0, Sample* x1);
};