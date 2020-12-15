/*
LogisticRegression.h
trains a logistic regression classifier.
Estimates the parameters of a logistic model.
f = sigmoid(sum(wi * xi) + b)

A gradient descent method is used with RmsProp optimizer.
Where the loss function is the cross entropy.

Greg Smith
gregjksmith@gmail.com
*/

#pragma once
#include <WeakLearner.h>
#include <cmath>

//base learning rate
#define LEARNING_RATE 0.1f

//maximum number of learning epochs
#define MAX_EPOCHS 100

//running average weight, used with the adaptive learning rate.
#define RUNNING_AVERAGE_WEIGHT 0.9f

//training optimum is considered if the absolute gradient is less
//than this threshold.
#define GRADIENT_THRESH 1e-3f

class LogisticRegression : public WeakLearner
{
public:
	LogisticRegression();
	~LogisticRegression();

	virtual float label(Sample* x);

	virtual void train(std::vector<Sample*>& samples, float* sampleWeights, int classIndex);

protected:
	float sigmoid(Sample* s);
	float sigmoidLabel(int class0, int class1);

	virtual void exportInternal(std::string& params);
	virtual void importInternal(std::string& params);

private:
	void clearBuffer(float* buffer, int numSamples);

	float* _w = nullptr;	//logistic weights.
	float _b;	//logistic bias.
	int _sampleSize;
};