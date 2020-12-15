/*
NaiveBayes.h
Trains a classifier using Naive Bayes.
Each attribute for each sample is assumed to independent, and
gaussian distributed. The means and variances of each
attitube given a label is calculated. The most likely label
can then be computed given the sample.

Greg Smith
gregjksmith@gmail.com
*/

#pragma once
#include <WeakLearner.h>

class NaiveBayes : public WeakLearner
{
public:
	NaiveBayes();

	virtual ~NaiveBayes();
	virtual float label(Sample* x);
	virtual void train(std::vector<Sample*>& samples, float* sampleWeights, int classIndex);

protected:
	virtual void exportInternal(std::string& params);
	virtual void importInternal(std::string& params);

private:
	int _n;	//number of attributes in a sample.
	float* _mean;	//attribute means for positive and negative samples.
	float* _var;	//attribute variance for positibe and negative samples.
};