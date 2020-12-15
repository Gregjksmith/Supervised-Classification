/*
WeakLearner.h
Base abstact class which provides an interface for training a set of samples, 
computing the estimated label given a sample and computing the training error given a
training set.

All supervised classification algorithms derive from this class.

Greg Smith
gregjksmith@gmail.com
*/

#pragma once
#include <Sample.h>
#include <vector>
#include <fstream>
#include <string>
#include <ExportStringUtils.h>

/*
WeakLearner. Base Learner class. All supervised algorithms derive from this class.
*/
class WeakLearner
{
public:
	WeakLearner();
	virtual ~WeakLearner();

	/*
	Computes the estimated label betwen [-1,1] of a sample given the learned
	parameters of the algorithm.
	*/
	virtual float label(Sample* x) = 0;
	
	/*
	Trains a supervised learning algorithm given a set of samples and a set class.
	Training is performed one agains many, where the sample is considered positive (+1)
	if it matches the desired class negative 'classIndex' and negative (-1) if it does not.
	
	std::vector<Sample*> samples: input training set;
	int classIndex: positive class Index.
	float* sampleWeights: provides a mechansism to weight samples during samples. Utilized with AdaBoost.
	*/
	void train(std::vector<Sample*>& samples, int classIndex);
	virtual void train(std::vector<Sample*>& samples, float* sampleWeights, int classIndex) = 0;

	/*
	Computes the classification error on a labeled data set 'samples', with classIndex.
	*/
	virtual float error(std::vector<Sample*>& samples, int classIndex);

	std::string exportParams();

	void importParams(std::string& params);

protected:
	/*
	Returns 1.0 if the two class indices match, -1.0 else.
	*/
	virtual float binaryLabel(int class0, int class1);

	virtual void exportInternal(std::string& params) = 0;
	virtual void importInternal(std::string& params) = 0;
};