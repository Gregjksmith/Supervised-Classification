/*
WeakLearner.cpp
Base abstact class which provides an interface for training a set of samples,
computing the estimated label given a sample and computing the training error given a
training set.

All supervised classification algorithms derive from this class.

Greg Smith
gregjksmith@gmail.com
*/

#include <WeakLearner.h>

WeakLearner::WeakLearner()
{

}

WeakLearner::~WeakLearner()
{

}

/*
Trains a supervised learning algorithm given a set of samples and a set class.
Training is performed one agains many, where the sample is considered positive (+1)
if it matches the desired class negative 'classIndex' and negative (-1) if it does not.

std::vector<Sample*> samples: input training set;
int classIndex: positive class Index.
*/
void WeakLearner::train(std::vector<Sample*>& samples, int classIndex)
{
	//no sample weighting is specified. Call the training algoirithm
	//using uniform sampling.
	int sampleSize = samples.size();
	float* w = new float[sampleSize];
	for (int i = 0; i < sampleSize; i++)
	{
		w[i] = 1.0f / (float)sampleSize;
	}
	train(samples, w, classIndex);

	delete[] w;
}

/*
Trains a supervised learning algorithm given a set of samples and a set class.
Training is performed one agains many, where the sample is considered positive (+1)
if it matches the desired class negative 'classIndex' and negative (-1) if it does not.

std::vector<Sample*> samples: input training set;
int classIndex: positive class Index.
float* sampleWeights: provides a mechansism to weight samples during samples. Utilized with AdaBoost.
*/
float WeakLearner::error(std::vector<Sample*>& samples, int classIndex)
{
	int numErrors = 0;
	for (int i = 0; i < samples.size(); i++)
	{
		Sample* x = samples[i];
		float l = label(x);
		if (l * binaryLabel(x->y(), classIndex) <= 0.0f)
			numErrors++;
	}
	return (float)numErrors / (float)samples.size();
}

/*
Returns 1.0 if the two class indices match, -1.0 else.
*/
float WeakLearner::binaryLabel(int class0, int class1)
{
	if (class0 == class1)
		return 1.0f;
	return -1.0f;
}