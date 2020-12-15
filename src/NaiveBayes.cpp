#include <NaiveBayes.h>
#include <cmath>
#include <vector>
#include <Accumulator.h>

NaiveBayes::NaiveBayes() : WeakLearner()
{
	_n = 0;
	_mean = nullptr;
	_var = nullptr;
}

NaiveBayes::~NaiveBayes()
{
	delete[] _mean;
	delete[] _var;
}

float NaiveBayes::label(Sample* x)
{
	double positiveP = 0.0f;
	double negativeP = 0.0f;

	//add the log of the probabilities of each sample.
	for (int i = 0; i < x->n(); i++)
	{
		float meanSamplePositive = _mean[i * 2 + 0];
		float meanSampleNegative = _mean[i * 2 + 1];

		float varSamplePositive = _var[i * 2 + 0];
		float varSampleNegative = _var[i * 2 + 1];

		double positive = exp(-pow(x->x(i) - meanSamplePositive, 2.0f) / (2.0f * varSamplePositive));
		double negative = exp(-pow(x->x(i) - meanSampleNegative, 2.0f) / (2.0f * varSampleNegative));

		positive /= fmax(positive + negative, 1e-9f);
		negative = 1.0f - positive;

		positiveP += log(fmax(positive,1e-12));
		negativeP += log(fmax(negative, 1e-12));
	}

	double pSum = positiveP + negativeP;
	positiveP = 1.0f - positiveP / pSum;
	negativeP = 1.0f - positiveP;

	float l = positiveP - negativeP;
	return l;
}

void NaiveBayes::train(std::vector<Sample*>& samples, float* sampleWeights, int classIndex)
{
	_n = samples[0]->n();
	_mean = new float[_n * 2];
	_var = new float[_n * 2];

	//iterate through each attribute.
	for (int j = 0; j < _n; j++)
	{
		Accumulator positiveMeanSum;
		Accumulator negativeMeanSum;

		Accumulator positiveWeightSum;
		Accumulator negativeWeightSum;

		//iterate through each sample to compute the means.
		for (int i = 0; i < samples.size(); i++)
		{
			Sample* x = samples[i];
			//get the attribute sum of positive samples.
			if (x->y() == classIndex)
			{
				positiveMeanSum += sampleWeights[i] * x->x(j);
				positiveWeightSum += sampleWeights[i];
			}
			//get the attribute sum of negative samples.
			else
			{
				negativeMeanSum += sampleWeights[i] * x->x(j);
				negativeWeightSum = sampleWeights[i];
			}
		}

		//compute the means.
		positiveMeanSum = positiveMeanSum.sum() / positiveWeightSum.sum();
		negativeMeanSum = negativeMeanSum.sum() / negativeWeightSum.sum();

		Accumulator positiveVarSum;
		Accumulator negativeVarSum;

		//iterate through each sample to compute the variances.s
		for (int i = 0; i < samples.size(); i++)
		{
			Sample* x = samples[i];
			//compute the positive attribute variance.
			if (x->y() == classIndex)
			{
				positiveVarSum += sampleWeights[i] * pow(x->x(j) - positiveMeanSum.sum(), 2.0f);
			}
			//compute the negative attribute variance.
			else
			{
				negativeVarSum = sampleWeights[i] * pow(x->x(j) - negativeMeanSum.sum(), 2.0f);
			}
		}

		//get the variances.
		positiveVarSum = positiveVarSum.sum() / positiveWeightSum.sum();
		negativeVarSum = negativeVarSum.sum() / negativeWeightSum.sum();

		_mean[j * 2 + 0] = positiveMeanSum.sum();
		_mean[j * 2 + 1] = negativeMeanSum.sum();

		_var[j * 2 + 0] = fmax(positiveVarSum.sum(), 1e-5f);
		_var[j * 2 + 1] = fmax(negativeVarSum.sum(), 1e-5f);
	}
}

void NaiveBayes::exportInternal(std::string& params)
{
	params += std::to_string(_n) + WEAK_LEARNER_DELIM;

	for (int i = 0; i < _n * 2; i++)
		params += std::to_string(_mean[i]) + WEAK_LEARNER_DELIM;	

	for (int i = 0; i < _n * 2; i++)
		params += std::to_string(_var[i]) + WEAK_LEARNER_DELIM;
}
void NaiveBayes::importInternal(std::string& params)
{
	_n = atoi(getNextParam(params, WEAK_LEARNER_DELIM).c_str());
	
	_mean = new float[_n * 2];
	for (int i = 0; i < _n * 2; i++)
		_mean[i] = atof(getNextParam(params, WEAK_LEARNER_DELIM).c_str());
	
	_var = new float[_n * 2];
	for (int i = 0; i < _n * 2; i++)
		_var[i] = atof(getNextParam(params, WEAK_LEARNER_DELIM).c_str());
}