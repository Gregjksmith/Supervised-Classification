#include <NaiveBayes.h>
#include <cmath>
#include <vector>

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
		float positiveMeanSum = 0.0f;
		float negativeMeanSum = 0.0f;

		float positiveWeightSum = 0.0f;
		float negativeWeightSum = 0.0f;

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
				negativeWeightSum += sampleWeights[i];
			}
		}

		//compute the means.
		positiveMeanSum /= positiveWeightSum;
		negativeMeanSum /= negativeWeightSum;

		positiveWeightSum = 0.0f;
		negativeWeightSum = 0.0f;

		float positiveVarSum = 0.0f;
		float negativeVarSum = 0.0f;

		//iterate through each sample to compute the variances.s
		for (int i = 0; i < samples.size(); i++)
		{
			Sample* x = samples[i];
			//compute the positive attribute variance.
			if (x->y() == classIndex)
			{
				positiveVarSum += sampleWeights[i] * pow(x->x(j) - positiveMeanSum, 2.0f);
				positiveWeightSum += sampleWeights[i];
			}
			//compute the negative attribute variance.
			else
			{
				negativeVarSum += sampleWeights[i] * pow(x->x(j) - negativeMeanSum, 2.0f);
				negativeWeightSum += sampleWeights[i];
			}
		}

		//get the variances.
		positiveVarSum /= positiveWeightSum;
		negativeVarSum /= negativeWeightSum;

		_mean[j * 2 + 0] = positiveMeanSum;
		_mean[j * 2 + 1] = negativeMeanSum;

		_var[j * 2 + 0] = fmax(positiveVarSum, 1e-5f);
		_var[j * 2 + 1] = fmax(negativeVarSum, 1e-5f);
	}
}