/*
AdaBoost.h
Adaptive Boosting is a technique to improve the performance of a classifier by leveraging multiple,
separately trained classifiers. The trained classifiers are refered to Weak Classifiers and are combined
into a strong classifier.

Greg Smith
gregjksmith@gmail.com
*/

#pragma once
#include <vector>
#include <Sample.h>
#include <WeakLearner.h>
#include <Accumulator.h>

template <class T>
class AdaBoost
{
public:
	/*
	Constructor:
	std::vector<Sample*>& samples: training set. provided as a vector of Samples*.
		each sample contains the raw input data x, and an associated label y.
		Training supports multiple classes, where the label can be any non-negative integer.
	int numWeakLearners: number of weak learners trained. Setting numWeakLearners = 1
		results in standard non-boosted classification.
	*/
	AdaBoost(std::vector<Sample*>& samples, int numWeakLearners)
	{
		_ensembles = nullptr;
		train(samples, numWeakLearners);
	}
	virtual ~AdaBoost()
	{
		for (int k = 0; k < _k; k++)
			delete _ensembles[k];
		delete[] _ensembles;
	}

	/*
	Computes the classification error of a data set.
	*/
	float error(std::vector<Sample*>& samples)
	{
		int numNegativeSamples = 0;
		for (int i = 0; i < samples.size(); i++)
		{
			float confidence;
			int n = label(samples[i], confidence);
			if (n != samples[i]->y())
				numNegativeSamples++;
		}
		return (float)numNegativeSamples / (float)samples.size();
	}

	/*
	Returns the most likely label for a sample given the learned parameters.
	Sample* x: input sample.
	float& confidence: likelihood of the label is stored here.
	*/
	int label(Sample* x, float &confidence)
	{
		float expSum = 0.0f;
		int maxClassIndex = 0;
		float maxLabel = _ensembles[0]->label(x);
		expSum += exp(maxLabel);
		for (int i = 1; i < _k; i++)
		{
			float l = _ensembles[i]->label(x);
			expSum += exp(l);
			if (l > maxLabel)
			{
				maxLabel = l;
				maxClassIndex = i;
			}
		}

		confidence = exp(maxLabel) / expSum;
		return maxClassIndex;
	}

	std::string exportParams()
	{
		std::string params;
		params += std::to_string(_numWeakLearners) + ENSEMBLE_DELIM;
		params += std::to_string(_n) + ENSEMBLE_DELIM;
		params += std::to_string(_k) + ENSEMBLE_DELIM;
		
		for (int k = 0; k < _k; k++)
		{
			for (int w = 0; w < _numWeakLearners; w++)
			{
				params += std::to_string(_ensembles[k]->weight(w)) + ENSEMBLE_DELIM;
				params += _ensembles[k]->weakLearner(w)->exportParams() + ENSEMBLE_DELIM;
			}
		}
		return params;
	}

	void importParams(std::string& params)
	{
		_numWeakLearners = atoi(getNextParam(params, ENSEMBLE_DELIM).c_str());
		_n = atoi(getNextParam(params, ENSEMBLE_DELIM).c_str());
		_k = atoi(getNextParam(params, ENSEMBLE_DELIM).c_str());

		if (_ensembles != nullptr)
			delete[] _ensembles;

		_ensembles = new Ensemble * [_k];
		for (int k = 0; k < _k; k++)
			_ensembles[k] = new Ensemble();

		for (int k = 0; k < _k; k++)
		{
			for (int w = 0; w < _numWeakLearners; w++)
			{
				float alpha = atof(getNextParam(params, ENSEMBLE_DELIM).c_str());
				std::string weakLearnerParams = getNextParam(params, ENSEMBLE_DELIM);

				WeakLearner* weakLearner = new T();
				weakLearner->importParams(weakLearnerParams);

				_ensembles[k]->addWeakLearner(weakLearner, alpha);
			}
		}
	}

private:

	/*
	Ensembe
	private nested container class.
	Contains a vector of trained weak learners and their associated weights.
	*/
	class Ensemble
	{
	public:
		Ensemble()
		{ }
		virtual ~Ensemble()
		{
			for (int i = 0; i < _weakLearners.size(); i++)
			{
				delete _weakLearners[i];
			}
			_weakLearners.clear();
			_weights.clear();
		}

		/*
		Appends the trained weak learner, and its associated weights
		to the ensemble.
		*/
		void addWeakLearner(WeakLearner* weakLearner, float w)
		{
			_weakLearners.push_back(weakLearner);
			_weights.push_back(w);
		}

		/*
		Computes the label of the sample x, given the ensemble of
		weak learners.
		*/
		float label(Sample* x)
		{
			float l = 0.0f;
			for (int i = 0; i < _weakLearners.size(); i++)
			{
				l += _weakLearners[i]->label(x) * _weights[i];
			}
			return l;
		}

		WeakLearner* weakLearner(int index)
		{
			return _weakLearners[index];
		}

		float weight(int index)
		{
			return _weights[index];
		}

	private:
		std::vector<WeakLearner*> _weakLearners;
		std::vector<float> _weights;
	};

	void train(std::vector<Sample*>& samples, int numWeakLearners)
	{
		int numSamples = samples.size();

		//create the samples weights, initialize with uniform weighting.
		float* w = new float[numSamples];

		//get the number of attributes for each sample.
		_n = samples[0]->n();
		_numWeakLearners = numWeakLearners;

		//compute the number of unique classes in the sample set.
		_k = getNumClasses(samples);

		_ensembles = new Ensemble * [_k];
		for (int k = 0; k < _k; k++)
			_ensembles[k] = new Ensemble();
		
		float* computedLabels = new float[numSamples];

		for (int k = 0; k < _k; k++)
		{
			int numPositiveSamples = 0;
			for (int i = 0; i < numSamples; i++)
			{
				if (samples[i]->y() == k)
					numPositiveSamples++;
			}
			
			for (int i = 0; i < numSamples; i++)
			{
				if (samples[i]->y() == k)
					w[i] = 1.0f / (float)(numPositiveSamples * 2);
				else
					w[i] = 1.0f / (float)((numSamples - numPositiveSamples) * 2);
			}

			for (int wl = 0; wl < _numWeakLearners; wl++)
			{
				//train a weak learner with the sample weights.
				WeakLearner* weakLearner = new T();
				weakLearner->train(samples, w, k);

				Accumulator errorSum;
				for (int i = 0; i < numSamples; i++)
				{
					Sample* x = samples[i];
					computedLabels[i] = weakLearner->label(x);
					if (computedLabels[i] * binaryLabel(x->y(), k) <= 0.0f)
					{
						errorSum += w[i];
					}
				}

				//compute the AdaBoost ensemble weight.
				float alpha = log((1.0f - fmax(errorSum.sum(), 1e-9f)) / fmax(errorSum.sum(), 1e-9f));
				_ensembles[k]->addWeakLearner(weakLearner, alpha);

				//recalculate the sample weights.
				Accumulator weightSum;
				for (int i = 0; i < numSamples; i++)
				{
					Sample* x = samples[i];
					float wFactor = exp(-alpha * computedLabels[i] * binaryLabel(x->y(), k));
					w[i] = w[i] * wFactor;
					weightSum += w[i];
				}

				for (int i = 0; i < numSamples; i++)
				{
					w[i] /= weightSum.sum();
				}
			}
		}
	}

	/*
	Get the total number of unique classes in a sample set. 
	*/
	int getNumClasses(std::vector<Sample*>& samples)
	{
		int maxClassIndex = 0;
		for (int i = 0; i < samples.size(); i++)
		{
			maxClassIndex = fmax(maxClassIndex, samples[i]->y());
		}
		return maxClassIndex + 1;
	}

	float binaryLabel(int class0, int class1)
	{
		if (class0 == class1)
			return 1.0f;
		return -1.0f;
	}

	Ensemble** _ensembles;

	int _numWeakLearners;
	int _n;
	int _k;
};