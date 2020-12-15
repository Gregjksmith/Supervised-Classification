#include <LogisticRegression.h>
#include <Accumulator.h>

LogisticRegression::LogisticRegression() : WeakLearner()
{
	_w = nullptr;
	_b = 0.0f;
}

LogisticRegression::~LogisticRegression()
{
	delete[] _w;
}

float LogisticRegression::label(Sample* x)
{
	//ensure the label is between [-1,1].
	return 2.0f * sigmoid(x) - 1.0f;
}

void LogisticRegression::train(std::vector<Sample*>& samples, float* sampleWeights, int classIndex)
{
	if (samples.size() <= 0)
		return;
	
	int numSamples = samples.size();
	int vectorSize = samples[0]->n();
	float batchSize = (float)numSamples;

	//create the weight and bias buffers.
	if (_w != nullptr)
		delete[] _w;
	_w = new float[vectorSize];
	for (int i = 0; i < vectorSize; i++)
		_w[i] = 0.0f;
	_b = 0.0f;
	_sampleSize = vectorSize;

	//create the weight and bias buffers.
	Accumulator* weight = new Accumulator[vectorSize];
	Accumulator bias;

	//create the gradient buffers.
	Accumulator* weightGradient = new Accumulator[vectorSize];
	Accumulator biasGradient;

	//create the weight root mean squared buffers.
	Accumulator* weightRms = new Accumulator[vectorSize];
	Accumulator biasRms;

	float averageGradient = 0.0f;
	for (int iter = 0; iter < MAX_EPOCHS; iter++)
	{
		Accumulator cost;
		for (int i = 0; i < vectorSize; i++)
		{
			weightGradient[i].clear();
			weightRms[i].clear();
		}
		biasGradient.clear();
		biasRms.clear();

		Accumulator weightSum;
		Accumulator dSum;

		for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++)
		{
			Sample* sample = samples[sampleIndex];
			float sig = sigmoid(sample);
			float y = sigmoidLabel(sample->y(), classIndex);

			//compute the cross entropy loss.
			cost += -(y * log(sig + 1e-9f) + (1.0f - y) * log(1.0f - sig + 1e-9f));
			weightSum += sampleWeights[sampleIndex];

			//compute and accumulate the weight gradient of the sample.
			for (int i = 0; i < vectorSize; i++)
			{
				float d = sampleWeights[sampleIndex] * (sig - y) * sample->x(i);
				dSum += fabs(d);
				weightGradient[i] += d;
			}
			//copmute and accumulate the bias gradient of the sample.
			float d = sampleWeights[sampleIndex] * (sig - y);
			biasGradient += d;
			dSum += fabs(d);
		}

		dSum = dSum.sum() / (weightSum.sum() * (float)(vectorSize + 1));

		//normalize the gradient.
		for (int j = 0; j < vectorSize; j++)
			weightGradient[j] = weightGradient[j].sum() / weightSum.sum();
		biasGradient = biasGradient.sum() / weightSum.sum();

		//get the running gradient mean square.
		for (int j = 0; j < vectorSize; j++)
			weightRms[j] = RUNNING_AVERAGE_WEIGHT * weightRms[j].sum() + (1.0f - RUNNING_AVERAGE_WEIGHT) * pow(weightGradient[j].sum(), 2.0f);	
		biasRms = RUNNING_AVERAGE_WEIGHT * biasRms.sum() + (1.0f - RUNNING_AVERAGE_WEIGHT) * pow(biasGradient.sum(), 2.0f);

		//update the weights and bias with the adaptive learning rate
		for (int j = 0; j < vectorSize; j++)
		{
			float scale = 1.0f / (sqrt(weightRms[j].sum()) + 1e-5f);
			weight[j] -= LEARNING_RATE * weightGradient[j].sum() / (sqrt(weightRms[j].sum()) + 1e-5f);
			_w[j] = weight[j].sum();
		}
		bias -= LEARNING_RATE * biasGradient.sum() / (sqrt(biasRms.sum()) + 1e-5f);
		_b = bias.sum();

		//if the gradient is small enough, the loss is at an optimal value.
		if (dSum.sum() < GRADIENT_THRESH)
			break;
	}

	delete[] weightRms;
	delete[] weightGradient;
	delete[] weight;
}

void LogisticRegression::exportInternal(std::string& params)
{
	params += std::to_string(_sampleSize) + WEAK_LEARNER_DELIM;

	for (int i = 0; i < _sampleSize; i++)
		params += std::to_string(_w[i]) + WEAK_LEARNER_DELIM;
	
	params += std::to_string(_b) + WEAK_LEARNER_DELIM;
}
void LogisticRegression::importInternal(std::string& params)
{
	_sampleSize = atoi(getNextParam(params, WEAK_LEARNER_DELIM).c_str());

	_w = new float[_sampleSize];
	for (int i = 0; i < _sampleSize; i++)
	{
		_w[i] = atof(getNextParam(params, WEAK_LEARNER_DELIM).c_str());
	}
	_b = atof(getNextParam(params, WEAK_LEARNER_DELIM).c_str());
}

float LogisticRegression::sigmoid(Sample* s)
{
	float z = 0.0f;
	for (int i = 0; i < s->n(); i++)
	{
		z += s->x(i) * _w[i];
	}
	z += _b;
	return (1.0f / (1.0f + exp(-z)));
}

void LogisticRegression::clearBuffer(float* buffer, int numSamples)
{
	for (int i = 0; i < numSamples; i++)
		buffer[i] = 0.0f;
}

float LogisticRegression::sigmoidLabel(int class0, int class1)
{
	if (class0 == class1)
		return 1.0f;
	return 0.0f;
}