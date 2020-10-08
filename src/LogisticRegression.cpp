#include <LogisticRegression.h>

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
	clearBuffer(_w, vectorSize);
	_b = 0.0f;

	//create the weight root mean squared buffers.
	float* wRms = new float[vectorSize];
	clearBuffer(wRms, vectorSize);
	float bRms = 0.0f;

	//create the gradient buffers.
	float* dw = new float[vectorSize];
	float db = 0.0f;

	float cost = 0.0f;
	float averageGradient = 0.0f;
	for (int iter = 0; iter < MAX_EPOCHS; iter++)
	{
		cost = 0.0f;
		clearBuffer(dw, vectorSize);
		db = 0.0f;

		float dSum = 0.0f;
		float weightSum = 0.0f;
		for (int i = 0; i < numSamples; i++)
		{
			Sample* sample = samples[i];
			float sig = sigmoid(sample);
			float y = sigmoidLabel(sample->y(), classIndex);

			//compute the cross entropy loss.
			cost += -(y * log(sig + 1e-9f) + (1.0f - y) * log(1.0f - sig + 1e-9f));
			weightSum += sampleWeights[i];

			//compute and accumulate the weight gradient of the sample.
			for (int j = 0; j < vectorSize; j++)
			{
				float d = sampleWeights[i] * (sig - y) * sample->x(j);
				dSum += fabs(d);
				dw[j] += d;
			}
			//copmute and accumulate the bias gradient of the sample.
			db += sampleWeights[i] * (sig - y);
			dSum += fabs(db);
		}

		//normalize the gradient.
		for (int j = 0; j < vectorSize; j++)
			dw[j] /= weightSum;
		db /= weightSum;

		//compute the average absolute gradient.
		dSum /= weightSum;
		dSum /= (float)(vectorSize + 1); 

		//get the running gradient mean square.
		for (int j = 0; j < vectorSize; j++)
		{
			wRms[j] = (RUNNING_AVERAGE_WEIGHT)*wRms[j] + (1.0f - RUNNING_AVERAGE_WEIGHT) * (dw[j] * dw[j]);
		}
		bRms = RUNNING_AVERAGE_WEIGHT * bRms + (1.0f - RUNNING_AVERAGE_WEIGHT) * (db * db);

		//update the weights and bias with the adaptive learning rate
		for (int j = 0; j < vectorSize; j++)
		{
			float scale = 1.0f / (sqrt(wRms[j]) + 1e-5f);
			_w[j] -= LEARNING_RATE * dw[j] / (sqrt(wRms[j]) + 1e-5f);
		}
		_b -= LEARNING_RATE * db / (sqrt(bRms) + 1e-5f);

		//if the gradient is small enough, the loss is at an optimal value.
		if (dSum < GRADIENT_THRESH)
			break;
	}
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