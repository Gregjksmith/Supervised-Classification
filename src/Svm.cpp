#include <svm.h>

Svm::Svm() : WeakLearner()
{
	_w = nullptr;
	_b = 0.0f;
	_n = 0;
}

Svm::~Svm()
{
	delete[] _w;
}

void Svm::train(std::vector<Sample*>& samples, float* sampleWeights, int classIndex)
{
	int numSamples = samples.size();

	//get the number of vector samples.
	_n = samples[0]->n();

	//init the plane normal.
	if (_w != nullptr)
		delete[] _w;

	_w = new float[_n];
	for (int n = 0; n < _n; n++)
	{
		_w[n] = 0.0f;
	}

	//init the plane offset
	_b = 0.0f;

	float* alpha = new float[numSamples];

	for (int i = 0; i < numSamples; i++)
		alpha[i] = 0.0f;

	int numPasses = 0;
	//iterate until the training weights are unchanged for a certain number of iterations.
	while (numPasses < MAX_PASSES)
	{
		bool alphaModified = false;
		for (int i = 0; i < numSamples; i++)
		{
			//compute the boundary hyperparmeter given the sample weights.
			float slackTolerance = C * sampleWeights[i] * (float)numSamples;
			float a1 = alpha[i];
			Sample* xi = samples[i];
			float g1 = label(samples[i]);

			//check the sample on the KKT conditions. All samples must pass the KKT
			//conditions for dual / primal optimality.
			bool kktConditions = false;
			kktConditions |= (a1 == 0.0f && binaryLabel(xi->y(), classIndex) * g1 >= 1.0f);
			kktConditions |= (a1 == slackTolerance && binaryLabel(xi->y(), classIndex) * g1 <= 1.0f);
			kktConditions |= (a1 > 0.0f && a1 < slackTolerance && fabs(binaryLabel(xi->y(), classIndex) * g1 - 1.0f) < EQUALITY_TOLERANCE);

			//if the kkt conditions are not met.
			if (!kktConditions)
			{
				//randomly take another sample. And compute the optimal
				//parameters in a greedy manner.
				int j;
				while (true)
				{
					j = rand() % numSamples;
					if (i != j)
						break;
				}

				Sample* xj = samples[j];
				float a2 = alpha[j];
				float g2 = label(xj);

				float a1Old = a1;
				float a2Old = a2;

				//compute the optimal parameters.
				float L, H;
				if (binaryLabel(xi->y(), classIndex) * binaryLabel(xj->y(), classIndex) < 0.0f)
				{
					L = fmax(0.0f, a2Old - a1Old);
					H = fmin(slackTolerance, slackTolerance + a2Old - a1Old);
				}
				else
				{
					L = fmax(0.0f, a1Old + a2Old - slackTolerance);
					H = fmin(slackTolerance, a1Old + a2Old);
				}

				float e1 = g1 - (float)binaryLabel(xi->y(), classIndex);
				float e2 = g2 - (float)binaryLabel(xj->y(), classIndex);

				float n = 2.0f * innerProduct(xi, xj) - innerProduct(xi, xi) - innerProduct(xj, xj);
				if (n == 0.0f)
					break;

				a2 = a2Old - binaryLabel(xj->y(), classIndex) * (e1 - e2) / n;

				a2 = fmin(a2, H);
				a2 = fmax(a2, L);

				a1 = a1Old + (binaryLabel(xi->y(), classIndex) * binaryLabel(xj->y(), classIndex)) * (a2Old - a2);

				alpha[i] = a1;
				alpha[j] = a2;

				//compute w
				for (int vIndex = 0; vIndex < _n; vIndex++)
					_w[vIndex] = 0.0f;

				//recompute the hyperplane normal.
				for (int i = 0; i < numSamples; i++)
				{
					for (int vIndex = 0; vIndex < _n; vIndex++)
					{
						_w[vIndex] += alpha[i] * (float)binaryLabel(samples[i]->y(), classIndex) * samples[i]->x(vIndex);
					}
				}

				//recompute the hyperplane bias.
				float b1 = _b - e1 - binaryLabel(xi->y(), classIndex) * (a1 - a1Old) * innerProduct(xi, xi) - binaryLabel(xj->y(), classIndex) * (a2 - a2Old) * innerProduct(xi, xj);
				float b2 = _b - e2 - binaryLabel(xi->y(), classIndex) + (a1 - a1Old) * innerProduct(xi, xj) - binaryLabel(xj->y(), classIndex) * (a2 * a2Old) * innerProduct(xj, xj);

				if (a1 > 0.0f && a1 < slackTolerance)
				{
					_b = b1;
				}
				else if (a2 > 0.0f && a2 < slackTolerance)
				{
					_b = b2;
				}
				else
				{
					_b = (b1 + b2) / 2.0f;
				}

				alphaModified = true;
			}
		}

		if (!alphaModified) //if no training parmaters are changed in a pass.
			numPasses++;
		else
			numPasses = 0;
	}
	
	delete[] alpha;
}

/*
Computes the inner product / dot product between two samples x0 and x1.
*/
float Svm::innerProduct(Sample* x0, Sample* x1)
{
	float dp = 0.0f;
	for (int i = 0; i < x0->n(); i++)
	{
		dp += x0->x(i) * x1->x(i);
	}
	return dp;
}

float Svm::label(Sample* x)
{
	float sum = 0.0f;
	for (int i = 0; i < _n; i++)
	{
		sum += x->x(i) * _w[i];
	}
	sum += _b;
	return sum;
}