#include <DecisionTree.h>

DecisionTree::DecisionTree()
{
	_splitAttributeIndex = 0;
	_splitThresh = 0.0f;
	_nodeLabel = 0.0f;
	_childNode[0] = nullptr;
	_childNode[1] = nullptr;
}
DecisionTree::~DecisionTree()
{
	delete _childNode[0];
	delete _childNode[1];
}

float DecisionTree::label(Sample* x)
{
	if (_childNode[0] == nullptr && _childNode[1] == nullptr)
		return _nodeLabel;

	if (x->x(_splitAttributeIndex) > _splitThresh)
		return _childNode[1]->label(x);
	else
		return _childNode[0]->label(x);
}

void DecisionTree::train(std::vector<Sample*>& samples, float* sampleWeights, int classIndex)
{
	if (samples.size() == 0)
		return;

	//if all the samples are part of the sample class, the node is a leaf.
	if (isSameClass(samples, classIndex, _nodeLabel))	
		return;


	int numAttributes = samples[0]->n();
	

	//compute the information gain of each attribute, get the maximum.
	float maxInformationGain = informationGain(samples, sampleWeights, classIndex, 0);
	int maxAttributeIndex = 0;
	for (int i = 1; i < numAttributes; i++)
	{
		float ig = informationGain(samples, sampleWeights, classIndex, i);
		if (ig > maxInformationGain)
		{
			maxInformationGain = ig;
			maxAttributeIndex = i;
		}
	}

	//the attribute with the maximum information gain is the split attribute.
	_splitAttributeIndex = maxAttributeIndex;
	//compute the split threshold.
	_splitThresh = threshold(samples, sampleWeights, classIndex, _splitAttributeIndex);

	//get the number of positive samples and negative samples given the split.
	int numPositiveSamples = 0;
	int numNegativeSamples = 0;
	for (int i = 0; i < samples.size(); i++)
	{
		if (samples[i]->x(_splitAttributeIndex) > _splitThresh)
			numPositiveSamples++;
		else
			numNegativeSamples++;
	}

	//if there is no split, the node is a leaf.
	if (numPositiveSamples == 0 || numNegativeSamples == 0)
		return;

	std::vector<Sample*> positiveSamples;
	std::vector<Sample*> negativeSamples;

	float* positiveSampleWeights = nullptr;
	float* negativeSampleWeights = nullptr;

	if (numPositiveSamples > 0)
		positiveSampleWeights = new float[numPositiveSamples];
	if (numNegativeSamples > 0)
		negativeSampleWeights = new float[numNegativeSamples];

	//iterate through the samples, compute the split samples and the corresponding
	//sample weights.
	for (int i = 0; i < samples.size(); i++)
	{
		if (samples[i]->x(_splitAttributeIndex) > _splitThresh)
		{
			positiveSampleWeights[positiveSamples.size()] = sampleWeights[i];
			positiveSamples.push_back(samples[i]);
		}
		else
		{
			negativeSampleWeights[negativeSamples.size()] = sampleWeights[i];
			negativeSamples.push_back(samples[i]);
		}
	}

	//create the leaf nodes.
	_childNode[0] = new DecisionTree();
	_childNode[0]->train(negativeSamples, negativeSampleWeights, classIndex);

	_childNode[1] = new DecisionTree();
	_childNode[1]->train(positiveSamples, positiveSampleWeights, classIndex);

	delete[] negativeSampleWeights;
	delete[] positiveSampleWeights;
}

/*
returns true if all samples in a training set are the same class.
*/
bool DecisionTree::isSameClass(std::vector<Sample*>& samples, int classIndex, float& majorityClass)
{
	if (samples.size() <= 0)
		return true;

	int positiveSamples = 0;
	int negativeSamples = 0;
	for (int i = 0; i < samples.size(); i++)
	{
		if (samples[i]->y() == classIndex)
		{
			positiveSamples++;
		}
		else
		{
			negativeSamples++;
		}
	}

	if (positiveSamples > negativeSamples)
		majorityClass = 1.0f;
	else
		majorityClass = -1.0f;

	if (positiveSamples == 0 || negativeSamples == 0)
		return true;
	else
		return false;
}

/*
computes the classification threshold of the samples given an attribute index.
The mean attribute value is calcualted for positive samples and negative samples.
The threshold is the midpoint between the two means.
*/
float DecisionTree::threshold(std::vector<Sample*>& samples, float* sampleWeights, int classIndex, int attributeIndex)
{
	float positiveMean = 0.0f;
	float positiveSum = 0.0f;

	float negativeMean = 0.0f;
	float negativeSum = 0.0f;

	//get the positive and negative means.
	for (int i = 0; i < samples.size(); i++)
	{
		if (samples[i]->y() == classIndex)
		{
			positiveMean += sampleWeights[i] * samples[i]->x(attributeIndex);
			positiveSum += sampleWeights[i];
		}
		else
		{
			negativeMean += sampleWeights[i] * samples[i]->x(attributeIndex);
			negativeSum += sampleWeights[i];
		}
	}

	positiveMean /= fmax(positiveSum, 1e-9f);
	negativeMean /= fmax(negativeSum, 1e-9f);

	//set the threshold to the midpoint of the means.
	return positiveMean * 0.5f + negativeMean * 0.5f;
}

/*
Computes the entropy of the sample labels.
entropy = -p(y = 0) * log2(p(y = 0)) - -p(y = 1) * log2(p(y = 1))
*/
float DecisionTree::entropy(std::vector<Sample*>& samples, float* sampleWeights, int classIndex)
{
	double positiveP = 0.0;
	double negativeP = 0.0;
	double weightSum = 0.0;

	for (int i = 0; i < samples.size(); i++)
	{
		if (samples[i]->y() == classIndex)
			positiveP += sampleWeights[i];
		
		else		
			negativeP += sampleWeights[i];
		
		weightSum += sampleWeights[i];
	}

	positiveP /= weightSum;
	negativeP /= weightSum;

	float entropy = -positiveP * log2(fmax(positiveP, 1e-12f)) - negativeP * log2(fmax(negativeP, 1e-12f));
	return entropy;
}

/*
Computes the information gain of a set of samples.
The information gain of a set of samples, given an attribute is the entropy of the
sample set minus the entropy of a sample set given an set attribute.
IG(T,a) = H(T) - H(T,a).
*/
float DecisionTree::informationGain(std::vector<Sample*>& samples, float* sampleWeights, int classIndex, int attributeIndex)
{
	//get the sample entropy.
	float sampleEntropy = entropy(samples, sampleWeights, classIndex);
	float ig = sampleEntropy;

	float minAttributeSample = samples[0]->x(attributeIndex);
	float maxAttributeSample = minAttributeSample;
	//get the attribute min / max
	for (int i = 1; i < samples.size(); i++)
	{
		float attributeSample = samples[i]->x(attributeIndex);
		minAttributeSample = fmin(attributeSample, minAttributeSample);
		maxAttributeSample = fmax(attributeSample, maxAttributeSample);
	}

	for (int i = 0; i < NUM_BINS; i++)
	{
		_positiveHistogram[i] = 0.0;
		_negativeHistogram[i] = 0.0;
	}

	//compute the histogram for the attribute.
	double weightSum = 0.0;
	for (int i = 0; i < samples.size(); i++)
	{
		float attributeSample = samples[i]->x(attributeIndex);

		double normalizedBinIndex = (double)NUM_BINS * (attributeSample - minAttributeSample) / fmax(maxAttributeSample - minAttributeSample, 1e-6);
		int binIndex = (int)floor(normalizedBinIndex);
		int nextBinIndex = binIndex + 1;
		double r = normalizedBinIndex - floor(normalizedBinIndex);

		if (binIndex >= 0 && binIndex < NUM_BINS)
		{
			if (samples[i]->y() == classIndex)
				_positiveHistogram[binIndex] += sampleWeights[i] * (1.0 - r);
			else
				_negativeHistogram[binIndex] += sampleWeights[i] *(1.0 - r);

			weightSum += sampleWeights[i] * (1.0 - r);
		}
		if (nextBinIndex >= 0 && nextBinIndex < NUM_BINS)
		{
			if (samples[i]->y() == classIndex)
				_positiveHistogram[nextBinIndex] += sampleWeights[i] * r;
			else
				_negativeHistogram[nextBinIndex] += sampleWeights[i] * r;

			weightSum += sampleWeights[i] * r;
		}
	}

	//add the conditional entropy.
	for (int i = 0; i < NUM_BINS; i++)
	{
		double p = _positiveHistogram[i];
		double n = _negativeHistogram[i];
		double attributeSum = p + n;

		p = p / fmax(attributeSum, 1e-9);
		n = 1.0 - p;

		float attributeEntropy = -p * log2(fmax(p, 1e-12)) + n * log2(fmax(n, 1e-12));
		
		ig -= (attributeSum / weightSum) * attributeEntropy;
	}
	return ig;
}

void DecisionTree::exportInternal(std::string& params)
{
	params += std::to_string(_splitAttributeIndex) + WEAK_LEARNER_DELIM;
	params += std::to_string(_splitThresh) + WEAK_LEARNER_DELIM;
	params += std::to_string(_nodeLabel) + WEAK_LEARNER_DELIM;

	if (_childNode[0] != nullptr)
		params += std::to_string(1) + WEAK_LEARNER_DELIM;
	else
		params += std::to_string(0) + WEAK_LEARNER_DELIM;

	if (_childNode[1] != nullptr)
		params += std::to_string(1) + WEAK_LEARNER_DELIM;
	else
		params += std::to_string(0) + WEAK_LEARNER_DELIM;

	if (_childNode[0] != nullptr)
		_childNode[0]->exportInternal(params);
	if(_childNode[1] != nullptr)
		_childNode[1]->exportInternal(params);
}
void DecisionTree::importInternal(std::string& params)
{
	_splitAttributeIndex = atoi(getNextParam(params, WEAK_LEARNER_DELIM).c_str());
	_splitThresh = atof(getNextParam(params, WEAK_LEARNER_DELIM).c_str());
	_nodeLabel = atof(getNextParam(params, WEAK_LEARNER_DELIM).c_str());

	int childNode0 = atoi(getNextParam(params, WEAK_LEARNER_DELIM).c_str());
	int childNode1 = atoi(getNextParam(params, WEAK_LEARNER_DELIM).c_str());

	if (childNode0 == 1)
	{
		_childNode[0] = new DecisionTree();
		_childNode[0]->importInternal(params);
	}

	if (childNode1 == 1)
	{
		_childNode[1] = new DecisionTree();
		_childNode[1]->importInternal(params);
	}
}