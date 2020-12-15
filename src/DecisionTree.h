/*
DecisionTree.h
Computes a decision tree on a training set.
Classifies a sample recursively using a set of optimum decision boundaries.
The DecitionTree is trained using the ID3 algorithm.

Greg Smith
gregjksmith@gmail.com
*/

#pragma once
#include <WeakLearner.h>

//number of bins used to compute the attribute histograms.
#define NUM_BINS 25

class DecisionTree : public WeakLearner
{
public:
	DecisionTree();
	~DecisionTree();

	virtual float label(Sample* x);
	virtual void train(std::vector<Sample*>& samples, float* sampleWeights, int classIndex);

protected:
	virtual void exportInternal(std::string& params);
	virtual void importInternal(std::string& params);

private:

	int _splitAttributeIndex;	//attribute index in which the decision tree is split.
	float _splitThresh;		//attribute threshold in which the decision tree is split.
	float _nodeLabel;	//label of node.
	DecisionTree* _childNode[2];	//split nodes. If the nodes are null this node is a leaf.

	double _positiveHistogram[NUM_BINS];
	double _negativeHistogram[NUM_BINS];

	/*
	returns true if all samples in a training set are the same class.
	*/
	bool isSameClass(std::vector<Sample*>& samples, int classIndex, float& majorityClass);
	
	/*
	computes the classification threshold of the samples given an attribute index.
	The mean attribute value is calcualted for positive samples and negative samples.
	The threshold is the midpoint between the two means.
	*/
	float threshold(std::vector<Sample*>& samples, float* sampleWeights, int classIndex, int attributeIndex);

	/*
	Computes the entropy of the sample labels.
	entropy = -p(y = 0) * log2(p(y = 0)) - -p(y = 1) * log2(p(y = 1))
	*/
	float entropy(std::vector<Sample*>& samples, float* sampleWeights, int classIndex);
	
	/*
	Computes the information gain of a set of samples.
	The information gain of a set of samples, given an attribute is the entropy of the
	sample set minus the entropy of a sample set given an set attribute.
	IG(T,a) = H(T) - H(T,a).
	*/
	float informationGain(std::vector<Sample*>& samples, float* sampleWeights, int classIndex, int attributeIndex);
};