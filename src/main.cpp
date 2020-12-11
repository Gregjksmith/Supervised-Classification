/*
Training Examples:

Greg Smith
gregjksmith@gmail.com
*/

#include <stdio.h>
#include <vector>
#include <random>
#include <Sample.h>
#include <AdaBoost.h>
#include <LogisticRegression.h>
#include <DecisionTree.h>
#include <NaiveBayes.h>
#include <Svm.h>

float gaussianRV(float var)
{
	float xMax = sqrt(9.21f * var);
	
	while (true)
	{
		float xSample = (rand() / (float)RAND_MAX - 0.5f) * 2.0f * xMax;
		float p = exp(-pow(xSample, 2.0f) / (2.0f * var));
		float a = rand() / (float)RAND_MAX;
		if (a < p)
			return xSample;
	}
}

void computeRandomTrainingSet(std::vector<Sample*>& samples, int attributeSize, int numSamples, float var)
{
	float* x = new float[attributeSize];
	for (int i = 0; i < numSamples; i++)
	{
		int y = rand() % attributeSize;
		for (int j = 0; j < attributeSize; j++)
		{
			if (j == y)
				x[j] = 1.0f;
			else
				x[j] = -1.0f;
			
			float rv = gaussianRV(var);
			x[j] += rv;
		}

		Sample* s = new Sample(x, y, attributeSize);
		samples.push_back(s);
	}
	delete[] x;
}

void main()
{
	std::vector<Sample*> samples;
	const int numSamples = 1000;
	const int numClasses = 10;
	const float sampleVariance = 0.125f;
	const int numWeakLearners[] = { 1,2,5 };
	computeRandomTrainingSet(samples, numClasses, numSamples, sampleVariance);


	printf("Training set, num samples: %i, num classes: %i, sample variance: %0.2f\n", numSamples, numClasses, sampleVariance);
	
	//test Naive Bayes
	for (int i = 0; i < 3; i++)
	{
		auto naiveBayes = new AdaBoost<NaiveBayes>(samples, numWeakLearners[i]);
		float error = naiveBayes->error(samples);

		printf("Training Naive Bayes: Weak Learners: %i, Classification Error %0.6f\n", numWeakLearners[i], error);

		delete naiveBayes;
	}

		
	//test Decision Tree
	for (int i = 0; i < 3; i++)
	{
		auto decisionTree = new AdaBoost<DecisionTree>(samples, numWeakLearners[i]);
		float error = decisionTree->error(samples);

		printf("Training Decision Tree: Weak Learners: %i, Classification Error %0.6f\n", numWeakLearners[i], error);

		delete decisionTree;
	}

	//test Logistic Regression
	for (int i = 0; i < 3; i++)
	{
		auto logisticRegression = new AdaBoost<LogisticRegression>(samples, numWeakLearners[i]);
		float error = logisticRegression->error(samples);

		printf("Training Logistic Regression: Weak Learners: %i, Classification Error %0.6f\n", numWeakLearners[i], error);

		delete logisticRegression;
	}

	//test SVM
	for (int i = 0; i < 3; i++)
	{
		auto svm = new AdaBoost<LogisticRegression>(samples, numWeakLearners[i]);
		float error = svm->error(samples);

		printf("Training SVM: Weak Learners: %i, Classification Error %0.6f\n", numWeakLearners[i], error);

		delete svm;
	}

	system("pause");
}