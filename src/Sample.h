/*
Sample.H
Structure for containing a training set sample.
consists of the input data x, and the associated integer label y.

Greg Smith
gregjksmith@gmail.com
*/
#pragma once

class Sample
{
public:
	/*
	Constructor
	float* x: input data.
	int y: integer label.
	int n: vector size of the input data.
	*/
	Sample(float* x, int y, int n)
	{
		if (n > 0)
		{
			_n = n;
			_x = new float[n];
			for (int i = 0; i < n; i++)
				_x[i] = x[i];

			_y = y;
		}
		else
		{
			_n = 0.0f;
			_x = nullptr;
			_y = 0.0f;
		}
	}
	virtual ~Sample()
	{
		delete[] _x;
	}

	/*
	Getters.
	*/
	float x(int i)
	{
		return _x[i];
	}
	int n()
	{
		return _n;
	}
	int y()
	{
		return _y;
	}

private:
	float* _x;
	int _n;
	int _y;
};