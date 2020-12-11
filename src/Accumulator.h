#pragma once

class Accumulator;

class Accumulator
{
public:
	Accumulator()
	{
		_sum = 0.0f;
		_residual = 0.0f;
	}
	~Accumulator()
	{

	}

	float sum()
	{
		return _sum;
	}

	void clear()
	{
		_sum = 0.0f;
		_residual = 0.0f;
	}

	Accumulator operator=(const float rhs)
	{
		_sum = rhs;
		_residual = 0.0f;
		return *this;
	}

	Accumulator operator=(const Accumulator& rhs)
	{
		_sum = rhs._sum;
		_residual = rhs._residual;
		return *this;
	}

	Accumulator operator+(const float rhs)
	{
		Accumulator acc = *this;

		float y = rhs - _residual;
		float t = _sum + y;
		acc._residual = (t - _sum) - y;
		acc._sum = t;

		return acc;
	}

	Accumulator& operator-(const float rhs)
	{
		return (*this) - rhs;
	}

	Accumulator& operator+=(const float rhs)
	{
		float y = rhs - _residual;
		float t = _sum + y;
		_residual = (t - _sum) - y;
		_sum = t;

		return *this;
	}

	Accumulator& operator-=(const float rhs)
	{
		return (*this) += -rhs;
	}

private:
	float _sum;
	float _residual;
};