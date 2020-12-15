#pragma once

#include <string>

#define WEAK_LEARNER_DELIM ','
#define ENSEMBLE_DELIM ';'

inline std::string getNextParam(std::string& val, const char delim)
{
	int offset = 0;
	int findIndex = val.find_first_of(delim);
	std::string parsed = val.substr(offset, findIndex);

	val = val.substr(findIndex + 1, val.size() - findIndex);
	return parsed;
}