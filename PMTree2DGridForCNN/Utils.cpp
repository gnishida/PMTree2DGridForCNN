#include "Utils.h"
#include <random>

namespace utils {

float uniform() {
	return (float)(rand() % 100) / 100.0f;
}

float uniform(float a, float b) {
	return uniform() * (b - a) + a;
}

}