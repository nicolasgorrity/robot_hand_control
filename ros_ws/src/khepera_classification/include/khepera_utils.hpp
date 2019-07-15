#pragma once

#include <vector>

namespace utils {

    template<typename T>
    T mean(const std::vector<T>& vect) {
        T sum {};
        for (auto it=vect.begin(); it!=vect.end(); ++it) {
            sum += *it;
        }
        return sum / (T)vect.size();
    }

    template<typename T>
    void substract(std::vector<T>& vect, T value) {
        for (auto it=vect.begin(); it!=vect.end(); ++it) {
            (*it) -= value;
        }
    }

    template<typename T>
    void multiply(std::vector<T>& vect, T value) {
        for (auto it=vect.begin(); it!=vect.end(); ++it) {
            (*it) *= value;
        }
    }

}
