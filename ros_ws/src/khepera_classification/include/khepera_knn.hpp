#pragma once

#include <vector>
#include <memory>
#include <utility>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <map>
#include <algorithm>

namespace knn {

    ///*** TEMPLATE CLASS DECLARATION ***///
    template <typename T> using Vector = std::vector<T>;
    template <typename T> using Matrix = Vector<Vector<T>>;

    template <typename TX, typename Ty> class KnnClassifier {
    private:
        // Dataset
        Matrix<TX> data;
        Vector<Ty> labels;
        Vector<Ty> label_list;
        void computeLabelList();

        // Test data
        Matrix<TX> testset;
        Vector<Ty> ytest;
        Matrix<double> scores;

        void computeDistances(Vector<TX>& sample, Vector<double>& distances);
        void keepKSmallestDistances(Vector<double>& distances, Vector<unsigned int>& kIndex, unsigned int k);
        void selectMostRepresentedCluster(Vector<unsigned int>& kIndex, Ty& cluster, Vector<double>& score);

    public:
        // Constructor, called by friend constructor method
        KnnClassifier();
        KnnClassifier(Matrix<TX>& dataset, Vector<Ty>& y);
        KnnClassifier(Matrix<TX>&& dataset, Vector<Ty>&& y);
        KnnClassifier(std::initializer_list<std::initializer_list<TX>> X_list, std::initializer_list<Ty> y_list);

        KnnClassifier(const KnnClassifier& cpy)            = delete;
        KnnClassifier(KnnClassifier&& cpy)                 = delete;
        KnnClassifier& operator=(const KnnClassifier& cpy) = delete;
        KnnClassifier& operator=(KnnClassifier&& cpy)      = delete;
        ~KnnClassifier();

        unsigned int featuresDimension();
        unsigned int nbSamples();

        void fit(Matrix<TX>&  dataset, Vector<Ty>&  y);
        void fit(Matrix<TX>&& dataset, Vector<Ty>&& y);
        void printDataset();

        void predict(Matrix<TX>& testset_, unsigned int k);
        void predict(std::initializer_list<std::initializer_list<TX>> testlist, unsigned int k);
        void printPrediction();
        Vector<Ty>& getPrediction();
    };

    ///*** TEMPLATE CLASS IMPLEMENTATION ***///
    template <typename TX, typename Ty>
    KnnClassifier<TX, Ty>::KnnClassifier()
        : data(), labels() {}

    template <typename TX, typename Ty>
    KnnClassifier<TX, Ty>::KnnClassifier(Matrix<TX>& X, Vector<Ty>& y)
        : data(X), labels(y) {
        assert(X.size()>0 && X.size()==y.size());
        computeLabelList();
    }

    template <typename TX, typename Ty>
    KnnClassifier<TX, Ty>::KnnClassifier(Matrix<TX>&& X, Vector<Ty>&& y)
        : data(std::move(X)), labels(std::move(y)) {
        assert(X.size()>0 && X.size()==y.size());
        computeLabelList();
    }

    template <typename TX, typename Ty>
    KnnClassifier<TX, Ty>::KnnClassifier(std::initializer_list<std::initializer_list<TX>> X_list, std::initializer_list<Ty> y_list) {
        assert(X_list.size() > 0 && X_list.size() == y_list.size());
        data.resize(X_list.size());
        labels.resize(y_list.size());
        auto itSamples = X_list.begin();
        auto itLabels = y_list.begin();
        for (unsigned int i=0; i<data.size(); ++i) {
            labels[i] = *(itLabels++);
            data[i].resize((*itSamples).size());
            auto itSample = (*itSamples).begin();
            for (unsigned int j=0; j<(*itSamples).size(); ++j) {
                data[i][j] = *(itSample++);
            }
            itSamples++;
        }
        computeLabelList();
    }

    template <typename TX, typename Ty>
    KnnClassifier<TX, Ty>::~KnnClassifier() {}

    template <typename TX, typename Ty>
    void KnnClassifier<TX, Ty>::computeLabelList() {
        std::unordered_set<Ty> clusterSet;
        for (auto labelIt=labels.begin(); labelIt!=labels.end(); ++labelIt) {
            clusterSet.insert(*labelIt);
        }
        for (auto clusterIt=clusterSet.begin(); clusterIt!=clusterSet.end(); ++clusterIt) {
            label_list.push_back(*clusterIt);
        }
    }

    template <typename TX, typename Ty>
    unsigned int KnnClassifier<TX, Ty>::featuresDimension() {
        return data[0].size();
    }

    template <typename TX, typename Ty>
    unsigned int KnnClassifier<TX, Ty>::nbSamples() {
        return data.size();
    }

    template <typename TX, typename Ty>
    void KnnClassifier<TX, Ty>::fit(Matrix<TX>& X, Vector<Ty>& y) {
        data = X;
        labels = y;
        assert(X.size()>0 && X.size()==y.size());
        computeLabelList();
    }

    template <typename TX, typename Ty>
    void KnnClassifier<TX, Ty>::fit(Matrix<TX>&& X, Vector<Ty>&& y) {
        data = std::move(X);
        labels = std::move(y);
        assert(X.size()>0 && X.size()==y.size());
        computeLabelList();
    }

    template <typename TX, typename Ty>
    void KnnClassifier<TX, Ty>::printDataset() {
        for (unsigned int sample=0; sample<nbSamples(); ++sample) {
            std::cout << "Sample " << sample << ":" << std::endl << "x = ( ";
            for (unsigned int attr=0; attr<featuresDimension(); ++attr) {
                std::cout << data[sample][attr] << " ";
            }
            std::cout << ')' << std::endl;
            std::cout << "y = " << labels[sample] << std::endl << std::endl;
        }
    }

    ///*** KNN ALGORITHM ***///
    template <typename TX, typename Ty>
    void KnnClassifier<TX, Ty>::predict(Matrix<TX>& testset_, unsigned int k) {
        assert(k <= nbSamples());
        assert(testset_.size()>0 && testset_[0].size() == featuresDimension());
        testset = testset_;
        ytest.resize(testset.size());
        scores.resize(testset.size());
        for (unsigned int s=0; s<testset.size(); ++s) {
            Vector<TX>& sample = testset[s];
            assert(sample.size() == data[0].size());
            // Compute distances between tested sample and every sample of dataset
            Vector<double> distances;
            computeDistances(sample, distances);
            // Retrieve index of k samples with minimal distances to tested sample
            Vector<unsigned int> kIndex;
            keepKSmallestDistances(distances, kIndex, k);
            // Get most represented cluster among the K selected vectors
            selectMostRepresentedCluster(kIndex, ytest[s], scores[s]);
        }
    }

    template <typename TX, typename Ty>
    void KnnClassifier<TX, Ty>::predict(std::initializer_list<std::initializer_list<TX>> testlist, unsigned int k) {
        Matrix<TX> testset;
        testset.resize(testlist.size());
        auto itSamples = testlist.begin();
        for (unsigned int i=0; i<data.size(); ++i) {
            testset[i].resize((*itSamples).size());
            auto itSample = (*itSamples).begin();
            for (unsigned int j=0; j<(*itSamples).size(); ++j) {
                testset[i][j] = *(itSample++);
            }
            itSamples++;
        }
        predict(testset, k);
    }

    template <typename TX, typename Ty>
    void KnnClassifier<TX, Ty>::computeDistances(Vector<TX>& sample, Vector<double>& distances) {
        distances.resize(nbSamples());
        for (unsigned int s=0; s<nbSamples(); ++s) {
            double dist = 0;
            for (unsigned int feat=0; feat<featuresDimension(); ++feat) {
                dist += std::abs(data[s][feat] - sample[feat]);
            }
            distances[s] = dist;
        }
    }

    template <typename TX, typename Ty>
    void KnnClassifier<TX, Ty>::keepKSmallestDistances(Vector<double>& distances, Vector<unsigned int>& kIndex, unsigned int k) {
        kIndex.resize(nbSamples());
        // Generate vector of index 0... nbSamples()-1
        std::iota(kIndex.begin(), kIndex.end(), 0);
        // Sort index vector by comparing elements of distances vector
        std::sort(kIndex.begin(), kIndex.end(),
                  [&distances](unsigned int i1, unsigned int i2) {
                      return distances[i1] < distances[i2];
                  });
        // Crop sorted index vector to k minimum index
        kIndex.resize(k);
    }

    template <typename TX, typename Ty>
    void KnnClassifier<TX, Ty>::selectMostRepresentedCluster(Vector<unsigned int>& kIndex, Ty& cluster, Vector<double>& score) {
        score.resize(label_list.size());
        std::map<Ty, unsigned int> clusterStats;
        for (Ty label : label_list) {
            clusterStats[label]=0;
        }
        for (unsigned int index : kIndex) {
            clusterStats[labels[index]]++;
        }
        double maxScore = -1;
        for (unsigned int c=0; c<label_list.size(); ++c) {
            score[c] = ((double)clusterStats[label_list[c]]) / ((double)kIndex.size());
            if (score[c] > maxScore) {
                cluster = label_list[c];
                maxScore = score[c];
            }
        }
    }

    template <typename TX, typename Ty>
    void KnnClassifier<TX, Ty>::printPrediction() {
        for (unsigned int s=0; s<testset.size(); ++s) {
            std::cout << "Tested sample:" << std::endl << "x = ( ";
            for (unsigned int attr=0; attr<featuresDimension(); ++attr) {
                std::cout << testset[s][attr] << " ";
            }
            std::cout << ')' << std::endl << "Predicted label:" << std::endl << "y = ";
            std::cout << ytest[s] << std::endl << "Scores are:" << std::endl;
            for (unsigned int l=0; l<label_list.size(); ++l) {
                std::cout << "P(y = " << label_list[l] << ") = " << scores[s][l] << std::endl;
            }
            std::cout << std::endl;
        }
    }

    template <typename TX, typename Ty>
    Vector<Ty>& KnnClassifier<TX, Ty>::getPrediction() {
        return ytest;
    }
}
