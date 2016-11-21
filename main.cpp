/*******************************************************************
*   main.cpp
*   K2NN
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Sep 12, 2016
*******************************************************************/
//
// Fastest CPU implementation of both a brute-force
// and a custom Multi-Index Hash Table accelerator
// system for matching 512-bit binary descriptors
// in 2NN mode, i.e., a match is returned if the best
// match between a query vector and a training vector
// is more than a certain threshold number of bits
// better than the second-best match.
//
// Yes, that means the DIFFERENCE in popcounts is used
// for thresholding, NOT the ratio. This is the CORRECT
// approach for binary descriptors.
//
// Both 8-bit and 16-bit MIH tables are supported.
// I currently recommend 16-bit.
//
// All functionality is contained in the files K2NN.h and twiddle_table.h.
// 'main.cpp' is simply a sample test harness with example usage and
// performance testing.
//
// Example initialization of Matcher class
// Matcher<false> m(tvecs, size, qvecs, size, threshold, max_twiddles);
//
// Options:
//
// Brute-force complete (exact) match:
// m.bruteMatch();
//
// Single twiddle pass for a very fast partial match,
// with no false positives (i.e. if a match is returned, it's truly the best match):
// m.fastApproxMatch();
//
// Multi-index hash (MIH) complete (exact) match, with fall-back to brute force after max_twiddles passes:
// m.exactMatch();
//
// Match until complete or until 'n' passes elapse (partial):
// m.approxMatchToNTwiddles(n);
//
// Afterward, the finalized matches are waiting
// in the vector 'm.matches'.
//

#include "K2NN.h"

#include <chrono>
#include <iostream>
#include <vector>

using namespace std::chrono;

int main() {
	// ------------- Configuration ------------
	constexpr int warmups = 10;
	constexpr int runs = 25;
	constexpr int size = 10000;
	constexpr int threshold = 5;
	constexpr int max_twiddles = 2;
	// --------------------------------


	// ------------- Generation of Random Data ------------
	// obviously, this is not representative of real data;
	// it doesn't matter for brute-force matching
	// but the MIH methods will be much faster
	// on real data
	void *qvecs = malloc(64 * size), *tvecs = malloc(64 * size);
	srand(36);
	for (int i = 0; i < 64 * size; ++i) {
		reinterpret_cast<uint8_t*>(qvecs)[i] = static_cast<uint8_t>(rand());
		reinterpret_cast<uint8_t*>(tvecs)[i] = static_cast<uint8_t>(rand());
	}
	// --------------------------------

	// Initialization of Matcher class
	Matcher<false> m(tvecs, size, qvecs, size, threshold, max_twiddles);

	std::cout << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) m.fastApproxMatch();
	std::cout << "Testing..." << std::endl;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) m.fastApproxMatch();
	high_resolution_clock::time_point end = high_resolution_clock::now();

	const double sec = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(runs);
	std::cout << std::endl << "Brute force K2NN found " << m.matches.size() << " matches in " << sec * 1e3 << " ms" << std::endl;
	std::cout << "Throughput: " << static_cast<double>(size)*static_cast<double>(size) / sec * 1e-9 << " billion comparisons/second." << std::endl << std::endl;
}
