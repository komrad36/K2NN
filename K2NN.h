/*******************************************************************
*   K2NN.h
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

#pragma once

#include "twiddle_table.h"

#include <future>
#include <nmmintrin.h>
#include <vector>

//#define DEBUG_PRINT

#ifdef DEBUG_PRINT
#include <iostream>
#endif

// Reports the successful match of a query vector,
// at index 'q' in the original list of query vectors,
// with the training vector at index 't' in the original
// list of training vectors.
struct Match {
	int q, t;

	Match() {}
	Match(const int _q, const int _t) : q(_q), t(_t) {}
};

// Holds an in-progress match that requires at least
// one more twiddle (or a brute-force) to resolve.
struct Partial {
	int q;
	int best_i;
	int16_t best_v;
	int16_t second_v;

	Partial() {}
	Partial(const int _q, const int _best_i, const int16_t _best_v, const int16_t _second_v) : q(_q), best_i(_best_i), best_v(_best_v), second_v(_second_v) {}
};

struct uninitialized_int {
	int x;
	uninitialized_int() {}
};

// Supports 8- and 16-bit hash tables, templated
// for performance.
template <const bool eightbit>
class Matcher {
public:
	// Vector of successful matches.
	std::vector<Match> matches;

	// Vector of indices of best-matching training vector for each query vector.
	std::vector<uninitialized_int> match_idxs;

	// Vector of in-progress matches requiring more twiddles
	// (or brute force) to resolve.
	std::vector<Partial> remainder;

private:
	// Contiguous MIHT
	std::vector<int> compact_table;

	// Incoming 512-bit training vectors
	const void* __restrict tset;

	// Number of training vectors
	int tcount;

	// Incoming 512-bit query vectors
	const void* __restrict qset;

	// Number of query vectors
	int qcount;

	// Threshold by which the best match
	// must exceed the second-best match
	// to be considered a match
	int threshold;

	// Max twiddle passes before the system
	// switches to the brute-force solver
	// on the remaining query vectors
	int max_twiddles;

	// Multi-index hash table (MIHT)
    std::vector<int>* const __restrict raw_table;

     // Array of end indices in compact_table for each bin
    int* const __restrict ends;

	int hw_concur;

	std::future<void>* const __restrict fut;

	std::vector<Partial>* const __restrict rems;

public:
	Matcher() :
		raw_table(new std::vector<int>[eightbit ? 256 * 64 : 65536 * 32]),
		ends(new int[(eightbit ? 256 * 64 : 65536 * 32) + 1]),
		hw_concur(static_cast<int>(std::thread::hardware_concurrency())),
		fut(new std::future<void>[hw_concur]),
		rems(new std::vector<Partial>[hw_concur]) {}

	Matcher(const void* const __restrict _tset, const int _tcount, const void* const __restrict _qset,
		const int _qcount, const int _threshold, const int _max_twiddles) :
		tset(_tset), tcount(_tcount), qset(_qset), qcount(_qcount), threshold(_threshold), max_twiddles(_max_twiddles),
		raw_table(new std::vector<int>[eightbit ? 256 * 64 : 65536 * 32]),
		ends(new int[(eightbit ? 256 * 64 : 65536 * 32) + 1]),
		hw_concur(static_cast<int>(std::thread::hardware_concurrency())),
		fut(new std::future<void>[hw_concur]),
		rems(new std::vector<Partial>[hw_concur]) {}

	~Matcher() {
		delete[] raw_table;
		delete[] ends;
		delete[] fut;
		delete[] rems;
	}

private:
	// 512-bit descriptors, 8- or 16-bit table entries.
	// This means 512/8 or 512/16 == 64 or 32 tables.
	// Each table contains 2^8 or 2^16 == 256 or 65536 entries.
	// All 64 or 32 tables are smashed together into
	// a 256*64 or 65536*32 element table, where each element is a 
	// std::vector<int> of the indices of all the matching training vectors.
	void tabulate() {
		const int stride = ((eightbit ? 64 : 32) - 1) / hw_concur + 1;
		int i = 0;
		int start = 0;
		for (; i < std::min(qcount - 1, hw_concur - 1); ++i, start += stride) {
			fut[i] = std::async(std::launch::async, &Matcher::_tabulate, this, start, stride);
		}
		fut[i] = std::async(std::launch::async, &Matcher::_tabulate, this, start, (eightbit ? 64 : 32) - start);
		for (int j = 0; j <= i; ++j) fut[j].wait();
	}

	void _tabulate(const int start, const int count) {
		for (uint64_t i = static_cast<uint64_t>(start); i < static_cast<uint64_t>(start + count); ++i) {
			for (uint64_t j = 0; j < (eightbit ? 256 : 65536); ++j) raw_table[(i << (eightbit ? 8 : 16)) + j].clear();
			for (int t = 0; t < tcount; ++t) {
				(eightbit ?
					raw_table[(i << 8) + *(reinterpret_cast<const uint8_t* const __restrict>(tset) + (t << 6) + i)] :
					raw_table[(i << 16) + *(reinterpret_cast<const uint16_t* const __restrict>(tset) + (t << 5) + i)]
					).push_back(t);
			}
		}
	}

	// A second pass over the raw_table to compact it
	// into a contiguous stream, with a new second
	// vector storing the ends of the bins in the stream,
	// now that it's no longer evenly spaced.
	//
	// Note that the FIRST element of 'ends' is 0,
	// so that clients can query ends[i+1] to get (one past) the end of bin 'i',
	// and ends[i] to get the start of bin 'i'. This allows the idiom:
	//
	// for (auto j = ends[i]; j < ends[i + 1]; ++j)
	//
	void compact() {
		compact_table.clear();
		int* __restrict eptr = ends;
		*eptr++ = 0;
		for (int i = 0; i < (eightbit ? 256 * 64 : 65536 * 32); ++i) {
			for (int elem : raw_table[i]) {
				compact_table.push_back(elem);
			}
			*eptr++ = static_cast<int>(compact_table.size());
		}
	}

	// Step through the fixed-size 'match_idxs' and add
	// all valid matches to the 'matches' vector
	void addMatches() {
		for (int q = 0; q < qcount; ++q) {
			if (match_idxs[q].x != -1) {
				matches.emplace_back(q, match_idxs[q].x);
			}
		}
	}

	// High-speed, probabilistic MIH matcher. Will not find all (or even many, necessarily)
	// matches. However, it is guaranteed to return no false positives,
	// so the matches it DOES return are valid.
	// Useful for performance-critical tasks where
	// perfect matching accuracy is not required.
	void matchProb() {
		match_idxs.resize(qcount);
		const int stride = (qcount - 1) / hw_concur + 1;
		int i = 0;
		int start = 0;
		for (; i < std::min(qcount - 1, hw_concur - 1); ++i, start += stride) {
			fut[i] = std::async(std::launch::async, &Matcher::_matchProb, this, start, stride);
		}
		fut[i] = std::async(std::launch::async, &Matcher::_matchProb, this, start, qcount - start);
		for (int j = 0; j <= i; ++j) fut[j].wait();
	}

	// Probabilistic MIH matcher. Matches not guaranteed to be valid are stored
	// in 'remainder' for solving in another way, such as calling twiddle_1 and then
	// twiddle_n as required. This approach retains some of the performance while providing
	// the capability for perfect matching if that's required.
	void match() {
		match_idxs.resize(qcount);
		remainder.clear();
		const int stride = (qcount - 1) / hw_concur + 1;
		int i = 0;
		int start = 0;
		for (; i < std::min(qcount - 1, hw_concur - 1); ++i, start += stride) {
			fut[i] = std::async(std::launch::async, &Matcher::_match, this, i, start, stride);
		}
		fut[i] = std::async(std::launch::async, &Matcher::_match, this, i, start, qcount - start);
		for (int j = 0; j <= i; ++j) {
			fut[j].wait();
			remainder.insert(remainder.end(), rems[j].begin(), rems[j].end());
		}
	}

	// Tries all combinations of indices twiddled by 1 bit such that 2*num_tables - 1 errors can be tolerated
	// instead of just num_tables - 1.
	// Leaves still-uncertain matches in 'remainder' for additional twiddle passes or brute-force.
	void twiddle_1() {
		std::vector<Partial> new_remainder;
		int sz = static_cast<int>(remainder.size());
		const int stride = (sz - 1) / hw_concur + 1;
		int i = 0;
		int start = 0;
		for (; i < std::min(sz - 1, hw_concur - 1); ++i, start += stride) {
			fut[i] = std::async(std::launch::async, &Matcher::_twiddle_1, this, i, start, stride);
		}
		fut[i] = std::async(std::launch::async, &Matcher::_twiddle_1, this, i, start, sz - start);
		for (int j = 0; j <= i; ++j) {
			fut[j].wait();
			new_remainder.insert(new_remainder.end(), rems[j].begin(), rems[j].end());
		}
		remainder.swap(new_remainder);
	}

	// Tries all combinations of indices twiddled by n bits such that (n+1)*num_tables - 1 errors can be tolerated.
	// Leaves still-uncertain matches in 'remainder' for additional twiddle passes or brute-force.
	void twiddle_n(const int n) {
		std::vector<Partial> new_remainder;
		int sz = static_cast<int>(remainder.size());
		const int stride = (sz - 1) / hw_concur + 1;
		int i = 0;
		int start = 0;
		for (; i < std::min(sz - 1, hw_concur - 1); ++i, start += stride) {
			fut[i] = std::async(std::launch::async, &Matcher::_twiddle_n, this, i, n, start, stride);
		}
		fut[i] = std::async(std::launch::async, &Matcher::_twiddle_n, this, i, n, start, sz - start);
		for (int j = 0; j <= i; ++j) {
			fut[j].wait();
			new_remainder.insert(new_remainder.end(), rems[j].begin(), rems[j].end());
		}
		remainder.swap(new_remainder);
	}

	// Simple but optimized brute force (n^2) REMAINDER matcher. If just a few
	// query vectors remain after several twiddles, it might be worth
	// falling back to this function, which only matches those vectors remaining
	// in 'remainder'.
	void remainderBruteMatch() {
		int sz = static_cast<int>(remainder.size());
		match_idxs.resize(sz);
		const int stride = (sz - 1) / hw_concur + 1;
		int i = 0;
		int start = 0;
		for (; i < std::min(sz - 1, hw_concur - 1); ++i, start += stride) {
			fut[i] = std::async(std::launch::async, &Matcher::_remainderBruteMatch, this, start, stride);
		}
		fut[i] = std::async(std::launch::async, &Matcher::_remainderBruteMatch, this, start, sz - start);
		for (int j = 0; j <= i; ++j) fut[j].wait();
	}

public:
	// Simple but optimized brute force (n^2) matcher.
	void bruteMatch() {
		match_idxs.resize(qcount);
		const int stride = (qcount - 1) / hw_concur + 1;
		int i = 0;
		int start = 0;
		for (; i < std::min(qcount - 1, hw_concur - 1); ++i, start += stride) {
			fut[i] = std::async(std::launch::async, &Matcher::_bruteMatch, this, start, stride);
		}
		fut[i] = std::async(std::launch::async, &Matcher::_bruteMatch, this, start, qcount - start);
		for (int j = 0; j <= i; ++j) fut[j].wait();
		matches.clear();
		addMatches();
	}

	// Compute an exact match using MIH techniques.
	// Reverts to brute force after 'max_twiddles' twiddle passes.
	void exactMatch() {
		tabulate();
		compact();
		match();
#ifdef DEBUG_PRINT
		std::cout << "After twiddle 0, there remain " << remainder.size() << std::endl;
#endif
		if (max_twiddles && !remainder.empty()) {
			twiddle_1();
#ifdef DEBUG_PRINT
			std::cout << "After twiddle 1, there remain " << remainder.size() << std::endl;
#endif
		}
		int twiddle_level = 2;
		while (!remainder.empty() && twiddle_level <= max_twiddles) {
			twiddle_n(twiddle_level++);
#ifdef DEBUG_PRINT
			std::cout << "After twiddle " << twiddle_level - 1 << ", there remain " << remainder.size() << std::endl;
#endif
		}
		if (!remainder.empty()) remainderBruteMatch();
		matches.clear();
		addMatches();
	}

	// Compute a probabilistic fast match using one pass of MIH.
	// Will not find all (or even many, necessarily)
	// matches. However, it is guaranteed to return no false positives,
	// so the matches it DOES return are valid.
	// Useful for performance-critical tasks where
	// perfect matching accuracy is not required.
	void fastApproxMatch() {
		tabulate();
		matchProb();
		matches.clear();
		addMatches();
	}

	// Use MIH to match, but just give up after 'n' twiddles.
	void approxMatchToNTwiddles(const int n) {
		tabulate();
		compact();
		match();
		if (n > 0 && !remainder.empty()) twiddle_1();
		int twiddle_level = 2;
		while (!remainder.empty() && twiddle_level <= n) twiddle_n(twiddle_level++);
		matches.clear();
		addMatches();
	}

	// Update the Matcher's parameters. You MUST do this before
	// using the matcher, either in construction or with this function.
	void update(const void* const __restrict _tset, const int _tcount, const void* const __restrict _qset, const int _qcount, const int _threshold, const int _max_twiddles) {
		tset = _tset;
		tcount = _tcount;
		qset = _qset;
		qcount = _qcount;
		threshold = _threshold;
		max_twiddles = _max_twiddles;
	};

private:
	void _bruteMatch(const int start, const int count) {
		const uint64_t* const __restrict q64 = reinterpret_cast<const uint64_t* const __restrict>(qset);
		const uint64_t* const __restrict t64 = reinterpret_cast<const uint64_t* const __restrict>(tset);

		for (int q = start; q < start + count; ++q) {
			const uint64_t qp = q << 3;
			int best_i = -1;
			int16_t best_v = 10000;
			int16_t second_v = 20000;

			const register uint64_t qa = q64[qp];
			const register uint64_t qb = q64[qp + 1];
			const register uint64_t qc = q64[qp + 2];
			const register uint64_t qd = q64[qp + 3];
			const register uint64_t qe = q64[qp + 4];
			const register uint64_t qf = q64[qp + 5];
			const register uint64_t qg = q64[qp + 6];
			const register uint64_t qh = q64[qp + 7];

			for (int t = 0, tp = 0; t < tcount; ++t, tp += 8) {
				const int16_t score = static_cast<int16_t>(
					_mm_popcnt_u64(qa ^ t64[tp])
					+ _mm_popcnt_u64(qb ^ t64[tp + 1])
					+ _mm_popcnt_u64(qc ^ t64[tp + 2])
					+ _mm_popcnt_u64(qd ^ t64[tp + 3])
					+ _mm_popcnt_u64(qe ^ t64[tp + 4])
					+ _mm_popcnt_u64(qf ^ t64[tp + 5])
					+ _mm_popcnt_u64(qg ^ t64[tp + 6])
					+ _mm_popcnt_u64(qh ^ t64[tp + 7]));
				if (score < second_v) second_v = score;
				if (score < best_v) {
					second_v = best_v;
					best_v = score;
					best_i = t;
				}
			}

			if (second_v - best_v <= threshold) best_i = -1;
			match_idxs[q].x = best_i;
		}
	}

	void _remainderBruteMatch(const int start, const int count) {
		const uint64_t* const __restrict q64 = reinterpret_cast<const uint64_t* const __restrict>(qset);
		const uint64_t* const __restrict t64 = reinterpret_cast<const uint64_t* const __restrict>(tset);

		for (int partial = start; partial < start + count; ++partial) {
			const Partial& p = remainder[partial];
			const int qp = p.q << 3;
			int best_i = p.best_i;
			int16_t best_v = p.best_v;
			int16_t second_v = p.second_v;

			const register uint64_t qa = q64[qp];
			const register uint64_t qb = q64[qp + 1];
			const register uint64_t qc = q64[qp + 2];
			const register uint64_t qd = q64[qp + 3];
			const register uint64_t qe = q64[qp + 4];
			const register uint64_t qf = q64[qp + 5];
			const register uint64_t qg = q64[qp + 6];
			const register uint64_t qh = q64[qp + 7];

			for (int t = 0, tp = 0; t < tcount; ++t, tp += 8) {
				if (t == best_i) continue;
				const int16_t score = static_cast<int16_t>(
					_mm_popcnt_u64(qa ^ t64[tp])
					+ _mm_popcnt_u64(qb ^ t64[tp + 1])
					+ _mm_popcnt_u64(qc ^ t64[tp + 2])
					+ _mm_popcnt_u64(qd ^ t64[tp + 3])
					+ _mm_popcnt_u64(qe ^ t64[tp + 4])
					+ _mm_popcnt_u64(qf ^ t64[tp + 5])
					+ _mm_popcnt_u64(qg ^ t64[tp + 6])
					+ _mm_popcnt_u64(qh ^ t64[tp + 7]));
				if (score < best_v) {
					second_v = best_v;
					best_v = score;
					best_i = t;
				}
				else if (score < second_v) {
					second_v = score;
				}
			}

			if (second_v - best_v <= threshold) best_i = -1;
			match_idxs[p.q].x = best_i;
		}
	}

	void _matchProb(const int start, const int count) {
		const uint64_t* const __restrict q64 = reinterpret_cast<const uint64_t* const __restrict>(qset);
		const uint64_t* const __restrict t64 = reinterpret_cast<const uint64_t* const __restrict>(tset);

		for (int q = start; q < start + count; ++q) {
			const uint64_t qp = q << 3;
			int best_i = -1;
			int16_t best_v = 10000;
			int16_t second_v = 20000;

			for (int i = 0; i < (eightbit ? 64 : 32); ++i) {
				for (const int t : raw_table[(i << (eightbit ? 8 : 16)) + (eightbit ? *(reinterpret_cast<const uint8_t* const __restrict>(q64 + qp) + i) : *(reinterpret_cast<const uint16_t* const __restrict>(q64 + qp) + i))]) {
					if (t == best_i) continue;
					const int tp = t << 3;
					const int16_t score = static_cast<int16_t>(
						_mm_popcnt_u64(q64[qp] ^ t64[tp]) +
						_mm_popcnt_u64(q64[qp + 1] ^ t64[tp + 1]) +
						_mm_popcnt_u64(q64[qp + 2] ^ t64[tp + 2]) +
						_mm_popcnt_u64(q64[qp + 3] ^ t64[tp + 3]) +
						_mm_popcnt_u64(q64[qp + 4] ^ t64[tp + 4]) +
						_mm_popcnt_u64(q64[qp + 5] ^ t64[tp + 5]) +
						_mm_popcnt_u64(q64[qp + 6] ^ t64[tp + 6]) +
						_mm_popcnt_u64(q64[qp + 7] ^ t64[tp + 7]));
					if (score < second_v) second_v = score;
					if (score < best_v) {
						second_v = best_v;
						best_v = score;
						best_i = t;
					}
				}
			}
			int bvpt = best_v + threshold;
			if (second_v <= bvpt || bvpt >(eightbit ? 63 : 31)) best_i = -1;
			match_idxs[q].x = best_i;
		}
	}

	void _match(const int thread, const int start, const int count) {
		rems[thread].clear();

		const uint64_t* const __restrict q64 = reinterpret_cast<const uint64_t* const __restrict>(qset);
		const uint64_t* const __restrict t64 = reinterpret_cast<const uint64_t* const __restrict>(tset);

		for (int q = start; q < start + count; ++q) {
			const int qp = q << 3;
			int best_i = -1;
			int16_t best_v = 10000;
			int16_t second_v = 20000;

			for (uint64_t i = 0; i < (eightbit ? 64 : 32); ++i) {
				const uint64_t deref = (i << (eightbit ? 8 : 16)) + (eightbit ? *(reinterpret_cast<const uint8_t* const __restrict>(q64 + qp) + i) : *(reinterpret_cast<const uint16_t* const __restrict>(q64 + qp) + i));
				for (int j = ends[deref]; j < ends[deref + 1]; ++j) {
					const int t = compact_table[j];
					if (t == best_i) continue;
					const int tp = t << 3;
					const int16_t score = static_cast<int16_t>(
						_mm_popcnt_u64(q64[qp] ^ t64[tp]) +
						_mm_popcnt_u64(q64[qp + 1] ^ t64[tp + 1]) +
						_mm_popcnt_u64(q64[qp + 2] ^ t64[tp + 2]) +
						_mm_popcnt_u64(q64[qp + 3] ^ t64[tp + 3]) +
						_mm_popcnt_u64(q64[qp + 4] ^ t64[tp + 4]) +
						_mm_popcnt_u64(q64[qp + 5] ^ t64[tp + 5]) +
						_mm_popcnt_u64(q64[qp + 6] ^ t64[tp + 6]) +
						_mm_popcnt_u64(q64[qp + 7] ^ t64[tp + 7]));
					if (score < second_v) second_v = score;
					if (score < best_v) {
						second_v = best_v;
						best_v = score;
						best_i = t;
					}
				}
			}
			int bvpt = best_v + threshold;
			match_idxs[q].x = -1;
			if (second_v > bvpt || best_v >(eightbit ? 63 : 31)) {
				if (bvpt <= (eightbit ? 63 : 31)) {
					match_idxs[q].x = best_i;
				}
				else {
					rems[thread].emplace_back(q, best_i, best_v, second_v);
				}
			}
		}
	}

	void _twiddle_1(const int thread, const int start, const int count) {
		rems[thread].clear();

		const uint64_t* const __restrict q64 = reinterpret_cast<const uint64_t* const __restrict>(qset);
		const uint64_t* const __restrict t64 = reinterpret_cast<const uint64_t* const __restrict>(tset);

		for (int partial = start; partial < start + count; ++partial) {
			const Partial& p = remainder[partial];
			const int qp = p.q << 3;
			int best_i = p.best_i;
			int16_t best_v = p.best_v;
			int16_t second_v = p.second_v;
			for (uint64_t i = 0; i < (eightbit ? 64 : 32); ++i) {
				if (eightbit) {
					for (uint8_t twiddle = 1; twiddle; twiddle <<= 1) {
						const uint64_t deref = (i << 8) + ((*(reinterpret_cast<const uint8_t* const __restrict>(q64 + qp) + i)) ^ twiddle);
						for (int j = ends[deref]; j < ends[deref + 1]; ++j) {
							const int t = compact_table[j];
							if (t == best_i) continue;
							const int tp = t << 3;
							const int16_t score = static_cast<int16_t>(
								_mm_popcnt_u64(q64[qp] ^ t64[tp]) +
								_mm_popcnt_u64(q64[qp + 1] ^ t64[tp + 1]) +
								_mm_popcnt_u64(q64[qp + 2] ^ t64[tp + 2]) +
								_mm_popcnt_u64(q64[qp + 3] ^ t64[tp + 3]) +
								_mm_popcnt_u64(q64[qp + 4] ^ t64[tp + 4]) +
								_mm_popcnt_u64(q64[qp + 5] ^ t64[tp + 5]) +
								_mm_popcnt_u64(q64[qp + 6] ^ t64[tp + 6]) +
								_mm_popcnt_u64(q64[qp + 7] ^ t64[tp + 7]));
							if (score < second_v) second_v = score;
							if (score < best_v) {
								second_v = best_v;
								best_v = score;
								best_i = t;
							}
						}
					}
				}
				else {
					for (uint16_t twiddle = 1; twiddle; twiddle <<= 1) {
						const uint64_t deref = (i << 16) + ((*(reinterpret_cast<const uint16_t* const __restrict>(q64 + qp) + i)) ^ twiddle);
						for (int j = ends[deref]; j < ends[deref + 1]; ++j) {
							const int t = compact_table[j];
							if (t == best_i) continue;
							const int tp = t << 3;
							const int16_t score = static_cast<int16_t>(
								_mm_popcnt_u64(q64[qp] ^ t64[tp]) +
								_mm_popcnt_u64(q64[qp + 1] ^ t64[tp + 1]) +
								_mm_popcnt_u64(q64[qp + 2] ^ t64[tp + 2]) +
								_mm_popcnt_u64(q64[qp + 3] ^ t64[tp + 3]) +
								_mm_popcnt_u64(q64[qp + 4] ^ t64[tp + 4]) +
								_mm_popcnt_u64(q64[qp + 5] ^ t64[tp + 5]) +
								_mm_popcnt_u64(q64[qp + 6] ^ t64[tp + 6]) +
								_mm_popcnt_u64(q64[qp + 7] ^ t64[tp + 7]));
							if (score < second_v) second_v = score;
							if (score < best_v) {
								second_v = best_v;
								best_v = score;
								best_i = t;
							}
						}
					}
				}
			}
			const int bvpt = best_v + threshold;
			match_idxs[p.q].x = -1;
			if (second_v > bvpt || best_v > (eightbit ? 127 : 63)) {
				if (bvpt <= (eightbit ? 127 : 63)) {
					match_idxs[p.q].x = best_i;
				}
				else {
					rems[thread].emplace_back(p.q, best_i, best_v, second_v);
				}
			}
		}
	}

	void _twiddle_n(const int thread, const int n, const int start, const int count) {
		rems[thread].clear();
		const int thresh_delta = (n + 1) << (eightbit ? 6 : 5);

		const uint64_t* const __restrict q64 = reinterpret_cast<const uint64_t* const __restrict>(qset);
		const uint64_t* const __restrict t64 = reinterpret_cast<const uint64_t* const __restrict>(tset);

		for (int partial = start; partial < start + count; ++partial) {
			const Partial& p = remainder[partial];
			const int qp = p.q << 3;
			int best_i = p.best_i;
			int16_t best_v = p.best_v;
			int16_t second_v = p.second_v;
			for (uint64_t i = 0; i < (eightbit ? 64 : 32); ++i) {
				if (eightbit) {
					for (const uint8_t twiddle : twiddle_table_8[n]) {
						const uint64_t deref = (i << 8) + ((*(reinterpret_cast<const uint8_t* const __restrict>(q64 + qp) + i)) ^ twiddle);
						for (int j = ends[deref]; j < ends[deref + 1]; ++j) {
							const int t = compact_table[j];
							if (t == best_i) continue;
							const int tp = t << 3;
							const int16_t score = static_cast<int16_t>(
								_mm_popcnt_u64(q64[qp] ^ t64[tp]) +
								_mm_popcnt_u64(q64[qp + 1] ^ t64[tp + 1]) +
								_mm_popcnt_u64(q64[qp + 2] ^ t64[tp + 2]) +
								_mm_popcnt_u64(q64[qp + 3] ^ t64[tp + 3]) +
								_mm_popcnt_u64(q64[qp + 4] ^ t64[tp + 4]) +
								_mm_popcnt_u64(q64[qp + 5] ^ t64[tp + 5]) +
								_mm_popcnt_u64(q64[qp + 6] ^ t64[tp + 6]) +
								_mm_popcnt_u64(q64[qp + 7] ^ t64[tp + 7]));
							if (score < second_v) second_v = score;
							if (score < best_v) {
								second_v = best_v;
								best_v = score;
								best_i = t;
							}
						}
					}
				}
				else {
					for (const uint16_t twiddle : twiddle_table_16[n]) {
						const uint64_t deref = (i << 16) + ((*(reinterpret_cast<const uint16_t* const __restrict>(q64 + qp) + i)) ^ twiddle);
						for (int j = ends[deref]; j < ends[deref + 1]; ++j) {
							const int t = compact_table[j];
							if (t == best_i) continue;
							const int tp = t << 3;
							const int16_t score = static_cast<int16_t>(
								_mm_popcnt_u64(q64[qp] ^ t64[tp]) +
								_mm_popcnt_u64(q64[qp + 1] ^ t64[tp + 1]) +
								_mm_popcnt_u64(q64[qp + 2] ^ t64[tp + 2]) +
								_mm_popcnt_u64(q64[qp + 3] ^ t64[tp + 3]) +
								_mm_popcnt_u64(q64[qp + 4] ^ t64[tp + 4]) +
								_mm_popcnt_u64(q64[qp + 5] ^ t64[tp + 5]) +
								_mm_popcnt_u64(q64[qp + 6] ^ t64[tp + 6]) +
								_mm_popcnt_u64(q64[qp + 7] ^ t64[tp + 7]));
							if (score < second_v) second_v = score;
							if (score < best_v) {
								second_v = best_v;
								best_v = score;
								best_i = t;
							}
						}
					}
				}
			}
			const int bvpt = best_v + threshold;
			match_idxs[p.q].x = -1;
			if (second_v > bvpt || best_v >= thresh_delta) {
				if (bvpt < thresh_delta) {
					match_idxs[p.q].x = best_i;
				}
				else {
					rems[thread].emplace_back(p.q, best_i, best_v, second_v);
				}
			}
		}
	}
};
