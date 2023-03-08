/*!
* Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#ifndef LIGHTGBM_BOOSTING_SCORE_UPDATER_HPP_
#define LIGHTGBM_BOOSTING_SCORE_UPDATER_HPP_

#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/tree.h>
#include <LightGBM/tree_learner.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/nesterov_boosting.h>

#include <cstring>
#include <vector>

namespace LightGBM {
	/*!
	* \brief Used to store and update score for data
	*/
	class ScoreUpdater {
	public:
		/*!
		* \brief Constructor, will pass a const pointer of dataset
		* \param data This class will bind with this data set
		* \param use_nesterov_acc Set to true if Nesterov acceleration is used for boosting
		*/
		ScoreUpdater(const Dataset* data, int num_tree_per_iteration,
			bool use_nesterov_acc = false) : data_(data) {
			num_data_ = data->num_data();
			score_size_ = static_cast<int64_t>(num_data_)* num_tree_per_iteration;
			score_.resize(score_size_);
			// default start score is zero
			std::memset(score_.data(), 0, score_size_ * sizeof(double));
			has_init_score_ = false;
			const double* init_score = data->metadata().init_score();
			// if exists initial score, will start from it
			if (init_score != nullptr) {
				if ((data->metadata().num_init_score() % num_data_) != 0
					|| (data->metadata().num_init_score() / num_data_) != num_tree_per_iteration) {
					Log::Fatal("Number of class for initial score error");
				}
				has_init_score_ = true;
#pragma omp parallel for schedule(static, 512) if (score_size_ >= 1024)
				for (int64_t i = 0; i < score_size_; ++i) {
					score_[i] = init_score[i];
				}
			}
			if (use_nesterov_acc) {
				InitializeScoreLag1();
			}
		}
		/*! \brief Destructor */
		~ScoreUpdater() {
		}

		inline bool has_init_score() const { return has_init_score_; }

		inline void AddScore(double val, int cur_tree_id) {
			Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
			const size_t offset = static_cast<size_t>(num_data_)* cur_tree_id;
#pragma omp parallel for schedule(static, 512) if (num_data_ >= 1024)
			for (int i = 0; i < num_data_; ++i) {
				score_[offset + i] += val;
			}
		}

		/*! \brief Initialize score_lag1_ for Nesterov momentum step */
		inline void InitializeScoreLag1() {
			score_lag1_.resize(score_size_);
#pragma omp parallel for schedule(static)
			for (int64_t i = 0; i < score_size_; ++i) {
				score_lag1_[i] = score_[i];
			}
			score_lag1_initialized_ = true;
		}

		inline void MultiplyScore(double val, int cur_tree_id) {
			const size_t offset = static_cast<size_t>(num_data_)* cur_tree_id;
#pragma omp parallel for schedule(static, 512) if (num_data_ >= 1024)
			for (int i = 0; i < num_data_; ++i) {
				score_[offset + i] *= val;
			}
		}
		/*!
		* \brief Using tree model to get prediction number, then adding to scores for all data
		*        Note: this function generally will be used on validation data too.
		* \param tree Trained tree model
		* \param cur_tree_id Current tree for multiclass training
		*/
		inline void AddScore(const Tree* tree, int cur_tree_id) {
			Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
			const size_t offset = static_cast<size_t>(num_data_)* cur_tree_id;
			tree->AddPredictionToScore(data_, num_data_, score_.data() + offset);
		}
		/*!
		* \brief Adding prediction score, only used for training data.
		*        The training data is partitioned into tree leaves after training
		*        Based on which We can get prediction quickly.
		* \param tree_learner
		* \param cur_tree_id Current tree for multiclass training
		*/
		inline void AddScore(const TreeLearner* tree_learner, const Tree* tree, int cur_tree_id) {
			Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
			const size_t offset = static_cast<size_t>(num_data_)* cur_tree_id;
			tree_learner->AddPredictionToScore(tree, score_.data() + offset);
		}
		/*!
		* \brief Using tree model to get prediction number, then adding to scores for parts of data
		*        Used for prediction of training out-of-bag data
		* \param tree Trained tree model
		* \param data_indices Indices of data that will be processed
		* \param data_cnt Number of data that will be processed
		* \param cur_tree_id Current tree for multiclass training
		*/
		inline void AddScore(const Tree* tree, const data_size_t* data_indices,
			data_size_t data_cnt, int cur_tree_id) {
			Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
			const size_t offset = static_cast<size_t>(num_data_)* cur_tree_id;
			tree->AddPredictionToScore(data_, data_indices, data_cnt, score_.data() + offset);
		}

		/*!
		* \brief Apply a momentum step (for Nesterov accelerated boosting)
		* \param iter Boosting iteration number
		* \param mu Neterov acceleration rate
		*/
		inline void ApplyMomentumStep(double mu) {
			CHECK(score_lag1_initialized_);
			DoOneMomentumStep(score_.data(), score_lag1_.data(), score_size_, mu);
		}

		/*! \brief Pointer of score */
		inline const double* score() const { return score_.data(); }

		inline data_size_t num_data() const { return num_data_; }

		/*! \brief Disable copy */
		ScoreUpdater& operator=(const ScoreUpdater&) = delete;
		/*! \brief Disable copy */
		ScoreUpdater(const ScoreUpdater&) = delete;

	private:
		/*! \brief Number of total data */
		data_size_t num_data_;
		/*! \brief Pointer of data set */
		const Dataset* data_;
		/*! \brief Scores for data set */
		std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>> score_;
		/*! \brief Lag 1 of scores for boosting with Nesterov acceleration */
		std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>> score_lag1_;
		bool has_init_score_;
		/*! \brief True if score_lag1_ has been set (used for Nesterov accelerated boosting) */
		bool score_lag1_initialized_;
		/*! \brief Number of scores */
		int64_t score_size_;
	};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_SCORE_UPDATER_HPP_
