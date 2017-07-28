#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  use_hard_mining_ = this->layer_param_.softmax_with_loss_param().use_hard_mining();
  batch_size_ = this->layer_param_.softmax_with_loss_param().batch_size();
  CHECK_GT(batch_size_, 0);
  int temp_size = inner_num_ * outer_num_;
  batch_size_ = std::min(batch_size_, temp_size);
  Dtype hard_ratio = this->layer_param_.softmax_with_loss_param().hard_ratio();
  CHECK(hard_ratio >= 0 && hard_ratio <= 1);
  hard_size_ = round(batch_size_ * hard_ratio);
  // for (int i = 0; i < batch_size_; i ++) {
  //   selected_indexes_.push_back(i);        
  // }
  // for (int i = 0; i < temp_size; i ++) {
  //   losses_.push_back(std::make_pair(i, 0));
  // }
  // CHECK(false) << inner_num_ << "," << outer_num_;
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  losses_.clear();
  selected_indexes_.clear();
  ignored_indexes_.clear();
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        losses_.push_back(std::make_pair(j + i * inner_num_, 0));
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      losses_.push_back(std::make_pair(j + i * inner_num_,
                                    float(-log(std::max(prob_data[i * dim + label_value * inner_num_ + j], 
                                    Dtype(FLT_MIN))))));
      loss += losses_[j + i * inner_num_].second;
      ++count;
    }
  }
  if (use_hard_mining_) {
    if (hard_size_ > 0) {
      top[0]->mutable_cpu_data()[0] = 0;
      // std::sort(losses_.begin(), losses_.end(), comp);
      for (int i = 0; i < hard_size_; i ++) {
        // std::cout << losses_[i].second << std::endl;
        selected_indexes_.push_back(losses_[i].first);
        top[0]->mutable_cpu_data()[0] += losses_[i].second;
      }
    }
    int norm_size = batch_size_ - hard_size_;
    if (norm_size > 0) {
      // random_shuffle(losses_.begin() + hard_size_, losses_.end());
      for (int i = hard_size_; i < batch_size_; i ++) {
        selected_indexes_.push_back(losses_[i].first);
        top[0]->mutable_cpu_data()[0] += losses_[i].second;
      }
    }
    int temp_size = inner_num_ * outer_num_;
    for (int i = batch_size_; i < temp_size; i ++) {
      ignored_indexes_.push_back(losses_[i].first);
    }
    top[0]->mutable_cpu_data()[0] /= batch_size_;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  }
  if (top.size() == 2) { // hai you zhe zhong cao zuo ?
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    if (use_hard_mining_) {
      //for (int i = 0; i < bottom[0]->count(); i ++) {
      //  std::cout << bottom_diff[i] << " ";
      //}
      //std::cout << "\n";

      //std::cout << "###" << inner_num_ << "###" << outer_num_ << std::endl;
      for (int sid = 0; sid < selected_indexes_.size(); sid ++) {
        int j = selected_indexes_[sid] % inner_num_;
        int i = selected_indexes_[sid] / inner_num_;
        //std::cout << i << std::endl;
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
      for (int iid = 0; iid < ignored_indexes_.size(); iid ++) {
        int j = ignored_indexes_[iid] % inner_num_;
        int i = ignored_indexes_[iid] / inner_num_;
        //std::cout << i << std::endl;
        for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
          bottom_diff[i * dim + c * inner_num_ + j] = 0;
        }
      }
      //std::cout << "#################" << std::endl;
    } else {
      for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
          } else {
            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            ++count;
          }
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe

