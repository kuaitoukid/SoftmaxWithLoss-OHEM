#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  if (use_hard_mining_) {
    losses_.clear(); 
    selected_indexes_.clear();
    ignored_indexes_.clear();
    // assign vector losses_
    for (int i = 0; i < nthreads; i ++) {
      // std::cout <<  float(bottom[0]->cpu_diff()[i]) << std::endl;
      // std::cout << i << "/" << losses_.size() << std::endl;
      losses_.push_back(std::make_pair(i, float(bottom[0]->cpu_diff()[i])));
    }
    if (hard_size_ > 0) {
      top[0]->mutable_cpu_data()[0] = 0;
      std::sort(losses_.begin(), losses_.end(), comp);
      for (int i = 0; i < hard_size_; i ++) {
        selected_indexes_.push_back(losses_[i].first);
        top[0]->mutable_cpu_data()[0] += losses_[i].second;
      }
    }
    int norm_size = batch_size_ - hard_size_;
    if (norm_size > 0) {
      random_shuffle(losses_.begin() + hard_size_, losses_.end());
      for (int i = hard_size_; i < batch_size_; i ++) {
        selected_indexes_.push_back(losses_[i].first);
        top[0]->mutable_cpu_data()[0] += losses_[i].second;
      }
    }
    for (int i = batch_size_; i < nthreads; i ++) {
      ignored_indexes_.push_back(losses_[i].first);
    }
    top[0]->mutable_cpu_data()[0] /= batch_size_;
    
  } else {
    Dtype loss;
    caffe_gpu_asum(nthreads, loss_data, &loss);
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                          valid_count);      
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    if (use_hard_mining_) {
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const Dtype* prob_data = prob_.cpu_data();
      caffe_copy(prob_.count(), prob_data, bottom_diff);
      const Dtype* label = bottom[1]->cpu_data();
      int dim = prob_.count() / outer_num_;
      int valid_count = 0;
      for (int sid = 0; sid < selected_indexes_.size(); sid ++) {
        int j = selected_indexes_[sid] % inner_num_;
        int i = selected_indexes_[sid] / inner_num_;
        // std::cout << j << "," << i << std::endl;
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++valid_count;
        }
      }
      for (int iid = 0; iid < ignored_indexes_.size(); iid ++) {
        int j = ignored_indexes_[iid] % inner_num_;
        int i = ignored_indexes_[iid] / inner_num_;
        for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
          bottom_diff[i * dim + c * inner_num_ + j] = 0;
        }
      }
      const Dtype loss_weight = top[0]->cpu_diff()[0] /
                              get_normalizer(normalization_, valid_count);
      caffe_gpu_scal(prob_.count(), loss_weight, bottom[0]->mutable_gpu_diff());
    } else {
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* prob_data = prob_.gpu_data();
      const Dtype* top_data = top[0]->gpu_data();
      caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
      const Dtype* label = bottom[1]->gpu_data();
      const int dim = prob_.count() / outer_num_;
      const int nthreads = outer_num_ * inner_num_;
      // Since this memory is never used for anything else,
      // we use to to avoid allocating new GPU memory.
      Dtype* counts = prob_.mutable_gpu_diff();
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
      
      Dtype valid_count = -1;
      // Only launch another CUDA kernel if we actually need the count of valid
      // outputs.
      if (normalization_ == LossParameter_NormalizationMode_VALID &&
          has_ignore_label_) {
        caffe_gpu_asum(nthreads, counts, &valid_count);
      }
      const Dtype loss_weight = top[0]->cpu_diff()[0] /
                              get_normalizer(normalization_, valid_count);
      caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe

