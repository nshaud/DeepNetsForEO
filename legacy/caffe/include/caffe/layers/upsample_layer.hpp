#ifndef CAFFE_UPSAMPLE_LAYER_HPP_
#define CAFFE_UPSAMPLE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

/* Upsample layer from Kendall et al.
   https://github.com/alexgkendall/caffe-segnet
*/

namespace caffe {

template <typename Dtype>
class UpsampleLayer : public Layer<Dtype> {
 public:
  explicit UpsampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Upsample"; }
  // [input, encoder max-pooling mask]
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_;
  int width_;
  int scale_h_, scale_w_;
  bool pad_out_h_, pad_out_w_;
  int upsample_h_, upsample_w_;
};

}  // namespace caffe

#endif  // CAFFE_UPSAMPLE_LAYER_HPP_
