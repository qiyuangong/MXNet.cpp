/*!
*  Copyright (c) 2016 by Contributors
* \file model.h
* \brief MXNET.cpp model module
* \author Zhang Chen
*/

#ifndef MXNETCPP_MODEL_H
#define MXNETCPP_MODEL_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/symbol.h"
#include "mxnet-cpp/ndarray.h"
#include "mxnet-cpp/kvstore.h"
using namespace std;
// TODO remove 
#include <map>


namespace mxnet {
namespace cpp {


struct FeedForwardConfig {
  Symbol symbol;
  std::vector<Context> ctx;
  int num_epoch;
  int epoch_size;
  Optimizer* optimizer;
  int batch_size;
  float learning_rate;
  float weight_decay;
  std::map<string, NDArray> args_map;
  // TODO(zhangchen-qinyinghua) More implement
  // initializer=Uniform(0.01),
  // numpy_batch_size=128,
  // arg_params=None, aux_params=None,
  // allow_extra_params=False,
  // begin_epoch=0,
  // **kwargs):
  FeedForwardConfig(const FeedForwardConfig &other) 
  : symbol(other.symbol)
  , ctx(other.ctx)
  , num_epoch(other.num_epoch)
  , epoch_size(other.epoch_size)
  , optimizer(other.optimizer)
  , batch_size(other.batch_size)
  , learning_rate(other.learning_rate)
  , weight_decay(other.weight_decay)
  , args_map(other.args_map)
  {}

  FeedForwardConfig()
    : ctx({Context::cpu()})
    , num_epoch(0)
    , epoch_size(0)
    , batch_size(64)
    , learning_rate(1e-4)
    , weight_decay(1e-4)
    {}
};
class FeedForward {
 public:
  FeedForward() {

  }
  explicit FeedForward(const FeedForwardConfig &conf) : conf_(conf) {}
  void Predict();
  void Score();
  void Fit(MXDataIter &train_iter, MXDataIter &val_iter) {
    // stepup metric

    // create kvstore for multiple devices and machines
    // CreateKVStore();
    // init optmizer


    // do training
    TrainMultiDevice(train_iter, val_iter);
  }
  void Save();
  void Load();
  static FeedForward Create();

 private:
  void InitParams();
  void InitPredictor();
  void InitIter();
  void InitEvalIter();
  FeedForwardConfig conf_;
  void InitKVStore(KVStore &kvstore, bool update_on_kvstore) {
    for (auto param in ){
      kvstore.init();
      if (update_on_kvstore)
        kvstore.Pull();
    }

  }
  void UpdateParamsOnKVStore();
  void MultipleCallBacks();
  void CreateKVStore();
  void TrainMultiDevice(MXDataIter &train_iter, MXDataIter &val_iter, KVStore* kvstore=nullptr, bool update_on_kvstore=false) {
    // kvstore
    if (kvstore != nullptr) 
      InitKVStore(kvstore, conf_.arg_map, update_on_kvstore);
    if (update_on_kvstore)
      kvstore.SetOptimizer(conf_.optimizer);
    // training
    for (int iter = 0; iter < conf_.num_epoch; ++iter) {
      LG << "Epoch: " << iter;
      train_iter.Reset();
      while (train_iter.Next()) {
        auto data_batch = train_iter.GetDataBatch();
        conf_.args_map["data"] = data_batch.data.Copy(Context::cpu());
        conf_.args_map["data_label"] = data_batch.label.Copy(Context::cpu());
        NDArray::WaitAll();
        auto *exec = conf_.symbol.SimpleBind(Context::cpu(), conf_.args_map);
        exec->Forward(true);
        exec->Backward();
        // TODO kvstore
        exec->UpdateAll(conf_.optimizer, conf_.learning_rate, conf_.weight_decay);
        delete exec;
      }
      // Accuray
      Accuracy acu;
      val_iter.Reset();
      while (val_iter.Next()) {
        auto data_batch = val_iter.GetDataBatch();
        conf_.args_map["data"] = data_batch.data.Copy(Context::cpu());
        conf_.args_map["data_label"] = data_batch.label.Copy(Context::cpu());
        NDArray::WaitAll();
        auto *exec = conf_.symbol.SimpleBind(Context::cpu(), conf_.args_map);
        exec->Forward(false);
        NDArray::WaitAll();
        acu.Update(data_batch.label, exec->outputs[0]);
        delete exec;
      }
      LG << "Accuracy: " << acu.Get();
    }
    MXNotifyShutdown();
  }
};

}  // namespace cpp
}  // namespace mxnet

#endif /* end of include guard: MXNETCPP_MODEL_H */

