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
  std::map<string, NDArray> arg_params;
  std::map<string, NDArray> aux_params;
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
  void Fit(MXDataIter &train_iter, MXDataIter &val_iter, string kvstore="local") {
    // stepup metric
    std::vector<std::string> arg_names;
    std::vector<std::string> param_names;
    std::vector<std::string> aux_names;
    InitParams(arg_names, param_names, aux_names);
    // create kvstore for multiple devices and machines
    bool update_on_kvstore = false;
    KVStore* kv = nullptr;
    if (kvstore != "local") {
      LG << "KVStore Type " << kvstore;
      kv = CreateKVStore(kvstore);
      // point
      // LG << kv;
      update_on_kvstore = true;
    }
    // do training
    if (update_on_kvstore) 
      TrainMultiDevice(train_iter, val_iter, arg_names, param_names, kv);
    else  
      TrainMultiDevice(train_iter, val_iter, arg_names, param_names);
  }
  void Save();
  void Load();
  static FeedForward Create();

 private:
  void InitParams(std::vector<std::string> & arg_names, std::vector<std::string> &param_names, std::vector<std::string> &aux_names) {
    //TODO initParams
    std::vector<std::vector<mx_uint> > in_shapes, aux_shapes, out_shapes;
    std::map<std::string, std::vector<mx_uint> > arg_shapes;
    // std::set<std::string> input_names = {"data", "data_label"};
    arg_names = conf_.symbol.ListArguments();
    for (const auto arg_name : arg_names) {
      if (arg_name != "data" && arg_name != "data_label") 
        param_names.push_back(arg_name);
    }
    for (auto key: arg_names)
      LG << "arg_name --" << key;
    for (auto key: param_names)
      LG << "param_name --" << key;
    aux_names = conf_.symbol.ListAuxiliaryStates();
    // TODO for multiple devices
    // conf_.arg_params = arg_params;
    // conf_.aux_params = aux_params;
  }
  void InitPredictor();
  void InitIter();
  void InitEvalIter();
  FeedForwardConfig conf_;
  void InitKVStore(
      KVStore* kvstore,
      std::vector<NDArray> &param_arrays,
      std::map<string, NDArray> &arg_params,
      std::vector<string> &param_names,
      bool update_on_kvstore=true) {
    //TODO initkvstore
    LG << "Init KVStore";
    // param_arrays, arg_params, param_names
    LG << param_arrays.size();
    for (int i = 0; i < param_arrays.size(); i++) {
      LG << "Init " << i << " on " << param_names[i];
      kvstore->Init(i, arg_params[param_names[i]]);
      if (update_on_kvstore)
        kvstore->Pull(i, &param_arrays[i], -1 * i);
    }
  }
  void UpdateParamsOnKVStore(KVStore* kvstore, std::vector<NDArray> &arg_arrays, std::vector<NDArray> &grad_arrays) {
    for (int i = 0; i < arg_arrays.size(); i++) {
      kvstore->Push(i, grad_arrays[i], -1 * i);
      kvstore->Pull(i, &arg_arrays[i],  -1 * i);
    }
  }
  void MultipleCallBacks();
  KVStore* CreateKVStore(string kvstore, int num_device=1) {
    // if (num_device <= 1)
    //   return nullptr;
    LG << "KVStore Created";
    KVStore* kv = new KVStore();
    LG << kv->GetRole() << " " << kv->GetNumWorkers();
    return kv;
  }
  void TrainMultiDevice(
      MXDataIter &train_iter,
      MXDataIter &val_iter,
      std::vector<std::string> arg_names,
      std::vector<std::string> param_names,
      KVStore* kvstore=nullptr,
      bool update_on_kvstore=false) {
    // kvstore set params
    std::map<string, std::vector<mx_uint>> arg_shapes;
    // std::vector<NDArray> arg_arrays;
    std::vector<NDArray> param_arrays;
    for (const auto &arg_name : arg_names) {
      auto iter = conf_.args_map.find(arg_name);
      if (iter != conf_.args_map.end())
        arg_shapes[arg_name] = iter->second.GetShape();
    }
    // conf_.symbol.InferShape(arg_shapes, &in_shapes, &aux_shapes, &out_shapes);
    std::map<string, NDArray> arg_params;
    for (const auto &arg_name : param_names) {
      // auto curr = NDArray(Shape(arg_shapes[arg_name]));
      // arg_arrays.push_back(curr);
      LG << Shape(arg_shapes[arg_name]);
      // auto curr = NDArray(Shape(arg_shapes[arg_name]), Context::cpu());
      param_arrays.push_back(NDArray(Shape(arg_shapes[arg_name]), Context::cpu()));
      arg_params[arg_name] = NDArray(Shape(arg_shapes[arg_name]), Context::cpu());
    }

    if (kvstore != nullptr) {
      update_on_kvstore = true;
      InitKVStore(kvstore, param_arrays, arg_params, param_names, update_on_kvstore);
    }
    if (update_on_kvstore)
      kvstore->SetOptimizer(std::unique_ptr<Optimizer>(conf_.optimizer));
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
        if (update_on_kvstore)
          UpdateParamsOnKVStore(kvstore, exec->arg_arrays, exec->grad_arrays);
        else
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

