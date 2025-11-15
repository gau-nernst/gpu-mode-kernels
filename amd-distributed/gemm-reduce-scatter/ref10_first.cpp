// #define ZZ_DEBUG
#include <ATen/core/TensorBase.h>
#include <Python.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/library.h>
namespace zz{
void barrier(int id);
void init(int id, uint64_t* tensor);
void clear();

// static bool sheMemInitd = false;
void barrier_cpp(int64_t id) { barrier(id); }
at::Tensor init_cpp(int64_t id) {
  at::Tensor t = at::empty({16}, at::device(at::kCUDA).dtype(at::kUInt64));
  init(id, t.data_ptr<uint64_t>());
  return t;
}
void clear_cpp(){
  clear();
}

TORCH_LIBRARY(my_ops, m) {
  m.def("barrier", &barrier_cpp);
  m.def("init", &init_cpp);
  m.def("clear", &clear_cpp);
};



PyMODINIT_FUNC PyInit_noname(void) {
  static struct PyModuleDef foo = {PyModuleDef_HEAD_INIT, "no_name", nullptr,
                                   -1, nullptr};
  return PyModule_Create(&foo);
}

}