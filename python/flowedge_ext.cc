#include "api/engine.h"

#include <cstddef>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

#if defined(__MINGW32__) && defined(__clang__)
namespace std {
__thread void* __once_callable = nullptr;
__thread void (*__once_call)() = nullptr;
} // namespace std
#endif

namespace py = pybind11;

namespace {

// Mamba + flow-matching action head
class Engine
{
public:
  explicit Engine(const std::string& path) : engine_{fe_engine_load(path.c_str())}
  {
    if (engine_ == nullptr)
      throw std::runtime_error("FlowEdge: cannot load " + path);
  }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  ~Engine() { fe_engine_free(engine_); }

  [[nodiscard]] std::size_t action_dim() const { return fe_engine_action_dim(engine_); }
  [[nodiscard]] std::size_t d_model() const
  {
    std::size_t dm{0uz};
    std::size_t nl{0uz};
    fe_engine_dims(engine_, &dm, &nl);
    return dm;
  }

  // tokens -> hidden states [seq_len, d_model]
  py::array_t<float> run(
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> tokens)
  {
    const std::size_t dm = d_model();
    py::array_t<float> out({static_cast<std::size_t>(tokens.size()), dm});
    if (fe_engine_run(engine_, tokens.data(), tokens.size(), out.mutable_data()) != 0)
      throw std::runtime_error("fe_engine_run failed");
    return out;
  }

  // prefix conditions the SSM
  // ODE noise -> action [action_dim]
  py::array_t<float> sample(
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> prefix,
      py::array_t<float, py::array::c_style | py::array::forcecast> noise, std::size_t steps,
      std::string_view method)
  {
    const std::size_t a = action_dim();
    if (a == 0uz)
      throw std::runtime_error("checkpoint has no flow head");
    if (static_cast<std::size_t>(noise.size()) != a)
      throw std::runtime_error("noise length must equal action_dim");
    const int m = (method == "rk4") ? 2 : (method == "heun") ? 1 : 0;
    py::array_t<float> action(a);
    if (fe_engine_sample(engine_, prefix.data(), prefix.size(), noise.data(), steps, m,
                         action.mutable_data()) != 0)
      throw std::runtime_error("fe_engine_sample failed");
    return action;
  }

private:
  fe_engine* engine_;
};

} // namespace

PYBIND11_MODULE(flowedge, m)
{
  m.doc() = "FlowEdge: flow-matching action-head inference";

  py::class_<Engine>(m, "Engine")
      .def(py::init<const std::string&>(), py::arg("path"))
      .def_property_readonly("action_dim", &Engine::action_dim)
      .def_property_readonly("d_model", &Engine::d_model)
      .def("run", &Engine::run, py::arg("tokens"))
      .def("sample", &Engine::sample, "sample action trajectory", py::arg("prefix"),
           py::arg("noise"), py::arg("steps") = 10uz, py::arg("method") = "euler");
}
