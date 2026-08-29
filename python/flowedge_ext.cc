#include "api/engine.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#if defined(__MINGW32__) && defined(__clang__)
namespace std {
__thread void* __once_callable = nullptr;
__thread void (*__once_call)() = nullptr;
} // namespace std
#endif

namespace py = pybind11;

namespace {

template<typename T> class TypedBuffer
{
public:
  explicit TypedBuffer(py::handle object, bool writable = false)
  {
    const int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object.ptr(), &view_, flags) != 0)
      throw py::error_already_set{};
    const bool format_matches = view_.format != nullptr &&
                                (std::is_same_v<T, float> ? std::strcmp(view_.format, "f") == 0
                                                          : (std::strcmp(view_.format, "i") == 0 ||
                                                             std::strcmp(view_.format, "l") == 0));
    if (view_.itemsize != static_cast<Py_ssize_t>(sizeof(T)) || !format_matches) {
      PyBuffer_Release(&view_);
      view_.obj = nullptr;
      throw std::runtime_error(std::is_same_v<T, float> ? "expected a C-contiguous float32 buffer"
                                                        : "expected a C-contiguous int32 buffer");
    }
  }

  ~TypedBuffer()
  {
    if (view_.obj != nullptr)
      PyBuffer_Release(&view_);
  }

  TypedBuffer(const TypedBuffer&) = delete;
  TypedBuffer& operator=(const TypedBuffer&) = delete;

  [[nodiscard]] const T* data() const noexcept { return static_cast<const T*>(view_.buf); }
  [[nodiscard]] T* mutable_data() noexcept { return static_cast<T*>(view_.buf); }
  [[nodiscard]] std::size_t size() const noexcept
  {
    return static_cast<std::size_t>(view_.len / view_.itemsize);
  }

private:
  Py_buffer view_{};
};

using FloatBuffer = TypedBuffer<float>;
using Int32Buffer = TypedBuffer<std::int32_t>;

py::object float_array(std::size_t size)
{
  return py::module_::import("numpy").attr("empty")(py::int_(size), py::str("float32"));
}

py::object float_matrix(std::size_t rows, std::size_t columns)
{
  return py::module_::import("numpy").attr(
      "empty")(py::make_tuple(py::int_(rows), py::int_(columns)), py::str("float32"));
}

py::object contiguous_array(py::handle object, const char* dtype)
{
  return py::module_::import("numpy").attr("ascontiguousarray")(object, py::str(dtype));
}

int method_id(std::string_view method)
{
  if (method == "euler")
    return FE_SOLVER_EULER;
  if (method == "heun")
    return FE_SOLVER_HEUN;
  if (method == "rk4")
    return FE_SOLVER_RK4;
  throw std::runtime_error("method must be one of: euler, heun, rk4");
}

// Mamba + flow-matching action head
class Engine
{
public:
  explicit Engine(const std::string& path) : engine_{fe_engine_load(path.c_str())}
  {
    if (engine_ == nullptr)
      throw std::runtime_error("FlowEdge: cannot load " + path + ": " + fe_engine_last_error());
  }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  ~Engine() { fe_engine_free(engine_); }

  [[nodiscard]] std::size_t action_dim() const { return fe_engine_action_dim(engine_); }
  [[nodiscard]] std::size_t condition_dim() const { return fe_engine_condition_dim(engine_); }
  [[nodiscard]] std::size_t d_model() const
  {
    std::size_t dm{0uz};
    std::size_t nl{0uz};
    fe_engine_dims(engine_, &dm, &nl);
    return dm;
  }

  // tokens -> hidden states [seq_len, d_model]
  py::object run(py::handle tokens_object)
  {
    py::object tokens_array = contiguous_array(tokens_object, "int32");
    const Int32Buffer tokens{tokens_array};
    py::object out = float_matrix(tokens.size(), d_model());
    FloatBuffer output{out, true};
    run_native(tokens, output);
    return out;
  }

  void run_into(py::handle tokens_object, py::handle output_object)
  {
    const Int32Buffer tokens{tokens_object};
    FloatBuffer output{output_object, true};
    run_native(tokens, output);
  }

  py::object step(std::int32_t token)
  {
    py::object out = float_array(d_model());
    FloatBuffer output{out, true};
    step_native(token, output);
    return out;
  }

  void step_into(std::int32_t token, py::handle output_object)
  {
    FloatBuffer output{output_object, true};
    step_native(token, output);
  }

  void step_native(std::int32_t token, FloatBuffer& output)
  {
    if (output.size() != d_model())
      throw std::runtime_error("output length must equal d_model");
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_step(engine_, token, output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  void reset() { fe_engine_reset(engine_); }

  py::bytes decode_state() const
  {
    std::string snapshot(fe_engine_decode_state_bytes(engine_), '\0');
    if (fe_engine_export_decode_state(engine_, snapshot.data(), snapshot.size()) != 0)
      throw std::runtime_error(fe_engine_last_error());
    return py::bytes(snapshot);
  }

  void restore_decode_state(const py::bytes& snapshot)
  {
    const std::string data = snapshot;
    if (fe_engine_import_decode_state(engine_, data.data(), data.size()) != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  // prefix conditions the SSM
  // ODE noise -> action [action_dim]
  py::object sample(py::handle prefix_object, py::handle noise_object, std::size_t steps,
                    std::string_view method)
  {
    py::object prefix_array = contiguous_array(prefix_object, "int32");
    py::object noise_array = contiguous_array(noise_object, "float32");
    const Int32Buffer prefix{prefix_array};
    const FloatBuffer noise{noise_array};
    py::object action = float_array(action_dim());
    FloatBuffer output{action, true};
    sample_native(prefix, noise, output, steps, method);
    return action;
  }

  void sample_into(py::handle prefix_object, py::handle noise_object, py::handle output_object,
                   std::size_t steps, std::string_view method)
  {
    const Int32Buffer prefix{prefix_object};
    const FloatBuffer noise{noise_object};
    FloatBuffer output{output_object, true};
    sample_native(prefix, noise, output, steps, method);
  }

  void sample_condition(py::handle condition_object, py::handle noise_object,
                        py::handle output_object, std::size_t steps, std::string_view method)
  {
    const FloatBuffer condition{condition_object};
    const FloatBuffer noise{noise_object};
    FloatBuffer action{output_object, true};
    validate_vectors(condition.size(), noise.size());
    validate_action_output(action.size());
    const int m = method_id(method);
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_sample_condition(engine_, condition.data(), noise.data(), steps, m,
                                      action.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  void flow_begin(py::handle condition_object, py::handle noise_object, std::size_t steps,
                  std::string_view method)
  {
    const FloatBuffer condition{condition_object};
    const FloatBuffer noise{noise_object};
    validate_vectors(condition.size(), noise.size());
    const int m = method_id(method);
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_flow_begin(engine_, condition.data(), noise.data(), steps, m);
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  std::size_t flow_advance(py::handle output_object, std::size_t step_budget)
  {
    FloatBuffer action{output_object, true};
    validate_action_output(action.size());
    std::size_t remaining{0uz};
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_flow_advance(engine_, step_budget, action.mutable_data(), &remaining);
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
    return remaining;
  }

private:
  void run_native(const Int32Buffer& tokens, FloatBuffer& output)
  {
    if (tokens.size() == 0uz)
      throw std::runtime_error("tokens must not be empty");
    const std::size_t model_width = d_model();
    if (model_width == 0uz ||
        tokens.size() > (std::numeric_limits<std::size_t>::max() / model_width) ||
        output.size() != tokens.size() * model_width)
      throw std::runtime_error("output size must equal token_count * d_model");
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_run(engine_, tokens.data(), tokens.size(), output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  void sample_native(const Int32Buffer& prefix, const FloatBuffer& noise, FloatBuffer& output,
                     std::size_t steps, std::string_view method)
  {
    if (action_dim() == 0uz)
      throw std::runtime_error("checkpoint has no flow head");
    if (prefix.size() == 0uz)
      throw std::runtime_error("prefix must not be empty");
    if (noise.size() != action_dim())
      throw std::runtime_error("noise length must equal action_dim");
    validate_action_output(output.size());
    const int m = method_id(method);
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_sample(engine_, prefix.data(), prefix.size(), noise.data(), steps, m,
                            output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  void validate_vectors(std::size_t condition_size, std::size_t noise_size) const
  {
    if (condition_size != condition_dim())
      throw std::runtime_error("condition length must equal condition_dim");
    if (noise_size != action_dim())
      throw std::runtime_error("noise length must equal action_dim");
  }

  void validate_action_output(std::size_t output_size) const
  {
    if (output_size != action_dim())
      throw std::runtime_error("output length must equal action_dim");
  }

  fe_engine* engine_;
};

} // namespace

PYBIND11_MODULE(flowedge, m)
{
  m.doc() = "FlowEdge: flow-matching action-head inference";

  py::class_<Engine>(m, "Engine")
      .def(py::init<const std::string&>(), py::arg("path"))
      .def_property_readonly("action_dim", &Engine::action_dim)
      .def_property_readonly("condition_dim", &Engine::condition_dim)
      .def_property_readonly("d_model", &Engine::d_model)
      .def("run", &Engine::run, py::arg("tokens"))
      .def("run_into", &Engine::run_into, py::arg("tokens"), py::arg("output"))
      .def("step", &Engine::step, py::arg("token"))
      .def("step_into", &Engine::step_into, py::arg("token"), py::arg("output"))
      .def("reset", &Engine::reset)
      .def("decode_state", &Engine::decode_state, "snapshot the streaming SSM state")
      .def("restore_decode_state", &Engine::restore_decode_state, "restore a streaming SSM state",
           py::arg("snapshot"))
      .def("sample", &Engine::sample, "sample action trajectory", py::arg("prefix"),
           py::arg("noise"), py::arg("steps") = 10uz, py::arg("method") = "euler")
      .def("sample_into", &Engine::sample_into, "sample into caller-owned storage",
           py::arg("prefix"), py::arg("noise"), py::arg("output"), py::arg("steps") = 10uz,
           py::arg("method") = "euler")
      .def("sample_condition", &Engine::sample_condition,
           "sample from an external condition vector", py::arg("condition"), py::arg("noise"),
           py::arg("output"), py::arg("steps") = 10uz, py::arg("method") = "euler")
      .def("flow_begin", &Engine::flow_begin, "begin a resumable flow solve", py::arg("condition"),
           py::arg("noise"), py::arg("steps") = 10uz, py::arg("method") = "euler")
      .def("flow_advance", &Engine::flow_advance, "advance a resumable flow solve",
           py::arg("output"), py::arg("step_budget") = 1uz);
}
