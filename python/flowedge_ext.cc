#include "api/engine.h"
#include "protocol/deadline_flow.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

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

py::object float_array(py::handle numpy, std::size_t size)
{
  return numpy.attr("empty")(py::int_(size), py::str("float32"));
}

py::object float_matrix(py::handle numpy, std::size_t rows, std::size_t columns)
{
  return numpy.attr("empty")(py::make_tuple(py::int_(rows), py::int_(columns)), py::str("float32"));
}

py::object contiguous_array(py::handle numpy, py::handle object, const char* dtype)
{
  return numpy.attr("ascontiguousarray")(object, py::str(dtype));
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

int diffusion_scheduler_id(std::string_view scheduler)
{
  if (scheduler == "ddim")
    return FE_DIFFUSION_DDIM;
  if (scheduler == "ddpm")
    return FE_DIFFUSION_DDPM;
  throw std::runtime_error("scheduler must be one of: ddim, ddpm");
}

// Mamba + flow-matching action head
class Engine
{
public:
  explicit Engine(const std::string& path, std::optional<unsigned> threads)
      : engine_{threads ? fe_engine_load_with_threads(path.c_str(), *threads)
                        : fe_engine_load(path.c_str())},
        numpy_{py::module_::import("numpy")}
  {
    if (engine_ == nullptr)
      throw std::runtime_error("FlowEdge: cannot load " + path + ": " + fe_engine_last_error());
  }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  ~Engine() { fe_engine_free(engine_); }

  [[nodiscard]] std::size_t action_dim() const { return fe_engine_action_dim(engine_); }
  [[nodiscard]] std::size_t action_horizon() const { return fe_engine_action_horizon(engine_); }
  [[nodiscard]] std::size_t condition_dim() const { return fe_engine_condition_dim(engine_); }
  [[nodiscard]] unsigned thread_count() const { return fe_engine_thread_count(engine_); }
  [[nodiscard]] py::dict diffusion_metadata() const
  {
    fe_diffusion_metadata metadata{};
    if (fe_engine_diffusion_metadata(engine_, &metadata) != 0)
      throw std::runtime_error(fe_engine_last_error());
    py::dict result;
    result["protocol_version"] = metadata.protocol_version;
    result["action_dim"] = metadata.action_dim;
    result["condition_dim"] = metadata.condition_dim;
    result["horizon"] = metadata.horizon;
    result["action_steps"] = metadata.action_steps;
    result["observation_steps"] = metadata.observation_steps;
    result["train_timesteps"] = metadata.train_timesteps;
    result["clip_sample"] = metadata.clip_sample != 0u;
    result["clip_sample_range"] = metadata.clip_sample_range;
    return result;
  }
  [[nodiscard]] py::dict model_metadata() const
  {
    fe_model_metadata metadata{};
    if (fe_engine_model_metadata(engine_, &metadata) != 0)
      throw std::runtime_error(fe_engine_last_error());
    py::dict result;
    result["protocol_version"] = metadata.protocol_version;
    result["architecture"] = metadata.architecture;
    result["precision"] = metadata.precision;
    result["model_digest"] = digest_hex(metadata.model_digest);
    result["d_model"] = metadata.d_model;
    result["n_layers"] = metadata.n_layers;
    result["action_dim"] = metadata.action_dim;
    result["action_horizon"] = action_horizon();
    result["condition_dim"] = metadata.condition_dim;
    result["decode_snapshot_bytes"] = metadata.decode_snapshot_bytes;
    return result;
  }
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
    py::object tokens_array = contiguous_array(numpy_, tokens_object, "int32");
    const Int32Buffer tokens{tokens_array};
    py::object out = float_matrix(numpy_, tokens.size(), d_model());
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

  // External VLM/proprioception encoder embeddings -> hidden states.
  py::object run_embeddings(py::handle embeddings_object)
  {
    py::object embeddings_array = contiguous_array(numpy_, embeddings_object, "float32");
    const FloatBuffer embeddings{embeddings_array};
    const std::size_t width = d_model();
    if (width == 0uz || embeddings.size() == 0uz || embeddings.size() % width != 0uz)
      throw std::runtime_error("embeddings must contain one or more complete d_model rows");
    py::object out = float_matrix(numpy_, embeddings.size() / width, width);
    FloatBuffer output{out, true};
    run_embeddings_native(embeddings, output);
    return out;
  }

  void run_embeddings_into(py::handle embeddings_object, py::handle output_object)
  {
    const FloatBuffer embeddings{embeddings_object};
    FloatBuffer output{output_object, true};
    run_embeddings_native(embeddings, output);
  }

  py::object step(std::int32_t token)
  {
    py::object out = float_array(numpy_, d_model());
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
    const std::size_t size = fe_engine_decode_state_bytes(engine_);
    if (size > static_cast<std::size_t>(std::numeric_limits<Py_ssize_t>::max()))
      throw std::runtime_error("decode snapshot is too large for a Python bytes object");
    py::bytes snapshot = py::reinterpret_steal<py::bytes>(
        PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(size)));
    if (!snapshot)
      throw py::error_already_set{};
    if (fe_engine_export_decode_state(engine_, PyBytes_AS_STRING(snapshot.ptr()), size) != 0)
      throw std::runtime_error(fe_engine_last_error());
    return snapshot;
  }

  void restore_decode_state(const py::bytes& snapshot)
  {
    char* data = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(snapshot.ptr(), &data, &size) != 0)
      throw py::error_already_set{};
    if (fe_engine_import_decode_state(engine_, data, static_cast<std::size_t>(size)) != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  // prefix conditions the SSM
  // ODE noise -> action [action_dim]
  py::object sample(py::handle prefix_object, py::handle noise_object, std::size_t steps,
                    std::string_view method)
  {
    py::object prefix_array = contiguous_array(numpy_, prefix_object, "int32");
    py::object noise_array = contiguous_array(numpy_, noise_object, "float32");
    const Int32Buffer prefix{prefix_array};
    const FloatBuffer noise{noise_array};
    py::object action = float_array(numpy_, action_dim());
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

  py::object sample_diffusion(py::handle condition_object, py::handle noise_object,
                              std::size_t steps, std::string_view scheduler, std::uint64_t seed)
  {
    py::object condition_array = contiguous_array(numpy_, condition_object, "float32");
    py::object noise_array = contiguous_array(numpy_, noise_object, "float32");
    const FloatBuffer condition{condition_array};
    const FloatBuffer noise{noise_array};
    py::object result = float_matrix(numpy_, action_horizon(), action_dim());
    FloatBuffer action{result, true};
    sample_diffusion_native(condition, noise, action, steps, scheduler, seed);
    return result;
  }

  py::object diffusion_denoise(py::handle condition_object, py::handle sample_object,
                               float timestep)
  {
    py::object condition_array = contiguous_array(numpy_, condition_object, "float32");
    py::object sample_array = contiguous_array(numpy_, sample_object, "float32");
    const FloatBuffer condition{condition_array};
    const FloatBuffer sample{sample_array};
    py::object result = float_matrix(numpy_, action_horizon(), action_dim());
    FloatBuffer predicted_noise{result, true};
    const std::size_t values = action_horizon() * action_dim();
    if (condition.size() != condition_dim() || sample.size() != values)
      throw std::runtime_error("invalid Diffusion Policy condition or sample shape");
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_diffusion_denoise(engine_, condition.data(), sample.data(), timestep,
                                       predicted_noise.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
    return result;
  }

  void sample_diffusion_into(py::handle condition_object, py::handle noise_object,
                             py::handle output_object, std::size_t steps,
                             std::string_view scheduler, std::uint64_t seed)
  {
    const FloatBuffer condition{condition_object};
    const FloatBuffer noise{noise_object};
    FloatBuffer action{output_object, true};
    sample_diffusion_native(condition, noise, action, steps, scheduler, seed);
  }

  void flow_begin(py::handle condition_object, py::handle noise_object, std::size_t steps,
                  std::string_view method, std::optional<std::uint64_t> generation,
                  std::uint64_t timestamp_ns, std::uint64_t deadline_ns)
  {
    const FloatBuffer condition{condition_object};
    const FloatBuffer noise{noise_object};
    validate_vectors(condition.size(), noise.size());
    const int m = method_id(method);
    int rc{0};
    {
      py::gil_scoped_release release;
      if (generation) {
        fe_condition_metadata metadata{};
        rc = fe_engine_make_condition_metadata(engine_, timestamp_ns, deadline_ns, *generation,
                                               steps, m, &metadata);
        if (rc == 0)
          rc = fe_engine_flow_begin_request(engine_, condition.data(), noise.data(), &metadata);
      } else {
        rc = fe_engine_flow_begin(engine_, condition.data(), noise.data(), steps, m);
      }
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  void cancel_before(std::uint64_t generation) { fe_engine_cancel_before(engine_, generation); }

  [[nodiscard]] py::dict flow_metadata() const
  {
    fe_action_metadata metadata{};
    if (fe_engine_flow_action_metadata(engine_, &metadata) != 0)
      throw std::runtime_error(fe_engine_last_error());
    py::dict result;
    result["protocol_version"] = metadata.protocol_version;
    result["solver"] = metadata.solver;
    result["status"] = metadata.status;
    result["model_digest"] = digest_hex(metadata.model_digest);
    result["timestamp_ns"] = metadata.timestamp_ns;
    result["deadline_ns"] = metadata.deadline_ns;
    result["generation"] = metadata.generation;
    result["condition_dim"] = metadata.condition_dim;
    result["action_dim"] = metadata.action_dim;
    result["remaining_nfe"] = metadata.remaining_nfe;
    return result;
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
  [[nodiscard]] static std::string digest_hex(const fe_model_digest& digest)
  {
    constexpr char alphabet[] = "0123456789abcdef";
    std::string result(2uz * FE_MODEL_DIGEST_BYTES, '0');
    for (std::size_t i{0uz}; i < FE_MODEL_DIGEST_BYTES; ++i) {
      result[2uz * i] = alphabet[digest.bytes[i] >> 4u];
      result[(2uz * i) + 1uz] = alphabet[digest.bytes[i] & 0x0fu];
    }
    return result;
  }

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

  void run_embeddings_native(const FloatBuffer& embeddings, FloatBuffer& output)
  {
    const std::size_t width = d_model();
    if (width == 0uz || embeddings.size() == 0uz || embeddings.size() % width != 0uz ||
        output.size() != embeddings.size())
      throw std::runtime_error("embedding input and output must be [seq_len, d_model]");
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_run_embeddings(engine_, embeddings.data(), embeddings.size() / width,
                                     output.mutable_data());
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

  void sample_diffusion_native(const FloatBuffer& condition, const FloatBuffer& noise,
                               FloatBuffer& action, std::size_t steps, std::string_view scheduler,
                               std::uint64_t seed)
  {
    if (action_horizon() <= 1uz)
      throw std::runtime_error("checkpoint has no Diffusion Policy head");
    if (condition.size() != condition_dim())
      throw std::runtime_error("condition length must equal condition_dim");
    const std::size_t values = action_horizon() * action_dim();
    if (noise.size() != values)
      throw std::runtime_error("noise size must equal action_horizon * action_dim");
    if (action.size() != values)
      throw std::runtime_error("output size must equal action_horizon * action_dim");
    const int scheduler_id = diffusion_scheduler_id(scheduler);
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_sample_diffusion(engine_, condition.data(), noise.data(), steps, scheduler_id,
                                      seed, action.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  fe_engine* engine_;
  py::object numpy_;
};

} // namespace

PYBIND11_MODULE(flowedge, m)
{
#if defined(__clang__)
  m.attr("compiler") = "Clang " __clang_version__;
#elif defined(__GNUC__)
  m.attr("compiler") = "GCC " __VERSION__;
#elif defined(_MSC_VER)
  m.attr("compiler") = std::string{"MSVC "} + std::to_string(_MSC_VER);
#else
  m.attr("compiler") = "unknown";
#endif
#if defined(NDEBUG)
  m.attr("build_type") = "Release";
#else
  m.attr("build_type") = "Debug";
#endif
  py::class_<fe::ExecutionPlan>(m, "ExecutionPlan")
      .def(py::init<>())
      .def_readwrite("id", &fe::ExecutionPlan::id)
      .def_readwrite("steps", &fe::ExecutionPlan::steps)
      .def_readwrite("solver", &fe::ExecutionPlan::solver)
      .def_readwrite("precision", &fe::ExecutionPlan::precision)
      .def_readwrite("trace_id", &fe::ExecutionPlan::trace_id)
      .def_readwrite("hardware_mask", &fe::ExecutionPlan::hardware_mask)
      .def_readwrite("latency_ns", &fe::ExecutionPlan::latency_ns)
      .def_readwrite("memory_bytes", &fe::ExecutionPlan::memory_bytes)
      .def_readwrite("quality", &fe::ExecutionPlan::quality);
  py::class_<fe::PlanContext>(m, "PlanContext")
      .def(py::init<>())
      .def_readwrite("slack_ns", &fe::PlanContext::slack_ns)
      .def_readwrite("buffered_actions", &fe::PlanContext::buffered_actions)
      .def_readwrite("period_ns", &fe::PlanContext::period_ns)
      .def_readwrite("reserve_ns", &fe::PlanContext::reserve_ns)
      .def_readwrite("switch_ns", &fe::PlanContext::switch_ns)
      .def_readwrite("available_memory_bytes", &fe::PlanContext::available_memory_bytes)
      .def_readwrite("hardware_mask", &fe::PlanContext::hardware_mask)
      .def_readwrite("minimum_quality", &fe::PlanContext::minimum_quality)
      .def_readwrite("healthy", &fe::PlanContext::healthy);
  py::class_<fe::DeadlineFlow>(m, "DeadlineFlow")
      .def(py::init([](const std::vector<fe::ExecutionPlan>& plans) {
        auto selector = fe::DeadlineFlow::create(plans);
        if (!selector)
          throw std::invalid_argument("expected 1-32 unique, calibrated execution plans");
        return *selector;
      }))
      .def("select", &fe::DeadlineFlow::select)
      .def("observe", &fe::DeadlineFlow::observe);
  m.doc() = "FlowEdge: flow-matching action-head inference";

  py::class_<Engine>(m, "Engine")
      .def(py::init<const std::string&, std::optional<unsigned>>(), py::arg("path"),
           py::arg("threads") = py::none())
      .def_property_readonly("action_dim", &Engine::action_dim)
      .def_property_readonly("action_horizon", &Engine::action_horizon)
      .def_property_readonly("condition_dim", &Engine::condition_dim)
      .def_property_readonly("d_model", &Engine::d_model)
      .def_property_readonly("thread_count", &Engine::thread_count)
      .def_property_readonly("model_metadata", &Engine::model_metadata)
      .def_property_readonly("diffusion_metadata", &Engine::diffusion_metadata)
      .def("run", &Engine::run, py::arg("tokens"))
      .def("run_into", &Engine::run_into, py::arg("tokens"), py::arg("output"))
      .def("run_embeddings", &Engine::run_embeddings, py::arg("embeddings"))
      .def("run_embeddings_into", &Engine::run_embeddings_into, py::arg("embeddings"),
           py::arg("output"))
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
      .def("sample_diffusion", &Engine::sample_diffusion,
           "sample an un-normalized Diffusion Policy action horizon", py::arg("condition"),
           py::arg("noise"), py::arg("steps") = 10uz, py::arg("scheduler") = "ddim",
           py::arg("seed") = 0u)
      .def("diffusion_denoise", &Engine::diffusion_denoise,
           "run one normalized ConditionalUnet1D epsilon prediction", py::arg("condition"),
           py::arg("sample"), py::arg("timestep"))
      .def("sample_diffusion_into", &Engine::sample_diffusion_into,
           "sample Diffusion Policy into caller-owned storage", py::arg("condition"),
           py::arg("noise"), py::arg("output"), py::arg("steps") = 10uz,
           py::arg("scheduler") = "ddim", py::arg("seed") = 0u)
      .def("flow_begin", &Engine::flow_begin, "begin a resumable flow solve", py::arg("condition"),
           py::arg("noise"), py::arg("steps") = 10uz, py::arg("method") = "euler",
           py::arg("generation") = py::none(), py::arg("timestamp_ns") = 0u,
           py::arg("deadline_ns") = 0u)
      .def("flow_advance", &Engine::flow_advance, "advance a resumable flow solve",
           py::arg("output"), py::arg("step_budget") = 1uz)
      .def("cancel_before", &Engine::cancel_before, py::arg("generation"),
           "cancel an active request from an older generation")
      .def_property_readonly("flow_metadata", &Engine::flow_metadata);
}
