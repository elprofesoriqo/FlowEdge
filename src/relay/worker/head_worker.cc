#include "relay/worker/head_worker.h"

#include <algorithm>
#include <utility>

namespace fe::relay {

std::expected<HeadWorker, std::string> HeadWorker::open(std::string_view model_path,
                                                        std::optional<unsigned> threads) noexcept
{
  try {
    const std::string path{model_path};
    fe_engine* const engine = threads ? fe_engine_load_with_threads(path.c_str(), *threads)
                                      : fe_engine_load(path.c_str());
    if (engine == nullptr)
      return std::unexpected(fe_engine_last_error());
    fe_model_metadata metadata{};
    if (fe_engine_model_metadata(engine, &metadata) != 0) {
      const std::string error = fe_engine_last_error();
      fe_engine_free(engine);
      return std::unexpected(error);
    }
    if (metadata.condition_dim == 0u || metadata.action_dim == 0u ||
        metadata.condition_dim > kMaxConditionDim || metadata.action_dim > kMaxActionDim) {
      fe_engine_free(engine);
      return std::unexpected("Relay worker requires a compatible flow-head checkpoint");
    }
    return HeadWorker{engine, metadata};
  } catch (...) {
    return std::unexpected("Out of memory opening Relay head worker");
  }
}

HeadWorker::~HeadWorker()
{
  fe_engine_free(engine_);
}

HeadWorker::HeadWorker(HeadWorker&& other) noexcept
    : engine_{std::exchange(other.engine_, nullptr)}, model_metadata_{other.model_metadata_},
      request_envelope_{other.request_envelope_}, generation_{other.generation_},
      busy_{std::exchange(other.busy_, false)}, last_error_{std::move(other.last_error_)}
{
}

HeadWorker& HeadWorker::operator=(HeadWorker&& other) noexcept
{
  if (this != &other) {
    fe_engine_free(engine_);
    engine_ = std::exchange(other.engine_, nullptr);
    model_metadata_ = other.model_metadata_;
    request_envelope_ = other.request_envelope_;
    generation_ = other.generation_;
    busy_ = std::exchange(other.busy_, false);
    last_error_ = std::move(other.last_error_);
  }
  return *this;
}

bool HeadWorker::begin(const ConditionMessage& request) noexcept
{
  last_error_.clear();
  if (validate(request) != ProtocolResult::kSuccess) {
    last_error_ = "Relay condition message is invalid";
    return false;
  }
  const int rc = fe_engine_flow_begin_request(engine_, request.condition_values().data(),
                                              request.noise_values().data(), &request.metadata);
  if (rc != 0) {
    last_error_ = fe_engine_last_error();
    busy_ = false;
    return false;
  }
  request_envelope_ = request.envelope;
  generation_ = request.metadata.generation;
  busy_ = true;
  return true;
}

WorkerStep HeadWorker::advance(ActionMessage& result) noexcept
{
  if (!busy_) {
    last_error_ = "Relay worker has no active request";
    return WorkerStep::kError;
  }
  result.envelope = {};
  result.metadata = {};
  result.envelope.kind = MessageKind::kAction;
  result.envelope.sequence = request_envelope_.sequence;
  result.envelope.session_id = request_envelope_.session_id;
  std::size_t remaining{0uz};
  const int rc = fe_engine_flow_advance(engine_, 1uz, result.action.data(), &remaining);
  if (fe_engine_flow_action_metadata(engine_, &result.metadata) != 0) {
    last_error_ = fe_engine_last_error();
    busy_ = false;
    return WorkerStep::kError;
  }
  result.envelope.struct_size = static_cast<std::uint32_t>(wire_size(result));
  if (rc == 8) {
    busy_ = false;
    return WorkerStep::kCancelled;
  }
  if (rc != 0) {
    last_error_ = fe_engine_last_error();
    result.metadata.status = FE_ACTION_FAILED;
    busy_ = false;
    return WorkerStep::kError;
  }
  if (remaining == 0uz) {
    busy_ = false;
    return WorkerStep::kComplete;
  }
  return WorkerStep::kInProgress;
}

void HeadWorker::cancel_before(std::uint64_t generation) noexcept
{
  fe_engine_cancel_before(engine_, generation);
}

} // namespace fe::relay
