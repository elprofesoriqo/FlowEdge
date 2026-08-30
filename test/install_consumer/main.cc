#include "api/engine.h"
#include "relay/scheduler/edf_scheduler.h"

int main()
{
  fe_engine* const engine = fe_engine_load("");
  fe_engine_free(engine);
  const fe::relay::EdfScheduler scheduler{1uz};
  return scheduler.capacity() == 1uz ? 0 : 1;
}
