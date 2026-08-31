# Relay observability

Relay records metrics without allocating on request submission, scheduling, worker completion, or
action publication. The fixed registry includes:

- received, corrupt, accepted, published, and dropped messages;
- complete, cancelled, failed, stale, deadline, capacity, and expiry outcomes;
- queue-depth and busy-worker high-water marks;
- worker failures; and
- power-of-two queue, execution, and end-to-end latency histograms in nanoseconds.

Enable one or more exporters on `flowedge-relayd`:

```bash
./build/src/relay/flowedge-relayd --model model.safetensors --create \
  --metrics-prometheus /var/lib/node_exporter/textfile_collector/flowedge.prom \
  --metrics-json run-metrics.json \
  --metrics-otlp-json run-otlp.json
```

Files are written at graceful shutdown. Add `--metrics-interval-ms 5000` to replace them periodically.
Periodic filesystem I/O is opt-in because it runs in the daemon transport loop; leave it disabled for
the cleanest latency benchmark.

The OTLP file is an OTLP/HTTP JSON request body. An existing deployment adapter may post it to a
Collector, for example:

```bash
curl -H 'Content-Type: application/json' --data-binary @run-otlp.json \
  http://127.0.0.1:4318/v1/metrics
```

`scripts/relay_demo.sh` generates and validates all three formats after one completed inference and
one deliberate deadline rejection.
