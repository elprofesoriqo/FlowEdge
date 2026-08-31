# Relay observability

Relay records metrics without allocating on request submission, scheduling, worker completion, or
action publication. The fixed registry includes:

- received, corrupt, accepted, published, and dropped messages;
- complete, cancelled, failed, stale, deadline, capacity, and expiry outcomes;
- queue-depth and busy-worker high-water marks;
- worker failures; and
- power-of-two queue, execution, and end-to-end latency histograms in nanoseconds.

## Choose an export path

| You need | Exporter | What to do with it |
|---|---|---|
| Prometheus dashboards and alerts | `--metrics-prometheus` | Point node exporter's textfile collector at the output directory |
| A compact run artifact for scripts or CI | `--metrics-json` | Archive or compare the JSON file after each run |
| An OpenTelemetry Collector pipeline | `--metrics-otlp-json` | POST the generated OTLP/HTTP JSON body to a Collector |

Counters are cumulative for the current daemon process. Histogram buckets are powers of two, so use
them for stable regression and SLO bands rather than treating a bucket boundary as an exact sample.

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

A successful export contains nonzero `received`, `accepted`, and `published` counters after a normal
request. A deliberate impossible deadline should increase `rejected_deadline` without increasing
`failed`. If the daemon is terminated without graceful shutdown and periodic export is disabled, no
final file update is expected.

The OTLP file is an OTLP/HTTP JSON request body. An existing deployment adapter may post it to a
Collector, for example:

```bash
curl -H 'Content-Type: application/json' --data-binary @run-otlp.json \
  http://127.0.0.1:4318/v1/metrics
```

`scripts/relay_demo.sh` generates and validates all three formats after one completed inference and
one deliberate deadline rejection.
