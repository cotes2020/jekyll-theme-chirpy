package medusaprometheus

import (
	"github.com/prometheus/client_golang/prometheus"
)

// =========== Medusa Prometheus Step up ===========
var (
	Medusa_Prometheus_Port = ":8080"
	Medusa_Prometheus_Path = "/metrics"
)

// =========== Custom Metric ===========
var (
	GET_QUEUE_URL_ERR = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "get_queue_url_errors",
			Help: "Exceptions getting SQS queue URL",
		},
	)
	GET_CLIENTS_ERR = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "get_clients_errors",
			Help: "Exceptions getting boto clients",
		},
		[]string{"device"},
	)

	INPUT_QUEUE_SIZE_APPROX = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "medusa_core_input_queue_size_approx",
			Help: "Approximate messages in input Eventbridge queue",
		},
	)

	CONFIG_BATCH_QUEUE_SIZE_APPROX = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "medusa_core_config_batch_queue_size_approx",
			Help: "Approximate messages in Config batch queue",
		},
	)

	OUTPUT_QUEUE_SIZE_APPROX = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "medusa_core_output_queue_size_approx",
			Help: "Approximate messages in output queue to medusa-core",
		},
	)
)

// ===========
var (
	RpcDurations = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Name:       "rpc_durations_seconds",
			Help:       "RPC latency distributions.",
			Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
		},
		[]string{"service"},
	)

	cpuTemp = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "cpu_temperature_celsius",
			Help: "Current temperature of the CPU.",
		})
	hdFailures = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "hd_errors_total",
			Help: "Number of hard-disk errors.",
		},
		[]string{"device"},
	)
)
