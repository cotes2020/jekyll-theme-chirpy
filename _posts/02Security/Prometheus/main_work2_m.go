<<<<<<< HEAD
package medusaprometheus
=======
package My_APPprometheus
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28

import (
	"github.com/prometheus/client_golang/prometheus"
)

<<<<<<< HEAD
// =========== Medusa Prometheus Step up ===========
var (
	Medusa_Prometheus_Port = ":8080"
	Medusa_Prometheus_Path = "/metrics"
=======
// =========== My_APP Prometheus Step up ===========
var (
	My_APP_Prometheus_Port = ":8080"
	My_APP_Prometheus_Path = "/metrics"
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
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
<<<<<<< HEAD
			Name: "medusa_core_input_queue_size_approx",
=======
			Name: "My_APP_core_input_queue_size_approx",
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
			Help: "Approximate messages in input Eventbridge queue",
		},
	)

	CONFIG_BATCH_QUEUE_SIZE_APPROX = prometheus.NewGauge(
		prometheus.GaugeOpts{
<<<<<<< HEAD
			Name: "medusa_core_config_batch_queue_size_approx",
=======
			Name: "My_APP_core_config_batch_queue_size_approx",
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
			Help: "Approximate messages in Config batch queue",
		},
	)

	OUTPUT_QUEUE_SIZE_APPROX = prometheus.NewGauge(
		prometheus.GaugeOpts{
<<<<<<< HEAD
			Name: "medusa_core_output_queue_size_approx",
			Help: "Approximate messages in output queue to medusa-core",
=======
			Name: "My_APP_core_output_queue_size_approx",
			Help: "Approximate messages in output queue to My_APP-core",
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
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
