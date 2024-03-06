package myapp_prometheus

import (
	"context"
	"log"
	"math/rand"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/sts"

	"github.com/prometheus/client_golang/prometheus"
)

// myapp Core - LocalMetric struct
type LocalMetric struct {
	MetricMap map[string]OneMetricTrack
}

var mylist LocalMetric = LocalMetric{
	MetricMap: map[string]OneMetricTrack{},
}

// local collector
func localcollector(onemetric OneMetricTrack) {
	log.Println("--- Push timer to endpoint")
	funcname := onemetric.funcname
	metricname := onemetric.metricname
	if funcname != "none" {
		mylist.MetricMap[funcname] = onemetric
	} else {
		mylist.MetricMap[metricname] = onemetric
	}
	log.Println("--- metric amount:", len(mylist.MetricMap))
}

// myapp Core - RunTimeTrack struct
type OneMetricTrack struct {
	stamp       time.Time
	funcname    string
	elapsed     time.Duration
	metricname  string
	metricvalue float64
}

func GetMetricTrack(
	start time.Time,
	funcname string,
	metricname string, metricvalue float64,
) OneMetricTrack {
	var cur OneMetricTrack
	cur.stamp = start
	cur.elapsed = time.Since(start)
	if funcname != "none" {
		cur.funcname = funcname
		cur.metricname = "none"
		cur.metricvalue = float64(0)
		log.Printf("--- %s took %s", cur.funcname, cur.elapsed)
	} else {
		cur.funcname = "none"
		cur.metricname = metricname
		cur.metricvalue = metricvalue
		log.Printf("--- %s: %s", cur.metricname, cur.metricvalue)
	}
	// Push timer to endpoint
	localcollector(cur)
	return cur
}

// =========== Custom Metric Exporter ===========
type myappCoreMetrics struct {
	ClusterName string

	Metric_a_Desc                      *prometheus.Desc
	myapp_core_getClients_runtime_Desc *prometheus.Desc

	// // PeerReceivedBytes is the metric that counts the number of bytes received from a given peer.
	// PeerReceivedBytesDesc *prometheus.Desc
	// // PeerTransmittedBytes is the metric that counts the number of bytes transmitted to a given peer.
	// PeerTransmittedBytesDesc *prometheus.Desc
	// // PeerLatency is the metric that exposes the latency towards a given peer.
	// PeerLatencyDesc *prometheus.Desc
	// // PeerIsConnected is the metric that outputs the connection status.
	// PeerIsConnectedDesc *prometheus.Desc
	// // MetricsLabels is the labels that are used for the metrics.
	MetricsLabels []string
}

func NewmyappCoreMetrics(clusterName string) *myappCoreMetrics {
	MetricsLabels := []string{
		"driver", "device",
		"cluster_id", "cluster_name",
	}
	return &myappCoreMetrics{
		ClusterName: clusterName,

		Metric_a_Desc: prometheus.NewDesc(
			"myapp_core_metric_a",
			"Checks if connection is working.",
			[]string{"host"},
			prometheus.Labels{"clusterName": clusterName},
		),

		myapp_core_getClients_runtime_Desc: prometheus.NewDesc(
			"myapp_core_getClients_runtime",
			"Checks if getClients times",
			MetricsLabels,
			prometheus.Labels{"clusterName": clusterName},
		),

		// PeerReceivedBytesDesc: prometheus.NewDesc(
		// 	"myapp_core_liqo_peer_receive_bytes_total",
		// 	"Number of bytes received from a given peer.",
		// 	MetricsLabels,
		// 	prometheus.Labels{"clusterName": clusterName},
		// ),

		// PeerTransmittedBytesDesc: prometheus.NewDesc(
		// 	"myapp_core_liqo_peer_transmit_bytes_total",
		// 	"Number of bytes transmitted to a given peer.",
		// 	MetricsLabels,
		// 	prometheus.Labels{"clusterName": clusterName},
		// ),

		// PeerLatencyDesc: prometheus.NewDesc(
		// 	"myapp_core_liqo_peer_latency_us",
		// 	"Latency of a given peer in microseconds.",
		// 	MetricsLabels,
		// 	prometheus.Labels{"clusterName": clusterName},
		// ),

		// PeerIsConnectedDesc: prometheus.NewDesc(
		// 	"myapp_core_liqo_peer_is_connected",
		// 	"Checks if connection is working.",
		// 	MetricsLabels, prometheus.Labels{"clusterName": clusterName},
		// 	// []string{"host"}, prometheus.Labels{},
		// ),
	}
}

func (m *myappCoreMetrics) receivemyappMetric() (
	metric_a_ByHost map[string]int32,
	myapp_core_getClients_runtime OneMetricTrack,
	// peerReceivedBytes map[string]int32,
	// peerTransmittedBytes int32,
	// peerLatency map[string]int32,
	// peerIsConnected map[string]int32,
) {
	for key, element := range mylist.MetricMap {
		log.Println("Key:", key, "=>", "Element:", element)
		myapp_core_getClients_runtime = mylist.MetricMap["myapp_core_getClients_runtime"]
	}

	metric_a_ByHost = map[string]int32{
		"foo.example.org": int32(rand.Int31n(1000)),
		"bar.example.org": int32(rand.Int31n(1000)),
	}
	// peerReceivedBytes = map[[]string]int32{
	// 	"DriverName":  88888,
	// 	"deviceName": 88888,
	// 	"ClusterID":   88888,
	// 	"ClusterName": 88888,
	// }

	// mylable := []string{

	// }
	// peerTransmittedBytes = 99

	// peerLatency = map[string]int32{
	// 	"good": int32(rand.Int31n(1000)),
	// 	"bad":  int32(rand.Int31n(1000)),
	// }

	// peerIsConnected = map[string]int32{
	// 	"suc":    int32(rand.Int31n(1000)),
	// 	"failed": int32(rand.Int31n(1000)),
	// }

	return
}

func (m *myappCoreMetrics) Collect(ch chan<- prometheus.Metric) {

	metric_a_ByHost, myapp_core_getClients_runtime := m.receivemyappMetric()

	for host, metric_a := range metric_a_ByHost {
		ch <- prometheus.MustNewConstMetric(
			m.Metric_a_Desc,
			prometheus.CounterValue,
			float64(metric_a), host,
		)
	}

	labels := []string{
		"Driver1", "device1",
		"123", "Cluster123",
	}

	log.Printf(myapp_core_getClients_runtime.funcname)

	ch <- prometheus.MustNewConstMetric(
		m.myapp_core_getClients_runtime_Desc,
		prometheus.CounterValue,
		myapp_core_getClients_runtime.metricvalue, labels...,
	)

	// ch <- prometheus.MustNewConstMetric(
	// 	m.PeerReceivedBytesDesc,
	// 	prometheus.CounterValue,
	// 	float64(999999), labels...,
	// )

	// ch <- prometheus.MustNewConstMetric(
	// 	m.PeerTransmittedBytesDesc,
	// 	prometheus.CounterValue,
	// 	float64(peerTransmittedBytes), labels...,
	// )

	c, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Fatal(err)

	}
	// result := 1
	// ch <- prometheus.MustNewConstMetric(
	// 	m.PeerIsConnectedDesc,
	// 	prometheus.GaugeValue,
	// 	float64(result), labels...,
	// )

	// result := 1
	// ch <- prometheus.MustNewConstMetric(
	// 	myapp_p.PeerIsConnected,
	// 	prometheus.GaugeValue,
	// 	float64(result), "foo.example.org",
	// )

	stsClient := sts.NewFromConfig(c)
	// sqsClient = sqs.NewFromConfig(c)

	input := &sts.GetCallerIdentityInput{}
	i, err := stsClient.GetCallerIdentity(context.Background(), input)
	if err != nil {
		log.Printf("Retrieve error: %s", err)
	}
	log.Printf(
		"+++ GetCallerIdentity: Account: %s, ARN: %s, UserID: %s",
		*i.Account, *i.Arn, *i.UserId)

	// for host, metric_a := range peerLatency {
	// 	ch <- prometheus.MustNewConstMetric(
	// 		m.PeerLatencyDesc,
	// 		prometheus.CounterValue,
	// 		float64(metric_a), host,
	// 	)
	// }
	// for host, metric_a := range peerIsConnected {
	// 	ch <- prometheus.MustNewConstMetric(
	// 		m.PeerIsConnectedDesc,
	// 		prometheus.CounterValue,
	// 		float64(metric_a), host,
	// 	)
	// }
}

func (m *myappCoreMetrics) Describe(ch chan<- *prometheus.Desc) {
	ch <- m.Metric_a_Desc
	ch <- m.myapp_core_getClients_runtime_Desc
	// ch <- m.PeerReceivedBytesDesc
	// ch <- m.PeerTransmittedBytesDesc
	// ch <- m.PeerLatencyDesc
	// ch <- m.PeerIsConnectedDesc
}

// MetricsErrorHandler is a function that handles metrics errors.
func (m *myappCoreMetrics) MetricsErrorHandler(err error, ch chan<- prometheus.Metric) {
	// ch <- prometheus.NewInvalidMetric(m.PeerReceivedBytesDesc, err)
	// ch <- prometheus.NewInvalidMetric(m.PeerTransmittedBytesDesc, err)
	// ch <- prometheus.NewInvalidMetric(m.PeerLatencyDesc, err)
	// ch <- prometheus.NewInvalidMetric(m.PeerIsConnectedDesc, err)
}
