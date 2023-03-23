package myapp_prometheus

import (
	"math/rand"

	"github.com/prometheus/client_golang/prometheus"
)

// =========== ClusterManager Exporter ===========
type ClusterManager struct {
	ClusterName  string
	OOMCountDesc *prometheus.Desc
	RAMUsageDesc *prometheus.Desc
}

func NewClusterManager(clusterName string) *ClusterManager {
	return &ClusterManager{
		ClusterName: clusterName,
		OOMCountDesc: prometheus.NewDesc(
			"a_clustermanager_oom_crashes_total",
			"Number of OOM crashes.",
			[]string{"host"}, prometheus.Labels{"clusterName": clusterName},
		),
		RAMUsageDesc: prometheus.NewDesc(
			"a_clustermanager_ram_usage_bytes",
			"RAM usage as reported to the cluster manager.",
			[]string{"host"}, prometheus.Labels{"clusterName": clusterName},
		),
	}
}

func (c *ClusterManager) receiveMetric() (
	oomCountByHost map[string]int32, ramUsageByHost map[string]float64,
) {
	oomCountByHost = map[string]int32{
		"foo.example.org": int32(rand.Int31n(1000)),
		"bar.example.org": int32(rand.Int31n(1000)),
	}
	ramUsageByHost = map[string]float64{
		"foo.example.org": rand.Float64() * 100,
		"bar.example.org": rand.Float64() * 100,
	}
	return
}

// =========== Collector ===========
func (c *ClusterManager) Collect(ch chan<- prometheus.Metric) {

	oomCountByHost, ramUsageByHost := c.receiveMetric()

	for host, oomCount := range oomCountByHost {
		ch <- prometheus.MustNewConstMetric(
			c.OOMCountDesc,
			prometheus.CounterValue,
			float64(oomCount), host,
		)
	}
	for host, ramUsage := range ramUsageByHost {
		ch <- prometheus.MustNewConstMetric(
			c.RAMUsageDesc,
			prometheus.GaugeValue,
			ramUsage, host,
		)
	}
}

func (c *ClusterManager) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.OOMCountDesc
	ch <- c.RAMUsageDesc
}
