package main

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"net/http"
)

type ClusterManager struct {
	Zone         string
	OOMCountDesc *prometheus.Desc
	RAMUsageDesc *prometheus.Desc
	// ... many more fields
}

// Simulate prepare the data
func (c *ClusterManager) ReallyExpensiveAssessmentOfTheSystemState() (
	oomCountByHost map[string]int, ramUsageByHost map[string]float64) {
	// Just example fake data.
	oomCountByHost = map[string]int{
		"foo.example.org": 42,
		"bar.example.org": 2001,
	}
	ramUsageByHost = map[string]float64{
		"foo.example.org": 6.023e23,
		"bar.example.org": 3.14,
	}
	return
}

// Describe simply sends the two Descs in the struct to the channel.
func (c *ClusterManager) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.OOMCountDesc
	ch <- c.RAMUsageDesc
}

func (c *ClusterManager) Collect(ch chan<- prometheus.Metric) {
	oomCountByHost, ramUsageByHost := c.ReallyExpensiveAssessmentOfTheSystemState()
	for host, oomCount := range oomCountByHost {
		ch <- prometheus.MustNewConstMetric(
			c.OOMCountDesc,
			prometheus.CounterValue,
			float64(oomCount),
			host,
		)
	}
	for host, ramUsage := range ramUsageByHost {
		ch <- prometheus.MustNewConstMetric(
			c.RAMUsageDesc,
			prometheus.GaugeValue,
			ramUsage,
			host,
		)
	}
}

// NewClusterManager creates the two Descs OOMCountDesc and RAMUsageDesc.
// Note that the zone is set as a ConstLabel.
// (It's different in each instance of the ClusterManager,
// but constant over the lifetime of an instance.)
// Then there is a variable label "host",
// since we want to partition the collected metrics by host.
// Since all Descs created in this way are consistent across instances,
// with a guaranteed distinction by the "zone" label, we can register different
// ClusterManager instances with the same registry.
func NewClusterManager(zone string) *ClusterManager {
	return &ClusterManager{
		Zone: zone,
		OOMCountDesc: prometheus.NewDesc(
			"clustermanager_oom_crashes_total",
			"Number of OOM crashes.",
			[]string{"host"},
			prometheus.Labels{"zone": zone},
		),
		RAMUsageDesc: prometheus.NewDesc(
			"clustermanager_ram_usage_bytes",
			"RAM usage as reported to the cluster manager.",
			[]string{"host"},
			prometheus.Labels{"zone": zone},
		),
	}
}

func main() {
	// Since we are dealing with custom Collector implementations, it might
	// be a good idea to try it out with a pedantic registry.
	reg := prometheus.NewPedanticRegistry()

	// Metrics have to be registered to be exposed:
	workerDB := NewClusterManager("db")
	workerCA := NewClusterManager("ca")
	reg.MustRegister(workerDB)
	reg.MustRegister(workerCA)

	http.Handle(
		"/metrics",
		promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	http.ListenAndServe(":8082", nil)
}
