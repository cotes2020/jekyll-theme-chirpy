package main

import (
	"net/http"

	qzPro "github.com/ocholuo/ocholuo.github.io/_posts/02Security/Prometheus/service/prometheus"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	prometheus.MustRegister(

		qzPro.CommonCounter, qzPro.FuncCounter, qzPro.VecCounter,

		// qzPro.CommonGauge, qzPro.FuncGauge, qzPro.VecGauge,
		// qzPro.CommonHistogram, qzPro.VecHistogram,
		// qzPro.CommonSummary, qzPro.VecSummary,
	)
	http.HandleFunc("/common_counter", qzPro.DealCommCounter)
	http.HandleFunc("/vec_counter", qzPro.DealVecCounter)

	// http.HandleFunc("/common_gauge", qzPro.DealCommGauge)
	// http.HandleFunc("/vec_gauge", qzPro.DealVecGauge)

	// http.HandleFunc("/common_histogram", qzPro.DealCommHistogram)
	// http.HandleFunc("/vec_histogram", qzPro.DealVecHistogram)

	// http.HandleFunc("/common_summary", qzPro.DealCommSummary)
	// http.HandleFunc("/vec_summary", qzPro.DealVecSummary)

	http.Handle("/metrics", promhttp.Handler()) // 暴露 metrics 指标
	http.ListenAndServe(":8090", nil)
}
