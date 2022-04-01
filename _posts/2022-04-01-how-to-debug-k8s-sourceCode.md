---
title: How to Debug Kubernetes Source Code?
author: Hulua
date: 2022-04-01 20:55:00 +0800
categories: [Kubernetes, Source Code]
tags: [kubernetes, debug, source code]
---

Reading the source code of kubernetes is a great way to learn the internals of kubernetes. In this post, we will explore how to use debugger (dlv) to track the source code execution flow.

# Prerequisite

To setup a debug environment, there are a few prerequisites. 
1. Linux system. 
2. A copy of Kubernetes source code (of course).
3. Learn how to use dlv.

The very first step is to ensure you are able to run a local single node k8s cluster. With help from the community, this has been simplified. Detailed steps to run a local k8s cluster is documented at https://github.com/kubernetes/community/blob/master/contributors/devel/running-locally.md.

Before reading the rest of this post, please go through the steps described in the link and make a running local k8s cluster (even if not for debugging the source, learn this process is helpful to quickly start a k8s cluster for other test purpose. You do not have to install minikube to start a local cluster...). 

If things work out as expected, try execute:

```console
sudo env "PATH=$PATH" hack/local-up-cluster.sh -O
```
Here I have `env "PATH=$PATH"` because I installed etcd under my home directory, so I reserved the path environment such that etcd can be found under `sudo`. This is not nessesary if you have etcd installed under system binary path. Another thing here is `-O` since I already compiled the source code. This is documented in the link above. You will be able to see something like:

```console
To start using your cluster, you can open up another terminal/tab and run:

  export KUBECONFIG=/var/run/kubernetes/admin.kubeconfig
  cluster/kubectl.sh

Alternatively, you can write to the default kubeconfig:

  export KUBERNETES_PROVIDER=local

  cluster/kubectl.sh config set-cluster local --server=https://localhost:6443 --certificate-authority=/var/run/kubernetes/server-ca.crt
  cluster/kubectl.sh config set-credentials myself --client-key=/var/run/kubernetes/client-admin.key --client-certificate=/var/run/kubernetes/client-admin.crt
  cluster/kubectl.sh config set-context local --cluster=local --user=myself
  cluster/kubectl.sh config use-context local
  cluster/kubectl.sh
```

Great! Now you are able to start a local k8s cluster from source code! (please also set the kubeconfig as prompted, `export KUBECONFIG=/var/run/kubernetes/admin.kubeconfig`). Now let's first look at what is happenning under the hood.


```console
ps -ef|grep kube

root      3731  3213 16 15:20 pts/6    00:00:21 ~/kubernetes/_output/bin/kube-apiserver --authorization-mode=Node,RBAC  --cloud-provider= --cloud-config=   --v=3 --vmodule= --audit-policy-file=/tmp/kube-audit-policy-file --audit-log-path=/tmp/kube-apiserver-audit.log --authorization-webhook-config-file= --authentication-token-webhook-config-file= --cert-dir=/var/run/kubernetes --egress-selector-config-file=/tmp/kube_egress_selector_configuration.yaml --client-ca-file=/var/run/kubernetes/client-ca.crt --kubelet-client-certificate=/var/run/kubernetes/client-kube-apiserver.crt --kubelet-client-key=/var/run/kubernetes/client-kube-apiserver.key --service-account-key-file=/tmp/kube-serviceaccount.key --service-account-lookup=true --service-account-issuer=https://kubernetes.default.svc --service-account-jwks-uri=https://kubernetes.default.svc/openid/v1/jwks --service-account-signing-key-file=/tmp/kube-serviceaccount.key --enable-admission-plugins=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,Priority,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota --disable-admission-plugins= --admission-control-config-file= --bind-address=0.0.0.0 --secure-port=6443 --tls-cert-file=/var/run/kubernetes/serving-kube-apiserver.crt --tls-private-key-file=/var/run/kubernetes/serving-kube-apiserver.key --storage-backend=etcd3 --storage-media-type=application/vnd.kubernetes.protobuf --etcd-servers=http://127.0.0.1:2379 --service-cluster-ip-range=10.0.0.0/24 --feature-gates=AllAlpha=false --external-hostname=localhost --requestheader-username-headers=X-Remote-User --requestheader-group-headers=X-Remote-Group --requestheader-extra-headers-prefix=X-Remote-Extra- --requestheader-client-ca-file=/var/run/kubernetes/request-header-ca.crt --requestheader-allowed-names=system:auth-proxy --proxy-client-cert-file=/var/run/kubernetes/client-auth-proxy.crt --proxy-client-key-file=/var/run/kubernetes/client-auth-proxy.key --cors-allowed-origins=/127.0.0.1(:[0-9]+)?$,/localhost(:[0-9]+)?$


root      4197  3213  3 15:20 pts/6    00:00:04 ~/kubernetes/_output/bin/kube-controller-manager --v=3 --vmodule= --service-account-private-key-file=/tmp/kube-serviceaccount.key --service-cluster-ip-range=10.0.0.0/24 --root-ca-file=/var/run/kubernetes/server-ca.crt --cluster-signing-cert-file=/var/run/kubernetes/client-ca.crt --cluster-signing-key-file=/var/run/kubernetes/client-ca.key --enable-hostpath-provisioner=false --pvclaimbinder-sync-period=15s --feature-gates=AllAlpha=false --cloud-provider= --cloud-config= --configure-cloud-routes=true --authentication-kubeconfig /var/run/kubernetes/controller.kubeconfig --authorization-kubeconfig /var/run/kubernetes/controller.kubeconfig --kubeconfig /var/run/kubernetes/controller.kubeconfig --use-service-account-credentials --controllers=* --leader-elect=false --cert-dir=/var/run/kubernetes --master=https://localhost:6443

root      4200  3213  1 15:20 pts/6    00:00:01 ~/kubernetes/_output/bin/kube-scheduler --v=3 --config=/tmp/kube-scheduler.yaml --feature-gates=AllAlpha=false --authentication-kubeconfig /var/run/kubernetes/scheduler.kubeconfig --authorization-kubeconfig /var/run/kubernetes/scheduler.kubeconfig --master=https://localhost:6443

root      4487  3213  0 15:20 pts/6    00:00:00 sudo -E ~/kubernetes/_output/bin/kubelet --v=3 --vmodule= --container-runtime=docker --hostname-override=127.0.0.1 --cloud-provider= --cloud-config= --bootstrap-kubeconfig=/var/run/kubernetes/kubelet.kubeconfig --kubeconfig=/var/run/kubernetes/kubelet-rotated.kubeconfig --config=/tmp/kubelet.yaml


root      4492  4487  7 15:20 pts/6    00:00:09 ~/kubernetes/_output/bin/kubelet --v=3 --vmodule= --container-runtime=docker --hostname-override=127.0.0.1 --cloud-provider= --cloud-config= --bootstrap-kubeconfig=/var/run/kubernetes/kubelet.kubeconfig --kubeconfig=/var/run/kubernetes/kubelet-rotated.kubeconfig --config=/tmp/kubelet.yaml

root      6203  3213  0 15:20 pts/6    00:00:00 sudo ~/kubernetes/_output/bin/kube-proxy --v=3 --config=/tmp/kube-proxy.yaml --master=https://localhost:6443
root      6550  6203  0 15:20 pts/6    00:00:00 ~/kubernetes/_output/bin/kube-proxy --v=3 --config=/tmp/kube-proxy.yaml --master=https://localhost:6443


```

These processess are the components of a k8s cluster. Namely, we have the apiserver, controller manager, scheduler, kubelet and proxy. Note that here for kubelet and proxy, we have two processes for each. I guess this is to simulate the case in worker node, we only run kubelet and proxy, and the other pair is assumed to be in master node.



# Debug kubelet

## 1. Launch Process

To debug kubelet, we will first compile a copy of the kubelet with debug infomation. This can be done by `make GOLDFLAGS="" WHAT="cmd/kubelet"`.

Now we kill the process "sudo -E ~/kubernetes/_output/bin/kubelet....". For the above example, we `sudo kill 4487`. After that, we restart the process using dlv:

```console

sudo env "PATH=$PATH" dlv exec ~/kubernetes/_output/bin/kubelet -- --v=3 --vmodule= --container-runtime=docker --hostname-override=127.0.0.1 --cloud-provider= --cloud-config= --bootstrap-kubeconfig=/var/run/kubernetes/kubelet.kubeconfig --kubeconfig=/var/run/kubernetes/kubelet-rotated.kubeconfig --config=/tmp/kubelet.yaml

Type 'help' for list of commands.
(dlv)
```

We simply copy the original starting parameters and add prefix `sudo env "PATH=$PATH" dlv exec`, as such, it will use `dlv` to start the process. Be very careful with the format when using dlv to pass parameters with `--`.

Now we can set a break point at `main.main`, and debug like any other programs using dlv:

```console
(dlv) b main.main
Breakpoint 1 set at 0x3766166 for main.main() _output/local/go/src/k8s.io/kubernetes/cmd/kubelet/kubelet.go:39
(dlv) c
2022-03-31T22:47:20-05:00 error layer=debugger error loading binary "/lib/x86_64-linux-gnu/libpthread.so.0": could not parse .eh_frame section: unknown CIE_id 0x9daad7e4 at 0x0
> main.main() _output/local/go/src/k8s.io/kubernetes/cmd/kubelet/kubelet.go:39 (hits goroutine(1):1 total:1) (PC: 0x3766166)
Warning: debugging optimized function
    34:         _ "k8s.io/component-base/metrics/prometheus/restclient"
    35:         _ "k8s.io/component-base/metrics/prometheus/version" // for version metric registration
    36:         "k8s.io/kubernetes/cmd/kubelet/app"
    37: )
    38:
=>  39: func main() {
    40:         command := app.NewKubeletCommand()
    41:
    42:         // kubelet uses a config file and does its own special
    43:         // parsing of flags and that config file. It initializes
    44:         // logging after it is done with that. Therefore it does
(dlv)
```

One thing to note, please do every command at the source code direcotry to avoid any source code loading issues.

Bascially, all the commands for k8s using Cobra command. For kubelet, the eventual entry point is located at `app.Run`, we can set another break point and go there:

```console

(dlv) b app.Run
Breakpoint 2 set at 0x375f292 for k8s.io/kubernetes/cmd/kubelet/app.Run() ./_output/local/go/src/k8s.io/kubernetes/cmd/kubelet/app/server.go:444
(dlv) c
Flag --cloud-provider has been deprecated, will be removed in 1.23, in favor of removing cloud provider code from Kubelet.
Flag --cloud-config has been deprecated, will be removed in 1.23, in favor of removing cloud provider code from Kubelet.
Flag --cloud-provider has been deprecated, will be removed in 1.23, in favor of removing cloud provider code from Kubelet.
Flag --cloud-config has been deprecated, will be removed in 1.23, in favor of removing cloud provider code from Kubelet.
I0331 23:08:16.519486   26991 mount_linux.go:222] Detected OS with systemd
> k8s.io/kubernetes/cmd/kubelet/app.Run() ./_output/local/go/src/k8s.io/kubernetes/cmd/kubelet/app/server.go:444 (hits goroutine(1):1 total:1) (PC: 0x375f292)
Warning: debugging optimized function
   439:
   440: // Run runs the specified KubeletServer with the given Dependencies. This should never exit.
   441: // The kubeDeps argument may be nil - if so, it is initialized from the settings on KubeletServer.
   442: // Otherwise, the caller is assumed to have set up the Dependencies object and a default one will
   443: // not be generated.
=> 444: func Run(ctx context.Context, s *options.KubeletServer, kubeDeps *kubelet.Dependencies, featureGate featuregate.FeatureGate) error {
   445:         // To help debugging, immediately log version
   446:         klog.InfoS("Kubelet version", "kubeletVersion", version.Get())
   447:         if err := initForOS(s.KubeletFlags.WindowsService, s.KubeletFlags.WindowsPriorityClass); err != nil {
   448:                 return fmt.Errorf("failed OS init: %w", err)
   449:         }
```

Up to this point, you can go ahead step by step and see how kubelet is started.  If you track all the way down, you will reach `startKubelet`:

```console

(dlv) list
> k8s.io/kubernetes/cmd/kubelet/app.RunKubelet() ./_output/local/go/src/k8s.io/kubernetes/cmd/kubelet/app/server.go:1230 (PC: 0x3764ac2)
Warning: debugging optimized function
  1225:                 if _, err := k.RunOnce(podCfg.Updates()); err != nil {
  1226:                         return fmt.Errorf("runonce failed: %w", err)
  1227:                 }
  1228:                 klog.InfoS("Started kubelet as runonce")
  1229:         } else {
=>1230:                 startKubelet(k, podCfg, &kubeServer.KubeletConfiguration, kubeDeps, kubeServer.EnableServer)
  1231:                 klog.InfoS("Started kubelet")
  1232:         }
  1233:         return nil
```

The function `startKubelet` is the final starting point. After line `1230`, kubelet will be started and working.  We can take a look at this entry function:

```go
func startKubelet(k kubelet.Bootstrap, podCfg *config.PodConfig, kubeCfg *kubeletconfiginternal.KubeletConfiguration, kubeDeps *kubelet.Dependencies, enableServer bool) {
	// start the kubelet, actual work is done here
	go k.Run(podCfg.Updates())

	// start the kubelet server, for HTTP purpose
	if enableServer {
		go k.ListenAndServe(kubeCfg, kubeDeps.TLSOptions, kubeDeps.Auth)
	}
	if kubeCfg.ReadOnlyPort > 0 {
		go k.ListenAndServeReadOnly(netutils.ParseIPSloppy(kubeCfg.Address), uint(kubeCfg.ReadOnlyPort))
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.KubeletPodResources) {
		go k.ListenAndServePodResources()
	}
}

```

Here `k` is actually the `Kubelet` object returned by `createAndInitKubelet`, which is a struct contains many components, and `kubeCfg` is the configuration, and `kubeDeps` is the dependencies (sounds like a runtime context for kubelet). All the work until this point is just to prepare the runtime context. 

In `startKubelet`, the runtime context is ready, so we can eventually make kubelet running.  The main function called is `k.Run`, here it will start each individual compoent such as volume manager, pod lifecycle event generator, etc. Eventually, it goes to a `syncLoop`, which will start to watch the API server and process pod life cycle events.

```go

// Run starts the kubelet reacting to config updates
func (kl *Kubelet) Run(updates <-chan kubetypes.PodUpdate) {
	if kl.logServer == nil {
		kl.logServer = http.StripPrefix("/logs/", http.FileServer(http.Dir("/var/log/")))
	}
	if kl.kubeClient == nil {
		klog.InfoS("No API server defined - no node status update will be sent")
	}

	// Start the cloud provider sync manager
	if kl.cloudResourceSyncManager != nil {
		go kl.cloudResourceSyncManager.Run(wait.NeverStop)
	}

	if err := kl.initializeModules(); err != nil {
		kl.recorder.Eventf(kl.nodeRef, v1.EventTypeWarning, events.KubeletSetupFailed, err.Error())
		klog.ErrorS(err, "Failed to initialize internal modules")
		os.Exit(1)
	}

	// Start volume manager
	go kl.volumeManager.Run(kl.sourcesReady, wait.NeverStop)

	if kl.kubeClient != nil {
		// Introduce some small jittering to ensure that over time the requests won't start
		// accumulating at approximately the same time from the set of nodes due to priority and
		// fairness effect.
		go wait.JitterUntil(kl.syncNodeStatus, kl.nodeStatusUpdateFrequency, 0.04, true, wait.NeverStop)
		go kl.fastStatusUpdateOnce()

		// start syncing lease
		go kl.nodeLeaseController.Run(wait.NeverStop)
	}
	go wait.Until(kl.updateRuntimeUp, 5*time.Second, wait.NeverStop)

	// Set up iptables util rules
	if kl.makeIPTablesUtilChains {
		kl.initNetworkUtil()
	}

	// Start component sync loops.
	kl.statusManager.Start()

	// Start syncing RuntimeClasses if enabled.
	if kl.runtimeClassManager != nil {
		kl.runtimeClassManager.Start(wait.NeverStop)
	}

	// Start the pod lifecycle event generator.
	kl.pleg.Start()
	kl.syncLoop(updates, kl)
}

```

Now let's take a look at `syncLoop`. 

```go

// syncLoop is the main loop for processing changes. It watches for changes from
// three channels (file, apiserver, and http) and creates a union of them. For
// any new change seen, will run a sync against desired state and running state. If
// no changes are seen to the configuration, will synchronize the last known desired
// state every sync-frequency seconds. Never returns.
func (kl *Kubelet) syncLoop(updates <-chan kubetypes.PodUpdate, handler SyncHandler) {
	klog.InfoS("Starting kubelet main sync loop")
	// The syncTicker wakes up kubelet to checks if there are any pod workers
	// that need to be sync'd. A one-second period is sufficient because the
	// sync interval is defaulted to 10s.
	syncTicker := time.NewTicker(time.Second)
	defer syncTicker.Stop()
	housekeepingTicker := time.NewTicker(housekeepingPeriod)
	defer housekeepingTicker.Stop()
	plegCh := kl.pleg.Watch() //The channel to watch POD life cycle events
	const (
		base   = 100 * time.Millisecond
		max    = 5 * time.Second
		factor = 2
	)
	duration := base
	// Responsible for checking limits in resolv.conf
	// The limits do not have anything to do with individual pods
	// Since this is called in syncLoop, we don't need to call it anywhere else
	if kl.dnsConfigurer != nil && kl.dnsConfigurer.ResolverConfig != "" {
		kl.dnsConfigurer.CheckLimitsForResolvConf()
	}

	for {
		if err := kl.runtimeState.runtimeErrors(); err != nil {
			klog.ErrorS(err, "Skipping pod synchronization")
			// exponential backoff
			time.Sleep(duration)
			duration = time.Duration(math.Min(float64(max), factor*float64(duration)))
			continue
		}
		// reset backoff if we have a success
		duration = base

		kl.syncLoopMonitor.Store(kl.clock.Now())
		if !kl.syncLoopIteration(updates, handler, syncTicker.C, housekeepingTicker.C, plegCh) {
			break
		}
		kl.syncLoopMonitor.Store(kl.clock.Now())
	}
}
```

Here, `updates` is the channel to receive pod update, it will be passed to `syncLoopIteration`, and there all the pod related events are processed.


## 2. Observe how a pod is created

Next, we first set a break point at `syncLoopIteration`:

```console

(dlv) b b HandlePodAdditions
Breakpoint 7 (enabled) at 0x3725cea for k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodAdditions() ./_output/local/go/src/k8s.io/kubernetes/pkg/kubelet/kubelet.go:2204 (0)

```
Then we are ready to go. After `c` command, you will see lots of logs.

```console
(dlv) c

...
d-bpjkz" containerName="coredns"
I0401 12:21:21.597873   22459 kubelet_pods.go:1076] "Clean up pod workers for terminated pods"
I0401 12:21:21.597923   22459 kubelet_pods.go:1105] "Clean up probes for terminating and terminated pods"
I0401 12:21:21.604405   22459 kubelet_pods.go:1142] "Clean up orphaned pod statuses"
I0401 12:21:21.610547   22459 kubelet_pods.go:1161] "Clean up orphaned pod directories"
I0401 12:21:21.610831   22459 kubelet_pods.go:1172] "Clean up orphaned mirror pods"
I0401 12:21:21.610858   22459 kubelet_pods.go:1179] "Clean up orphaned pod cgroups"
...

```

Now open another terminal, and set the KUBECONFIG. after that, we create a deployment:

```console
 kubectl apply -f https://k8s.io/examples/application/simple_deployment.yaml
  
deployment.apps/nginx-deployment created
```

Then go back to the debug console, we can see:

```consle
  2202: // HandlePodAdditions is the callback in SyncHandler for pods being added from
  2203: // a config source.
=>2204: func (kl *Kubelet) HandlePodAdditions(pods []*v1.Pod) {
  2205:         start := kl.clock.Now()
  2206:         sort.Sort(sliceutils.PodsByCreationTime(pods))
  2207:         for _, pod := range pods {
  2208:                 existingPods := kl.podManager.GetPods()
  2209:                 // Always add the pod to the pod manager. Kubelet relies on the pod
```

However, to create a pod, it is not as easy as just call some function and a pod is created there. The process is really really complex. Below is just a summary of the processes. We will stop here for this post. Interested readers can follow the source code and dig more details.