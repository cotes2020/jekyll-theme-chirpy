---
title: "From anything goes to restricted: practical PodSecurity hardening in Kubernetes"
date: 2025-01-20 09:00:00 +0100
categories: 
  - kubernetes
  - podsecurity
  - platform security
tags:
  - kubernetes
  - security
  - podsecurity
  - hardening
  - admission
  - baseline
  - restricted
---

In many clusters we see the same pattern over and over again:

- Containers running as root
- `privileged: true` sprinkled around “because otherwise it didn’t work”
- `hostPath` mounts into the node filesystem
- Linux capabilities that were never cleaned up

And usually: **no PodSecurity enforcement at all**.

PodSecurityPolicies are deprecated, PodSecurityAdmission is “something for later”, and in the meantime everything just keeps running.

In this post we’ll walk through how to move step by step from “anything goes” to a **realistic `restricted` setup**, without breaking your entire platform in one shot.

We’ll cover:

1. PodSecurity profiles: `privileged`, `baseline`, `restricted`
2. Enabling PodSecurity per namespace
3. Seeing what breaks _before_ you enforce
4. Typical problems and how to fix them:
   - containers running as root
   - `privileged: true` and `hostPath`
   - capabilities and seccomp
5. A migration strategy per environment (dev / staging / prod)

This post assumes you’re on Kubernetes 1.25+ or a managed cluster where PodSecurityAdmission is available.

---

## 1. PodSecurity profiles in a nutshell

Kubernetes ships with three standard PodSecurity levels:

- **`privileged`**  
  Almost everything is allowed. Compatible with legacy workloads, but very few guarantees.

- **`baseline`**  
  Protects against the worst foot-guns. Allows many “normal” workloads to run.

- **`restricted`**  
  Strict hardening. Pods must run as non-root, have limited capabilities, no hostPath, etc.

In practice:

- `privileged`: only for very specific infra/host-management workloads — and even then only in dedicated namespaces.
- `baseline`: a good minimum level for most dev/test environments.
- `restricted`: your target for most production workloads.

> Important: PodSecurityAdmission works **per namespace**, not per individual pod.  
> You label namespaces to choose the profile.

---

## 2. Enabling PodSecurity per namespace

PodSecurityAdmission uses labels on namespaces:

- `pod-security.kubernetes.io/enforce`
- `pod-security.kubernetes.io/audit`
- `pod-security.kubernetes.io/warn`

With values: `privileged`, `baseline` or `restricted`.

### Example: stricter in production

```bash
# production namespace
kubectl label namespace team-a-backend   pod-security.kubernetes.io/enforce=restricted   pod-security.kubernetes.io/audit=restricted   pod-security.kubernetes.io/warn=baseline
```

Meaning:

- `enforce=restricted`  
  Pods that don’t meet the restricted requirements are rejected.

- `audit=restricted`  
  Violations are recorded in the audit log.

- `warn=baseline`  
  The API server returns a warning if a pod doesn’t meet baseline.

### Example: gradual rollout in dev

```bash
# dev namespace
kubectl label namespace team-a-backend-dev   pod-security.kubernetes.io/enforce=baseline   pod-security.kubernetes.io/audit=restricted   pod-security.kubernetes.io/warn=restricted
```

Meaning:

- `enforce=baseline`  
  Gross misconfigurations are blocked.

- `audit + warn = restricted`  
  You already see which pods would fail `restricted` without actually blocking them.

---

## 3. Look first, then block

You **never** want to flip `restricted: enforce` on all namespaces in one go.

A safer process:

1. **Inventory**

   - Which namespaces exist?
   - Which are true production?
   - Which are dev/test/sandbox?

   ```bash
   kubectl get ns --show-labels
   ```

2. **Start in a non-critical dev namespace**

   ```bash
   kubectl label namespace playground      pod-security.kubernetes.io/enforce=baseline      pod-security.kubernetes.io/audit=restricted      pod-security.kubernetes.io/warn=restricted
   ```

3. **(Re)deploy existing workloads**  
   See which errors and warnings you get.

4. **Fix issues in dev**  
   Adjust manifests until `restricted` stops complaining.

5. **Repeat for staging/acceptance**  
   Only then move on to production.

---

## 4. Typical PodSecurity issues and how to fix them

Below are the most common problems we see once you turn on `restricted`, and how to address them.

### 4.1 Containers running as root

**Symptom**  
PodSecurity complains about `runAsNonRoot` / UID 0.

A typical “old style” deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api-service
  template:
    metadata:
      labels:
        app: api-service
    spec:
      containers:
        - name: api
          image: myregistry.local/api-service:latest
          ports:
            - containerPort: 8080
```

The container runs as root because:

- the image default user is root, and
- no `securityContext` is set.

**Fix 1 – Pod-level securityContext**

If your image supports non-root (strongly recommended):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api-service
  template:
    metadata:
      labels:
        app: api-service
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
        - name: api
          image: myregistry.local/api-service:latest
          ports:
            - containerPort: 8080
```

> Pro tip: update your Dockerfile so the default user is non-root:
>
> ```dockerfile
> FROM node:20-alpine
> WORKDIR /app
> COPY package*.json ./
> RUN npm ci
> COPY . .
> RUN addgroup -S app && adduser -S app -G app
> USER app
> CMD ["npm", "start"]
> ```

If you truly can’t change the image yet (legacy), you might temporarily relax settings in dev — but the goal should always be: **non-root by default**.

---

### 4.2 `privileged: true` and hostPath

**Symptom**  
Pods use:

- `securityContext.privileged: true`
- `hostPath` volumes to paths like `/var/run/docker.sock` or `/`

PodSecurity `restricted` will block this.

Example:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-debug
spec:
  selector:
    matchLabels:
      app: node-debug
  template:
    metadata:
      labels:
        app: node-debug
    spec:
      containers:
        - name: debug
          image: alpine
          securityContext:
            privileged: true
          volumeMounts:
            - name: rootfs
              mountPath: /host
      volumes:
        - name: rootfs
          hostPath:
            path: /
```

You generally only want this kind of workload in **very specific** infra contexts, not in regular app namespaces.

**Strategy**

1. **Isolate infra tools in their own namespace**, e.g. `infra-tools`:

   ```bash
   kubectl label ns infra-tools      pod-security.kubernetes.io/enforce=privileged      pod-security.kubernetes.io/audit=baseline      pod-security.kubernetes.io/warn=baseline
   ```

   - Node agents, CNI DaemonSets, storage drivers, etc. live here.

2. In _all other_ namespaces:

   - no `hostPath` volumes,
   - no `privileged: true`.

**Blocking hostPath with policies**

If you use Kyverno, you can enforce this for app namespaces:

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-hostpath
spec:
  validationFailureAction: enforce
  background: true
  rules:
    - name: hostpath-not-allowed
      match:
        any:
          - resources:
              kinds:
                - Pod
              namespaces:
                - team-a-backend
                - team-a-frontend
                - team-b-services
      validate:
        message: "hostPath volumes are not allowed in application namespaces."
        pattern:
          spec:
            =(volumes):
              - X(hostPath): "null"
```

---

### 4.3 Capabilities and seccomp

`restricted` requires:

- a seccomp profile (`RuntimeDefault` or a specific one),
- no extra Linux capabilities beyond what is strictly needed.

Minimal example:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api-service
  template:
    metadata:
      labels:
        app: api-service
    spec:
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: api
          image: myregistry.local/api-service:latest
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop:
                - ALL
```

If you do need a capability (for example `NET_BIND_SERVICE` to bind to a low port):

```yaml
securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
    add:
      - NET_BIND_SERVICE
```

> Tip: start from `drop: ALL` and **add** the minimal required capabilities, not the other way around.

---

## 5. Migration strategy per environment

Here’s a pragmatic migration path we use in real environments.

### Phase 1 – Observation (dev/sandbox)

- Pick 1–2 **non-critical namespaces** in a dev cluster.
- Label them:

  ```bash
  kubectl label ns playground     pod-security.kubernetes.io/enforce=baseline     pod-security.kubernetes.io/audit=restricted     pod-security.kubernetes.io/warn=restricted
  ```

- Deploy some workloads, see what breaks.
- Update manifests until `restricted` would be happy.

### Phase 2 – Roll out to staging

- Apply the same labels to a couple of namespaces in staging.
- Ensure CI/CD uses the updated manifests (including securityContext).
- Fix any issues that only appear in stage (old jobs, tools, etc.).

### Phase 3 – Production, namespace by namespace

- Choose a subset of production namespaces where you have good control.
- Label those with:

  ```bash
  kubectl label ns team-a-backend     pod-security.kubernetes.io/enforce=restricted     pod-security.kubernetes.io/audit=restricted     pod-security.kubernetes.io/warn=baseline
  ```

- Keep infra/host-related workloads in their own namespaces with a milder profile (`baseline`/`privileged`).
- Monitor failed deployments closely and iterate quickly based on dev/stage experience.

### Phase 4 – Standardization

- Define clear rules:
  - New namespaces must get PodSecurity labels by default.
  - New services must follow a minimal securityContext (non-root, seccomp, capabilities).
- Bake this into:
  - Helm charts,
  - templates,
  - CI/CD checks (e.g. policy that rejects deployments without `runAsNonRoot`).

---

## 6. Are you “restricted-ready”?

Some hard questions to see where you stand:

- Do your critical app namespaces already have:
  - `pod-security.kubernetes.io/enforce=baseline` or `restricted`?
- Can your key workloads run with:
  - `runAsNonRoot: true`,
  - `allowPrivilegeEscalation: false`,
  - `seccompProfile: RuntimeDefault`,
  - `capabilities.drop: [ALL]` (and a very small `add:` list)?
- Do infra/host-level tools live in their own namespaces with a different profile?
- Are hostPath usage and privileged pods the **exception**, not the norm?

If the honest answer is “no” to most of these, an attacker who gets a foothold in one pod can probably do a lot more than you’d like.

---

## Try this in your own cluster

You can use the examples from this post to experiment safely:

1. Create a test namespace, label it with `baseline/enforce` and `restricted/audit`.
2. Deploy a simple app and inspect the warnings.
3. Gradually add:
   - `runAsNonRoot`,
   - `seccompProfile`,
   - `allowPrivilegeEscalation: false`,
   - `capabilities.drop: [ALL]`,
   until `restricted` is happy.
4. Carry these patterns over into your real Helm charts and manifests.

In a follow-up post, we’ll look at concrete manifest diffs: before vs after, including examples of real-world workloads (Nginx, Node.js, .NET) that we make “restricted-ready”.

For now: pick one namespace, label it, and see what breaks.  
That’s usually the cheapest and most honest security review you’ll ever run on your cluster. 🙂
