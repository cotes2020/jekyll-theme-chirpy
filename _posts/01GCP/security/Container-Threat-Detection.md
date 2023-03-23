




- [Monitor and secure containers](#monitor-and-secure-containers)
  - [Security Command Center](#security-command-center)

---

# Monitor and secure containers

Container Threat Detection
- monitor and secure the container deployments in Google Cloud.
- a built-in service in [Security Command Center](https://cloud.google.com/security-command-center) Premium tier.
- detects the most common container runtime attacks
- and alerts to any suspicious activity.
- includes multiple new detection capabilities and provides an API.


key findings identified by Container Threat Detection:
- **Suspicious Binary Executions**
  - can see when a binary that <font color=blue> was not part of the original container image </font> is executed, and triggers a finding
  - indicating that an attacker may have control of the workload and executing suspicious software
    - such as malware or cryptocurrency mining software
- **Suspicious Library Loaded**
  - can detect when a library that <font color=blue> was not part of the original container image </font> is loaded
  - the attacker may has control of the workload and that they are executing arbitrary code.
- **Reverse Shell**
  - monitors for processes that get started with stream redirection to a remote connected socket.
  - An attacker can use a reverse shell to communicate from a compromised workload to an attacker controlled machine and perform malicious activities
  - for example as part of a botnet.


---

## Security Command Center

Security Command Center
- a native security and risk management platform for Google Cloud.
- it provides built-in services
  - Container Threat Detection,
  - gain visibility into the cloud assets,
  - discover misconfigurations and vulnerabilities in the resources,
  - help maintain compliance based on industry standards and benchmarks.

![Container_Threat_Detectio.1018027618020533.max-2800x2800](https://i.imgur.com/T8KiDWE.png)


**Start Container Threat Detection**
- enable the built-in service in the Security Command Center with a Premium subscription.
  - ![container threat detection (1).jpg](https://storage.googleapis.com/gweb-cloudblog-publish/images/container_threat_detection_.0394005507850100.max-900x900.jpg)
- To enable a Premium subscription, contact the Google Cloud Platform sales team.
- To trigger Container Threat Detection findings in a test environment, follow the steps outlined in this [Testing Container Threat Detection guide](https://cloud.google.com/security-command-center/docs/how-to-test-container-threat-detection).








.
