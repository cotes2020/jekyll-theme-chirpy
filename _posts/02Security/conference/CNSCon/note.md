- [Conference NOTE](#conference-note)
  - [FEB 1](#feb-1)
    - [Why Developer Laptop Security Is Key to Securing Your CI/CD Pipeline](#why-developer-laptop-security-is-key-to-securing-your-cicd-pipeline)
    - [Cryptographic Agility: Preparing Modern Apps for Quantum Safety and Beyond](#cryptographic-agility-preparing-modern-apps-for-quantum-safety-and-beyond)
    - [Secure your Software Supply Chain at Scale](#secure-your-software-supply-chain-at-scale)
  - [Openssf ScoreCard](#openssf-scorecard)
    - [Feature](#feature)
    - [status](#status)
      - [use](#use)
      - [community](#community)
  - [SLSA](#slsa)


---

### Tools


![Screenshot 2023-02-02 at 23.22.13](https://i.imgur.com/LxVfCGT.jpg)

- HeartBleed
- Spectre
- MeltDown

- [Anchore](https://anchore.com/software-supply-chain-security-report-2022/)


- Software Composition Analysis (SCA)

---

## Conference NOTE

---


### FEB 1

---

#### Why Developer Laptop Security Is Key to Securing Your CI/CD Pipeline

![Screenshot 2023-02-02 at 21.39.55](https://i.imgur.com/oNhVsTU.png)

---

#### Cryptographic Agility: Preparing Modern Apps for Quantum Safety and Beyond

Cryptographic Agility 敏捷

- The ability to <font color=red> reconfigure </font> an application or system with a different cryptographic algorithm (or implementation).

Cryptographic Agility Advantages
- Transition to New Algorithms
- Change Library
- Modifying Config
- Retiring Algorithms
- Compliance Standards
- Streamline Remediation

![Screenshot 2023-02-02 at 22.04.46](https://i.imgur.com/hJ7EpsQ.png)

**Current Landscape Problems**
- Lack of visibility
- No unification
- Rearchitecting required


![Screenshot 2023-02-02 at 22.08.13](https://i.imgur.com/XguVIUU.png)

![Screenshot 2023-02-02 at 22.09.04](https://i.imgur.com/HD3MxWF.png)

**Future landscape benefits**
- Standards migration
- Compliance
- Good engineering


Harvest Now, Decrypt Later (HNDL)

![Screenshot 2023-02-02 at 22.14.28](https://i.imgur.com/oTNwof5.png)


What can you do now?
- Identify crypto libraries in organization
- Communicate policies
- Identify most valuable assets
- Plan and build for change
- Create backup plans for CA


Policy-driven cryptography compliance and configuration platform


---

#### Secure your Software Supply Chain at Scale


Recent studies
- 85 to 97% of enterprise codebase uses open source software
- Three out of Five Companies Targeted - Anchore
- 62% of Organizations Have Been Impacted by Software Supply Chain Attacks - Anchore

![Screenshot 2023-02-02 at 23.13.37](https://i.imgur.com/gLQtOH2.png)

![Screenshot 2023-02-02 at 23.14.02](https://i.imgur.com/rVzvw19.png)

![Screenshot 2023-02-02 at 23.16.21](https://i.imgur.com/xUvlJiV.png)

![Screenshot 2023-02-02 at 23.16.54](https://i.imgur.com/rXabLKB.png)

![Screenshot 2023-02-02 at 23.19.36](https://i.imgur.com/nYO0OC1.png)

![Screenshot 2023-02-02 at 23.20.10](https://i.imgur.com/0csiftH.png)

![Screenshot 2023-02-02 at 23.20.39](https://i.imgur.com/kcAcFc7.png)



software supply chain

![Screenshot 2023-02-02 at 22.22.42](https://i.imgur.com/gJy5gwN.png)

problem

![Screenshot 2023-02-02 at 22.23.21](https://i.imgur.com/Y3Cs4Ix.png)

![Screenshot 2023-02-03 at 01.08.35](https://i.imgur.com/gmem7UI.png)

![Screenshot 2023-02-03 at 01.11.09](https://i.imgur.com/qjqgGJL.png)

![Screenshot 2023-02-03 at 01.13.40](https://i.imgur.com/yAhhNYF.png)


Build time vuln assessment

![Screenshot 2023-02-03 at 01.17.05](https://i.imgur.com/TuC6V2M.png)


Production deployment verification

![Screenshot 2023-02-03 at 01.18.25](https://i.imgur.com/1TGadON.png)

Image provenance check

![Screenshot 2023-02-03 at 01.19.39](https://i.imgur.com/kpHvbB4.png)


Image signature check


![Screenshot 2023-02-03 at 01.20.28](https://i.imgur.com/ynquh84.png)

Image freshness check

![Screenshot 2023-02-03 at 01.21.08](https://i.imgur.com/9Zzdr6t.png)



![Screenshot 2023-02-03 at 01.21.57](https://i.imgur.com/nHsDzK0.png)









---

### Openssf ScoreCard


- for
  - checking dependency
  - CI/CD
- Go language

- no token
- no limitation


gitlink: naveensrinicasan/score

#### Feature


```bash
https://api.securityscorecards.dev
curl https://api.securityscorecards.dev/projects/github.com/kube/kube
```

#### status


- API serves: 100k to 150k
- customer unknown

use case:
- score
- Binary artific


##### use

- copy badge in readme
- Action update: scorecard#score

##### community

Https://securityscorecards.dev/

community meetings











---


### SLSA

https://github.com/slsa-framework/slsa-verifier


GUAC

For artifacts relationship
















.
