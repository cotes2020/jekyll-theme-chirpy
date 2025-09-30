---
title: "Code Block - Step"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-04-19. 01:27 # Init
# last_modified_at: 2025-04-19. 01:27 #
---

## Step

---

```cs
public enum SomeStep
{
    None = 0,
    Step1 = 1,
    Step2 = 2,
    Step3 = 3,
    Step4 = 4,
    Step5 = 5,
}

public class SomeClass: MonoBehaviour
{
    private SomeStep step = SomeStep.None;

    public void SetStep(SomeStep newStep)
    {
        step = newStep;
    }

    private IEnumerator WaitForStep(SomeStep targetStep)
    {
        do yield return null;
        while (step != targetStep);
    }

    public void SomeMethod()
    {
        while (true)
        {
            // Do something
            Debug.Log("Doing something...");

            // Wait for step 3
            yield return StartCoroutine(WaitForStep(SomeStep.Step3));

            // Do something else
            Debug.Log("Doing something else...");

            // Wait for step 5
            yield return StartCoroutine(WaitForStep(SomeStep.Step5));
        }
    }
}
```
