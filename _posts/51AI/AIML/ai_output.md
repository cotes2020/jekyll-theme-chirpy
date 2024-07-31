Generally speaking: Due to the nature of LLMs, you can never guarantee a JSON response. You will have to adopt your strategy to cope with this fact. Out of the top of my head these are your options:

Prompt engineering --> gets you to 95%

With careful prompting and specific instructions you can maximize the likelihood of getting a JSON response. There are a lot of resources on prompt engineering out there, but since it is model dependent and subject to change, you will always have to experiment on what works best for your case.

Post-Processing --> gets you to 99%

Make your application code more resilient towards non JSON-only for example you could implement a regular expression to extract potential JSON strings from a response. As an example a very naive approach that simply extracts everything between the first { and the last }

```js
const naiveJSONFromText = (text) => {
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) return null;

    try {
        return JSON.parse(match[0]);
    } catch {
        return null;
    }
};
```

Validation Loop --> gets you to 100%

In the end you will always have to implement validation logic to check A: That you deal with a valid JSON object and B: That it has your expected format.

const isValidSomeObject = (obj) =>
    typeof obj?.field1 === 'number' && typeof obj?.field2 === 'number';
Depending on your use case I would recommend to automatically query the LLM again if this validation fails.

Closing thought: Even though the prompt engineering gets you the furthest. I would recommend to implement the other parts first to be able to get going and then try to reduce the amounts of validation fails by improving your prompt.
