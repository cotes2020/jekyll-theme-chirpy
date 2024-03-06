

json file for intents.


```json
{
  "interactionModel": {
    "languageModel": {
      "invocationName": "space facts",

      "intents": [
        {
          "name": "AMAZON.CancelIntent",
          "samples": []
        },

        {
          "name": "AMAZON.HelpIntent",
          "samples": []
        },

        {
          "name": "AMAZON.StopIntent",
          "samples": []
        },

        {
          "name": "AMAZON.FallbackIntent",
          "samples": []
        },

        {
          "name": "GetNewFactIntent",
          "samples": [
            "a fact",
            "a space fact",
            "tell me a fact",
            "tell me a space fact",
            "give me a fact",
            "give me a space fact",
            "tell me trivia",
            "tell me a space trivia",
            "give me trivia",
            "give me a space trivia",
            "give me some information",
            "give me some space information",
            "tell me something",
            "give me something"
          ],
          "slots": []
        },
        {
          "name": "AMAZON.NavigateHomeIntent",
          "samples": []
        }
      ]
    }
  }
}

```
