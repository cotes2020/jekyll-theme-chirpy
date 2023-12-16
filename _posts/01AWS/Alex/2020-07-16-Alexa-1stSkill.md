---
title: AWS Alex First Skill - RedVelvet Time
date: 2020-07-16 11:11:11 -0400
categories: [01AWS, Alexa]
tags: [AWS, Lab, Alexa]
math: true
image:
---

# Alex First Skill - RedVelvet Time

- [Alex First Skill - RedVelvet Time](#alex-first-skill---redvelvet-time)
  - [Create a Skill in Five Minutes](#create-a-skill-in-five-minutes)
  - [Introduction: How Users Will Interact With Cake Time](#introduction-how-users-will-interact-with-cake-time)
  - [Build the Cake Time Skill](#build-the-cake-time-skill)
    - [Step 1. Log in](#step-1-log-in)
    - [Step 2. Create your skill](#step-2-create-your-skill)
    - [Step 3. Greet the user](#step-3-greet-the-user)
    - [Step 4: Build](#step-4-build)
    - [Step 5: Test your skill](#step-5-test-your-skill)
    - [Code](#code)
  - [Collecting slots turn-by-turn with auto-delegation](#collecting-slots-turn-by-turn-with-auto-delegation)
    - [Step 1: Ask the user for their birthday](#step-1-ask-the-user-for-their-birthday)
    - [Step 2: Use an intent and slots to capture information](#step-2-use-an-intent-and-slots-to-capture-information)
    - [Step 3: Use dialog management](#step-3-use-dialog-management)
    - [Step 4: Define a new handler](#step-4-define-a-new-handler)
    - [Step 5: Test your skill](#step-5-test-your-skill-1)
    - [Wrap-up](#wrap-up)
  - [Adding memory to skill](#adding-memory-to-skill)
    - [Step 1: Use Amazon S3 to save and read data](#step-1-use-amazon-s3-to-save-and-read-data)
    - [Step 2: Save Data](#step-2-save-data)
    - [Step 3: Read stored data](#step-3-read-stored-data)
    - [How to delete or reset the user’s birthday](#how-to-delete-or-reset-the-users-birthday)
    - [test, so click the Test tab, then follow the steps below.](#test-so-click-the-test-tab-then-follow-the-steps-below)
  - [Using the Alexa Settings API](#using-the-alexa-settings-api)
    - [Step 1: Get Device ID, API endpoint, and Authorization Token for Alexa Settings API](#step-1-get-device-id-api-endpoint-and-authorization-token-for-alexa-settings-api)
    - [Step 2: Using the Alexa Settings API to retrieve the user time zone](#step-2-using-the-alexa-settings-api-to-retrieve-the-user-time-zone)


---

## Create a Skill in Five Minutes

create a skill: “Cake Time"

What you’ll learn:
- How to build a simple skill called Cake Time with step-by-step instructions
- How to use the Alexa Developer Console
- How to host your skill’s backend resources
- How to modify the response that Alexa speaks to customers
- How to test your skill

## Introduction: How Users Will Interact With Cake Time
build Cake Time, a simple skill that:
- asks the user for their birthday
- remembers it
- tells them how many days until their next birthday
- wishes them Happy Birthday on their birthday

At the end of module 3, your first Alexa skill will say “Hello! Welcome to Cake Time. That was a piece of cake! Bye!”
- The skill is simple to use yet a bit complex to build. The burden is on us, the skill builder, to make the interaction simple and natural.
- One way to make it as natural as possible is to mimic human conversational patterns. Humans have memory so your skill should too. It would be frustrating if your best friend always had to ask your name (which may be a sign that they really aren't your best friend at all). While you could build cake time in a day, because of its complexity you'll build cake time over four modules in this course.


## Build the Cake Time Skill
### Step 1. Log in

### Step 2. Create your skill
1. Click Create Skill on the right-hand side of the console. A new page displays.
1. In the Skill name field, enter Cake Time.
1. Leave the Default language set to English (US).
1. You are building a `custom skill`. Under Choose a model to add to your skill, select Custom.
  1. Skills have a **front end** and **backend**.
  1. The front end is where you map utterances (what the user says) into an intent (the desired action).
  1. You must decide how to handle the user's intent in the backend.
  1. Host the skill yourself using an `AWS Lambda function or HTTPS endpoint`, or choose Alexa to host the skill for you.
  1. There are limits to the AWS Free Tier, so if your skill goes viral, you may want to move to the self-hosted option.
  1. To follow the steps in this course, choose Alexa-Hosted (Node.js).
1. Choose a method to host your skill's backend resource: `Alexa-Hosted (Python)`.
1. Create skill.
1. exit and return to the Alexa developer console, find your skill on the Skills tab, in the Alexa Skills list. Click Edit to continue working on your skill.

### Step 3. Greet the user

1. The first thing a user will want to do with the skill is open it.
2. The intent of opening the skill is built into the experience, so you don't need to define this intent in your front end.
3. However, you need to respond to the intent in your backend.
  - update your backend code to greet the user when they open the skill.
4. Open the Cake Time skill. Click the **Code tab**.

5. There are two pieces to a handler:
    - `can_handle() function`: define what requests the handler responds to.
      - If your skill receives a request, the can_handle() function within each handler determines whether or not that handler can service the request.
      - In this case
      - the user wants to launch the skill, which is a `LaunchRequest`.
      - Therefore, the ca n_handle() function within the `LaunchRequestHandler` will let the SDK know it can fulfill the request.
      - In computer terms, the can_handle returns true to confirm it can do the work.

    - `handle() function`: returns a response to the user.
      - line that begins `speak_output =`.
      - This variable contains the string of words the skill should say back to the user when they launch the skill.

    - What should happen when a user launches the Cake Time skill?
      - want the skill to simply confirm that the user opened it by saying, "Hello! Welcome to Cake Time. That was a piece of cake! Bye!" Within the `LaunchRequestHandler` object, find the `handle()` function.
      - This function uses the `responseBuilder` function to compose and return the response to the user.

6. Within the `LaunchRequestHandler` object
   - the handle() function
   - Replace: `speak_output = "Hello! Welcome to Cake Time. That was a piece of cake! Bye!"`
   - `.ask() function`: omit this line of code for now.

```py
class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""

    def can_handle(self, handler_input):    # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("LaunchRequest")(handler_input)


    def handle(self, handler_input):        # type: (HandlerInput) -> Response

        speak_output = "Hello! Welcome to Cake Time. That was a piece of cake! Bye!"
        # words the skill should say back to the user when launch the skill

        return (
            handler_input.response_builder  # help build the response to the user
                .speak(speak_output)   # .speak(): tells responseBuilder to speak the value of speak_output to the user.
                # .ask(speak_output)   # .ask(): If the skill was supposed to listen for the user’s response
                .response              # converts the responseBuilder’s work into the response that the skill will return.
        )
```

7. **save**

8. **Deploy**


### Step 4: Build

1. Build > Invacation > change `Skill Invocation Name` > save > `Build Model`


### Step 5: Test your skill
1. Click the **Test tab**.
2. An alert may appear requesting to use your computer's microphone. Click Allow.
3. From the drop-down menu at the top left of the page, select `Development`.
   - 2 ways to test your skill in the console.
     - type what the user would say into the box at the top left.
     - speak to the skill by clicking and holding the microphone icon and speaking.

3. So far, the skill has one intent: LaunchRequest.
   - This function responds to the user when they ask Alexa to open or launch the skill.
   - The user will say, "Alexa, open Cake Time." Cake Time is the name of your skill and was automatically set as the invocation name for the skill. You can change the invocation name, but let's leave it as is for this exercise.

4. Test the skill. Type open Cake Time (not case sensitive) into the box at the top left and press ENTER, or click and hold the microphone icon and say, "Open Cake Time."

5. When testing on an Alexa-enabled device, you need the wake word: "Alexa, open Cake Time"


### Code

```py
# -*- coding: utf-8 -*-

# This sample demonstrates handling intents from an Alexa skill using the Alexa Skills Kit SDK for Python.
# Please visit https://alexa.design/cookbook for additional examples on implementing slots, dialog management,
# session persistence, api calls, and more.
# This sample is built using the handler classes approach in skill builder.
import logging
import ask_sdk_core.utils as ask_utils

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Hello! Welcome to Cake Time. That was a piece of cake! Bye!"

        return (
            handler_input.response_builder
                .speak(speak_output)
                #.ask(speak_output)
                .response
        )


class HelloWorldIntentHandler(AbstractRequestHandler):
    """Handler for Hello World Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("HelloWorldIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Hello World!"

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )


class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "You can say hello to me! How can I help?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Goodbye!"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )


class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # Any cleanup logic goes here.

        return handler_input.response_builder.response


class IntentReflectorHandler(AbstractRequestHandler):
    """The intent reflector is used for interaction model testing and debugging.
    It will simply repeat the intent the user said. You can create custom handlers
    for your intents by defining them above, then also adding them to the request
    handler chain below.
    """
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("IntentRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        intent_name = ask_utils.get_intent_name(handler_input)
        speak_output = "You just triggered " + intent_name + "."

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors. If you receive an error
    stating the request handler chain is not found, you have not implemented a handler for
    the intent being invoked or included it in the skill builder below.
    """
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)

        speak_output = "Sorry, I had trouble doing what you asked. Please try again."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

# The SkillBuilder object acts as the entry point for your skill, routing all request and response
# payloads to the handlers above. Make sure any new handlers or interceptors you've
# defined are included below. The order matters - they're processed top to bottom.


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(HelloWorldIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(IntentReflectorHandler()) # make sure IntentReflectorHandler is last so it doesn't override your custom intent handlers

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
```

---

## Collecting slots turn-by-turn with auto-delegation

At the end of this module, your Cake Time skill will be able to:
- Ask the user a question
- Listen for the answer
- Respond to the user

### Step 1: Ask the user for their birthday

1. Code tab.

2. Within the LaunchRequestHandler, in the handle() function, change `speak_output =`.

3. `.ask() function` does two things:
   - Tells the skill to wait for the user to reply, rather than simply exiting
   - Allows you to specify a way to ask the question to the user again, if they don’t respond. `reprompt_text`

4. replace `.ask(speak_output)` with `.ask(reprompt_text)`

```py
class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""

    def can_handle(self, handler_input):        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)



    def handle(self, handler_input):        # type: (HandlerInput) -> Response

        speak_output = "Aniyo! Welcome to red velevt Time. What is your birthday?"
        reprompt_text = "I was born Aug. 1st, 2014. When are you born?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(reprompt_text)
                .response
        )
```


### Step 2: Use an intent and slots to capture information

1. Now make adjustments to the skill's front end.
2. need to create an `intent` that will `interpret how the user responds to Alexa's question`.
   - When you name an intent, think about what the intent is going to do.
   - In this case, the intent is going to capture the user's birthday, so name it `CaptureBirthdayIntent`.
   - Notice the words are not separated by spaces, and each new word begins with an uppercase letter.
   - **intent**: an action to fulfill a user's request.
   - **utterance**: is what invokes the intent.
   - In response to the birthday question, a user might say - "I was born on November seventh, nineteen eighty three."
   - You will add this utterance to the CaptureBirthdayIntent by typing it in exactly the way the user is expected to say it.

3. In the **Sample Utterances** field
   - `I was born on November seventh nineteen eighty three`
   - press ENTER or click the + icon
   - When finished, the Cake Time skill will be able to capture any birthday.

4. In the dialog box, click the field under Create a new slot, type the name of the slot without curly brackets (for example, month), and click Add.
   - From this utterance
     - there are three key pieces of information to collect: month, day, and year. `slots`
     - need to let Alexa know which words are slots and what kind of slots they are.
       - Start with the month slot.
       - replace the word representing the month (November) with the word month in curly brackets `{ }`.
       - This creates a slot called month.
       - The utterance will then look like this: I was born on {month} seventh nineteen eighty three
     - 2 ways to create a slot.
       - 1.
         - select the word in the sample utterance where the slot should go,
         - type the name of the slot in curly brackets (for example, {month}).
       - 2.
         - select the word in the sample utterance and use the Select an Existing Slot dialog box when it appears.
   - `I was born on {month} {day} {year}`
   - `{month} {day} {year}` if the user omits the words I was born on

![Screen Shot 2020-07-16 at 14.28.15](https://i.imgur.com/sQzbfah.png)

5. Now define exactly what those slots are by assigning a `slot type` to each `slot`: `Intent Slots`
   - Slots are assigned from the `Slot Type`
   - 2 types of slot types: `custom` and `built-in`.
   - Wherever possible, use built-in slots. Alexa manages the definitions of built-in slots.
   - If an applicable built-in slot does not exist, create a custom slot and define the values it represents.

![Screen Shot 2020-07-16 at 14.31.49](https://i.imgur.com/QRofGSq.png)

6. Save Model.


### Step 3: Use dialog management

Slots can be `required` or `optional`.
- if you need a given value from the user, you can designate a slot as `required` using dialog management.
  - Marking a slot as required triggers Alexa to actively work to fill it. Start by making each of the slots required.

1. Click on “CaptureBirthdayIntent” on the left nav bar.
2. In the Intent Slots section, to the right of the month slot, click `Edit Dialog`.
3. Under Slot Filling, toggle to make the slot required: mark `Is this slot required to fulfill the intent?`
4. The **Alexa speech prompts** field:
   - enter text for Alexa to say if the user fails to provide a value for the month slot.
   - `What month were you born in?`
5. Repeat the process for the `day` and `year` slots.
   - if a user responds, "July nineteen eighty two,"
     - Alexa recognizes that the month and year slots are filled, but the day slot is not.
     - Alexa will prompt the user for each unfilled slot. In this example, Alexa would ask, "What day were you born?"

   - One of the great things about `dialog management`
     - the skill doesn't break or get confused if the user leaves out a piece of information or provides it out of the expected order.
     - Instead, Alexa takes on the responsibility of collecting information designated as required to ensure a useful experience.
     - You have built an `intent` that listens for the user's answer to the birthday question.
     - When the user responds, Alexa collects the user's birthday month, day, and year.
     - This information will be sent to the skill's **backend code** in a `JSON request`.

6. Delete the `HelloWorldIntent` intent

7. other intents (such as AMAZON.HelpIntent) were automatically added to your skill.
   - These are required for every skill and provide the user a means to cancel, stop, and get help.
   - Do not remove these.

8. Save Model.

9. Build Model.


### Step 4: Define a new handler

1. To make the Cake Time skill respond, you need to update the backend.
2. Code tab.
3. change `HelloWorldIntent` to `CaptureBirthdayIntent`
4. `canHandle() function` will be invoked when a `CaptureBirthdayIntent` request comes through.
5. creating three variables in the handler to save the slots the skill is collecting.
6. update the speak_output: `'Thanks, I will remember that you were born {month} {day} {year}.'.format(month=month, day=day, year=year)`

```py
class CaptureBirthdayIntentHandler(AbstractRequestHandler):
    """Handler for Hello World Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("CaptureBirthdayIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        slots = handler_input.request_envelope.request.intent.slots
        year = slots["year"].value
        month = slots["month"].value
        day = slots["day"].value

        speak_output = 'Thanks, I will remember that you were born {month} {day} {year}.'.format(month=month, day=day, year=year)

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )
```

7. Scroll down in the code until you find the line that begins `sb = SkillBuilder()`.
   - Replace the HelloWorld with CaptureBirthday

```py
import logging
import ask_sdk_core.utils as ask_utils

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ......

sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(CaptureBirthdayIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(IntentReflectorHandler()) # make sure IntentReflectorHandler is last so it doesn't override your custom intent handlers

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
```

8. Click Save.
9. Click Deploy. Because of the new handler, your skill will take a few moments to deploy.


### Step 5: Test your skill

The Redvelvet Time Time skill should now be able to do the following:
- Ask the user for their birthday
- Listen to the answer from the user and automatically follow up with questions if any required slots (month, day, year) are missing
- Respond to the user by repeating their birthday Let's test the skill.


1. Click the Test tab.
2. Test your skill by opening Cake Time and responding when Alexa asks for your birthday.


### Wrap-up

![Screen Shot 2020-07-16 at 16.40.25](https://i.imgur.com/vsSyTwV.png)


---


## Adding memory to skill

enable the Cake Time skill to remember the user’s birthday.

### Step 1: Use Amazon S3 to save and read data

- you have the birthday, month, and year within the code.
   - but the skill forgets these values when the code finishes running.
   - save the values to Amazon S3
   - so the skill can read them from session to session.

- The SDK provides mechanism for saving information across sessions: the `AttributesManager`
   - Withhe manager, your read/write code can remain the same, even if you change where you save your data later.

   - The `backend code` for Alexa skills can live on any `HTTPS server`.
   - Most of the time, Alexa developers write and host their backend code using AWS.
     - writing code in the developer console using an **Alexa-hosted skill**.
       - That code is running on the AWS Free Tier, which has limitations.
       - great for learning and publish simple skills.
       - When using an Alexa-hosted skill for backend code, it will be stored in `Amazon S3`.
     - if skill becomes popular, consider your own AWS **resources**.
       - If build code on your own AWS resources, it may use `Amazon DynamoDB`.
       - it will only require minor changes to work with DynamoDB if you decide to migrate to your own AWS resources later

1. use `AttributesManager` to save the user’s birthday in Cake Time.
   - Code tab.
   - requirements.txt file
   - add a requirement. import the dependency for the S3 adapter.
     - `ask-sdk-s3-persistence-adapter`
   - Click Save.

2. lambda_function.py tab
   - The new dependency allows to use the `AttributesManager` to save and read user data using `Amazon S3`.

   - import that dependency to the code.
     - find the line begins `import ask_sdk_core.utils as ask_utils`.
     - Create a new line just below it, and copy and paste in the following code:

```py
import os
from ask_sdk_s3.adapter import S3Adapter
s3_adapter = S3Adapter(bucket_name=os.environ["S3_PERSISTENCE_BUCKET"])
```

   - find the line that begins `from ask_sdk_core.skill_builder import SkillBuilder`.
     - Replace this line with the following code:
     - `from ask_sdk_core.skill_builder import CustomSkillBuilder`
       - This will import the `S3 Persistence adapter`,
       - create your S3 adapter and set you up with a bucket on S3 to store your data.
       - Once done, this section of code should look like:

   - find the line that begins `sb = SkillBuilder()`.
     - Replace this line with the following code:
     - `sb = CustomSkillBuilder(persistence_adapter=s3_adapter)`


3. Click Save.

```py
import logging
import ask_sdk_core.utils as ask_utils


import os
from ask_sdk_s3.adapter import S3Adapter
s3_adapter = S3Adapter(bucket_name=os.environ["S3_PERSISTENCE_BUCKET"])

from ask_sdk_core.skill_builder import CustomSkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#.....

sb = CustomSkillBuilder(persistence_adapter=s3_adapter)
```

You are now set up to use `AttributesManager` to save and read data to Amazon S3.

Later, if you decide to move your skill’s backend code to your own AWS resources, you will reverse the changes made in this step.


### Step 2: Save Data

1. Now modify the code to save the user’s birthday.

2. lambda_function.py file,
   - find the `CaptureBirthdayIntentHandler`. use the `AttributesManager` to save the user’s birthday.
     - create a new line `attributes_manager = handler_input.attributes_manager`
     - The Cake Time skill code receives the year, month, and day.
     - You need to tell Amazon S3 to save these values.
     - The code tells the AttributesManager what the data is, and the manager sends it to Amazon S3.

3. Within the `CaptureBirthdayIntentHandler`
   - mapping `the variables already declared in the code` to corresponding` variables that will be created in Amazon S3` when the code runs.
   - `birthday_attributes = { }`


4. `attributes_manager.persistent_attributes = birthday_attributes`
   - These variables are now declared as persistent (they are local to the function in which they are declared, yet their values are retained in memory between calls to the function).
   - Now you save the user’s data to them.
   - First, use the AttributesManager to set the data to save to Amazon S3.

5. `attributes_manager.save_persistent_attributes()`

6. Click Save.

7. `CaptureBirthdayIntentHandler` should now look like:

```py
class CaptureBirthdayIntentHandler(AbstractRequestHandler):
    """Handler for Hello World Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("CaptureBirthdayIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        slots = handler_input.request_envelope.request.intent.slots
        year = slots["year"].value
        month = slots["month"].value
        day = slots["day"].value

        attributes_manager = handler_input.attributes_manager

        birthday_attributes = {
            "year": year,
            "month": month,
            "day": day
        }

        attributes_manager.persistent_attributes = birthday_attributes
        attributes_manager.save_persistent_attributes()

        speak_output = 'Thanks, I will remember that you were born {month} {day} {year}.’.format(month=month, day=day, year=year)

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )
```


### Step 3: Read stored data

- now the user’s birthday is saved to Amazon S3.
- However, now the `skill needs to be updated`
  - so the next time the user opens Cake Time, the skill knows the user’s birthday information is stored and doesn’t have to ask for it.

- To do this,
- `modify the code to read the data stored in Amazon S3` before asking the user for their birthday.
  - If the data exists, the skill doesn’t need to ask for it.
  - If the data isn’t there, it will ask for the information.

- An `Amazon S3 bucket` is a public cloud storage resource.
  - A bucket is similar to a file folder for storing objects, consists of data and descriptive metadata.

- A `new handler` is needed to read the stored data.
  - The canHandle() and handle() functions in the new handler will communicate with Amazon S3.
  - You will add it between the `LaunchRequestHandler` and the `CaptureBirthdayIntentHandler`.


```py
class HasBirthdayLaunchRequestHandler(AbstractRequestHandler):
    """Handler for launch after they have set their birthday"""

    def can_handle(self, handler_input):
        # extract persistent attributes and check if they are all present
        attr = handler_input.attributes_manager.persistent_attributes
        attributes_are_present = ("year" in attr and "month" in attr and "day" in attr)

        return attributes_are_present and ask_utils.is_request_type("LaunchRequest")(handler_input)


    def handle(self, handler_input):
        attr = handler_input.attributes_manager.persistent_attributes
        year = attr['year']
        month = attr['month'] # month is a string, and we need to convert it to a month index later
        day = attr['day']

        # TODO:: Use the settings API to get current date and then compute how many days until user’s bday
        # TODO:: Say happy birthday on the user’s birthday

        speak_output = "Welcome back it looks like there are X more days until your y-th birthday."

        handler_input.response_builder.speak(speak_output)

        return handler_input.response_builder.response
```
1. Find the line that begins CaptureBirthdayIntentHandler. Create new code.
   - The new handler has the canHandle() and handle() functions.
   - The canHandle() function checks if the user's birthday information is saved in Amazon S3.
     - If it is, the handler lets the SDK know it can do the work (it has the user's birthday information and can do what comes next).
       - The handle() function tells Alexa to say, "Welcome back. It looks like there are x more days until your y-th birthday."

2. When changed the name of a handler in a previous section, also had to change the name in the list of handlers at the bottom of the code.


3. added a new handler, you must add the new handler to this list.
   - Toward the bottom of the code, find the line that begins with sb.add_request_handler(LaunchRequestHandler()), Create a new line just above it.
   - `sb.add_request_handler(HasBirthdayLaunchRequestHandler())`

4. Click Save.

5. Click Deploy.

That section of code:

```py
sb.add_request_handler(HasBirthdayLaunchRequestHandler())
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(CaptureBirthdayIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(IntentReflectorHandler()) # make sure IntentReflectorHandler is last so it doesn’t override your custom intent handlers

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
```


### How to delete or reset the user’s birthday

- When testing, you may need to delete or reset the user’s birthday. There are two ways to do this.
  1. **Test tab**, Type or say, "Alexa, tell Cake Time I was born on {month} {day} {year}."
  2. delete the saved information from Amazon S3:
     - **Code tab**
     - click **Media storage** on the bottom left-hand corner of the screen.
     - The S3 Management Console opens.
     - Click the breadcrumb that starts `amzn-1-ask-skill`.
     - Click on the check box next to the file(s) that begins with `amzn1.ask.account`.
     - Delete.
     - The user’s birthday is deleted.

### test, so click the Test tab, then follow the steps below.

1. Launch the skill first time
   - Say “open redvelvet time”
   - Alexa should respond, “Hello! This is Cake Time. When is your birthday?”
   - Tell Alexa your birthday
   - Once the skill has your birth month, day, and year, it should respond, “Thanks, I’ll remember that your birthday is {month} {day} {year}.”
   - The session ends. At this point, without the code you added in this section, the next time you invoke the skill, the skill would ask for your birthday again. Now, the skill stores this information.

2. Launch the skill a second time
   - Say “open redvelvet time”
   - Alexa should respond, “Welcome back. It looks like there are X more days until your y-th birthday.”
   - You probably noticed that, with the way the code works right now, Alexa is saying “X” and “Y T H”. Don’t worry. In the next section, you will work on the code to calculate how many days until the user’s next birthday so Alexa can respond with that information.


## Using the Alexa Settings API

- enable the Cake Time skill to calculate the number of days until the user’s next birthday.
  - To calculate the number of days until the user’s next birthday accurately, we need additional information, like current date, and user’s time zone.
  - can use the Alexa Settings API to get this information.
  - To do that, need to pass the following information to the Alexa Settings API:
    - Device ID
    - URL for the Alexa Settings API (API Endpoint)
    - Authorization token (Access Token)
    - Import supporting libraries (We will do this in Step 3)

### Step 1: Get Device ID, API endpoint, and Authorization Token for Alexa Settings API
### Step 2: Using the Alexa Settings API to retrieve the user time zone

requirements.txt:

```py
boto3==1.9.216
ask-sdk-core==1.11.0
ask-sdk-s3-persistence-adapter
pytz
# The pytz library allows accurate and cross platform timezone calculations, and will help us figure out the user's timezone accurately.
```


```py
# -*- coding: utf-8 -*-

# This sample demonstrates handling intents from an Alexa skill using the Alexa Skills Kit SDK for Python.
# Please visit https://alexa.design/cookbook for additional examples on implementing slots, dialog management,
# session persistence, api calls, and more.
# This sample is built using the handler classes approach in skill builder.
import logging
import ask_sdk_core.utils as ask_utils
import os
import requests
import calendar
from datetime import datetime
from pytz import timezone
from ask_sdk_s3.adapter import S3Adapter
s3_adapter = S3Adapter(bucket_name=os.environ["S3_PERSISTENCE_BUCKET"])

from ask_sdk_core.skill_builder import CustomSkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Hello! This is Cake Time. What is your birthday?"
        reprompt_text = "I was born Nov. 6th, 2015. When are you born?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(reprompt_text)
                .response
        )

class HasBirthdayLaunchRequestHandler(AbstractRequestHandler):
    """Handler for launch after they have set their birthday"""

    def can_handle(self, handler_input):
        # extract persistent attributes and check if they are all present
        attr = handler_input.attributes_manager.persistent_attributes
        attributes_are_present = ("year" in attr and "month" in attr and "day" in attr)

        return attributes_are_present and ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        attr = handler_input.attributes_manager.persistent_attributes
        year = int(attr['year'])
        month = attr['month'] # month is a string, and we need to convert it to a month index later
        day = int(attr['day'])

        # get device id
        sys_object = handler_input.request_envelope.context.system
        device_id = sys_object.device.device_id

        # get Alexa Settings API information
        api_endpoint = sys_object.api_endpoint
        api_access_token = sys_object.api_access_token

        # construct systems api timezone url
        url = '{api_endpoint}/v2/devices/{device_id}/settings/System.timeZone'.format(api_endpoint=api_endpoint, device_id=device_id)
        headers = {'Authorization': 'Bearer ' + api_access_token}

        userTimeZone = ""
        try:
	        r = requests.get(url, headers=headers)
	        res = r.json()
	        logger.info("Device API result: {}".format(str(res)))
	        userTimeZone = res
        except Exception:
	        handler_input.response_builder.speak("There was a problem connecting to the service")
	        return handler_input.response_builder.response

        # getting the current date with the time
        now_time = datetime.now(timezone(userTimeZone))

        # Removing the time from the date because it affects our difference calculation
        now_date = datetime(now_time.year, now_time.month, now_time.day)
        current_year = now_time.year

        # getting the next birthday
        month_as_index = list(calendar.month_abbr).index(month[:3].title())
        next_birthday = datetime(current_year, month_as_index, day)

        # check if we need to adjust bday by one year
        if now_date > next_birthday:
            next_birthday = datetime(
                current_year + 1,
                month_as_index,
                day
            )
            current_year += 1
        # setting the default speak_output to Happy xth Birthday!!
        # alexa will automatically correct the ordinal for you.
        # no need to worry about when to use st, th, rd
        speak_output = "Happy {}th birthday!".format(str(current_year - year))
        if now_date != next_birthday:
            diff_days = abs((now_date - next_birthday).days)
            speak_output = "Welcome back. It looks like there are \
                            {days} days until your {birthday_num}th\
                            birthday".format(
                                days=diff_days,
                                birthday_num=(current_year-year)
                            )

        handler_input.response_builder.speak(speak_output)

        return handler_input.response_builder.response

class CaptureBirthdayIntentHandler(AbstractRequestHandler):
    """Handler for Hello World Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("CaptureBirthdayIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        slots = handler_input.request_envelope.request.intent.slots
        year = slots["year"].value
        month = slots["month"].value
        day = slots["day"].value

        attributes_manager = handler_input.attributes_manager

        birthday_attributes = {
            "year": year,
            "month": month,
            "day": day
        }

        attributes_manager.persistent_attributes = birthday_attributes
        attributes_manager.save_persistent_attributes()

        speak_output = 'Thanks, I will remember that you were born {month} {day} {year}.'.format(month=month, day=day, year=year)

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )


class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "You can say hello to me! How can I help?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Goodbye!"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )


class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # Any cleanup logic goes here.

        return handler_input.response_builder.response


class IntentReflectorHandler(AbstractRequestHandler):
    """The intent reflector is used for interaction model testing and debugging.
    It will simply repeat the intent the user said. You can create custom handlers
    for your intents by defining them above, then also adding them to the request
    handler chain below.
    """
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("IntentRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        intent_name = ask_utils.get_intent_name(handler_input)
        speak_output = "You just triggered " + intent_name + "."

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors. If you receive an error
    stating the request handler chain is not found, you have not implemented a handler for
    the intent being invoked or included it in the skill builder below.
    """
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)

        speak_output = "Sorry, I had trouble doing what you asked. Please try again."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

# The SkillBuilder object acts as the entry point for your skill, routing all request and response
# payloads to the handlers above. Make sure any new handlers or interceptors you've
# defined are included below. The order matters - they're processed top to bottom.


sb = CustomSkillBuilder(persistence_adapter=s3_adapter)

sb.add_request_handler(HasBirthdayLaunchRequestHandler())
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(CaptureBirthdayIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(IntentReflectorHandler()) # make sure IntentReflectorHandler is last so it doesn't override your custom intent handlers

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
```












.
