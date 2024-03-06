---
title: AWS Alex
date: 2020-07-16 11:11:11 -0400
categories: [01AWS, Alexa]
tags: [AWS, Lab, Alexa]
math: true
image:
---


# Alexa Basic

- [Alexa Basic](#alexa-basic)
  - [Module 1: Why Build Alexa Skills](#module-1-why-build-alexa-skills)
  - [Skill](#skill)
  - [resources](#resources)
  - [Pre-Built Model](#pre-built-model)
  - [Alexa workflow](#alexa-workflow)
  - [Steps to Build a Skill?](#steps-to-build-a-skill)
  - [Requirements to build a skill for this tutorial](#requirements-to-build-a-skill-for-this-tutorial)
- [Module 2: Design an Engaging Voice User Interface](#module-2-design-an-engaging-voice-user-interface)
  - [How Users Interact With Alexa](#how-users-interact-with-alexa)
  - [Voice Design Concepts: Utterances, Intents, and Slots](#voice-design-concepts-utterances-intents-and-slots)
  - [Key Concepts: Interaction Model and Situational Design](#key-concepts-interaction-model-and-situational-design)
    - [Interaction model](#interaction-model)
    - [Voice design](#voice-design)
    - [Situational Design](#situational-design)
  - [Characteristics of a Well-Designed Voice User Interface](#characteristics-of-a-well-designed-voice-user-interface)
  - [Key Challenges of Voice Design](#key-challenges-of-voice-design)
  - [Five Best Practices for Voice Design](#five-best-practices-for-voice-design)


---

## Module 1: Why Build Alexa Skills

**Ease of access**
- VUIs are natural, conversational, and user-centric.
- A great voice experience allows for the many ways people express meaning and intent. It is rich and flexible. Because of this, building for voice isn’t the same as building graphical user interfaces (GUIs) for the web or mobile.
- The easier a skill is to use, the more speed and efficiency it offers.

**Speed and efficiency**
- bring speed and efficiency to mundane or habitual tasks—which is why voice is poised to become ubiquitous.
- Consider the kitchen timer. With Alexa, setting a timer is as easy as saying, “Alexa, set timer for 10 minutes.” Who would have guessed pushing a few buttons on the microwave would become the less convenient option?

**Skill monetization**

---

## Skill

![{2B8EB6C6-B204-4BFB-8B9F-CF8629D642A1}.png](https://i.imgur.com/75pd1Qf.jpg)

The `Alexa Skills Kit` offers pre-built interaction models which include predefined requests and utterances to help you start building quickly.

As the skill builder, you:
- Define the requests the skill can handle
- Define the name Alexa uses to identify your skill, called the invocation name, which you will learn more about in the next module
- Write the code to fulfill the request

![{C3A31028-3BC7-41EB-A705-55851A23BF56}.png](https://i.imgur.com/4fbgG1e.jpg)


---

## resources

[alex/skill-sample-nodejs-fact](https://github.com/alexa/skill-sample-nodejs-fact/tree/master/models)

---

## Pre-Built Model

**Smart Home Skills**
- This type of skill controls smart home devices such as cameras, lights, locks, thermostats, and smart TVs.
- The Smart Home Skill API gives you less control over a user's experience but simplifies development because you don't need to create the VUI yourself.

**Flash Briefing Skills**
- Use the Flash Briefing Skill API to provide your customers with news headlines and other short content.
- As the skill developer, you define the content feeds for the requested flash briefing. These feeds can contain audio content played to the user or text content read to the user.

**Video Skills**
- Use the Video Skill API to provide video content such as TV shows and movies for users.
- As the skill developer, you define the requests the skill can handle, such as searching for and playing video content, and how video content search results display on Alexa-enabled devices.

**Music Skills**
- Use the Music Skill API to provide audio content such as songs, playlists, or radio stations for users.
- This API handles the words a user can say to request and control audio content. These spoken words turn into requests that are sent to your skill. Your skill handles these requests and responds appropriately, sending back audio content for the user on an Alexa-enabled device

---

## Alexa workflow

The following is a simple workflow that demonstrates how Alexa works.

- In this example, the user invokes a simple Alexa skill called Hello World.

1. To launch the skill, the user says, "Alexa, open Hello World."
2. Alexa hears the wake word and listens.
3. The Alexa-enabled device sends the `utterance` to the Alexa service in cloud. There, the utterance is processed via automatic speech recognition, for conversion to text, and natural language understanding to recognize the intent of the text.
4. Alexa sends a `JavaScript Object Notation (JSON) request` to handle the intent to an `AWS Lambda function` in the cloud.
5. The Lambda function acts as the backend and executes code to handle the intent. In this case, the Lambda function returns, "Welcome to the Hello World skill."
   - The lambda function inspects the JSON request.
   - The lambda function determines how to respond.
   - The lambda function sends a `JSON response` to the Alexa service.
6. The Alexa service receives the JSON response and converts the output text to an audio file.
7. The Alexa-enabled device receives and plays the audio.
8. user interacts with an Alexa skill. It assumes you are using AWS Lambda, serverless cloud computing, to host your skill code.

![chapter1-2-how-diagram](https://i.imgur.com/PVi6FVY.png)

---

## Steps to Build a Skill?

1. Design the Voice User Interface `VUI`
   - designing the `voice interaction model` of skill.
   - Once you start designing, you will quickly understand that designing for voice is different than designing mobile or web-based apps.
   - You need to think about all the different ways a user might interact with your voice skill.
   - To provide a fluid and natural voice experience, it is important to script and then act out the different ways a user might talk to Alexa.
   - Also, if you have a multi-modal experience (voice and visual), you need to think of different workflows to navigate through your skill.

2. Build

   - **interaction model**
     - determines the requests a skill can handle.
     - certain words are required to invoke the request.
       - **custom model**
       - **pre-built model**
     - The interaction model is saved in `JSON format`, can be edit with any edit tool.

   - **Invocation name**
     - user: `Alexa, invocation name`
     - ![{0E74844A-5884-421C-86C7-852FAA7827C1}.png](https://i.imgur.com/Qjkng5a.jpg)
     - can change it at anytime
     - but not after skill is certified and published.
     - need to be able with for below.
       - 3 ways to invoke your skill
       - with a specific request: `alexa, ask/do request`
       - with invocation name: `alexa, xx`
       - with generic defined phrase: `alexa, open/run xx`

   - build the `utterances, intents, and slots` in the Alexa developer console.
     - **utterances**:
       - be flex.
       - more better the few.
       - add `can you, plz, will you`
     - **intents**:
       - the requests the skill can handle.
     - **slot**:

   - After your JSON interaction model is ready, build the backend `Lambda function` in the AWS Management Console.
   - Development environment appropriate for the programming language.
     - The ASK SDK and Lambda jointly support `Node.js, Python, and Java`.

   - **endpoint**
     - `Internet-accessible endpoint` for hosting your `backend cloud-based service`
     - provision **your own Lambda endpoint or use Alexa-hosted skills**, which provisions one for you without the need to create an AWS account.
       - can build and host most skills for free with AWS Lambda (first one million calls/mon)
       - Once the backend Lambda function is ready, integrate the Lambda function to your skill and test it in the Alexa developer console.
       - AWS Lambda ARN:
     - **build and host an HTTPS web service**
       - will need a cloud hosting provider and a Secure Sockets Layer (SSL) certificate.

3. Test
   - The Alexa developer console has a built-in Alexa simulator, which is similar to testing on an actual Alexa-enabled device.
   - testing your skill with the Alexa simulator, gathering user feedback to resolve issues and make improvements before submitting your skill for certification.


4. Certification and launch
   - After beta testing your skill, submit it for certification. Once your skill passes certification, it will be published in the Alexa Skills Store for anyone to discover and use. Start promoting it to reach more customers.
   Summary
   - These are the fundamental steps for building Alexa skills.
   - You will dive deeper into each step in subsequent modules of this tutorial.



- for a display device:

![Screen Shot 2020-07-17 at 18.53.22](https://i.imgur.com/vZZl0RW.png)

![Screen Shot 2020-07-17 at 18.54.14](https://i.imgur.com/gpiw3o6.png)

![Screen Shot 2020-07-17 at 18.55.35](https://i.imgur.com/yC2X7kF.png)

![Screen Shot 2020-07-17 at 18.55.59](https://i.imgur.com/Ben23Kk.png)



## Requirements to build a skill for this tutorial

Get ready to build by taking the following actions:
- `Sign up account on the Alexa developer console`. where build and optimize your skill.
- An `internet-accessible endpoint for hosting your backend cloud-based service`.
  - Your backend skill code is usually a Lambda function.
  - For this course you will create a skill with `Alexa-hosted skills`, where the developer console will provision a Lambda endpoint for you along with allowing you to use the Alexa Skills Kit (ASK) SDK directly on the console.
  - Keep in mind that if you plan to use the ASK SDK, the languages supported are Node.js, Python, and Java.
  - Alexa-hosted skills are only available in Node.js and Python.
  - Development environment appropriate for the programming language used. Lambda natively supports Java, Go, PowerShell, Node.js, C#, Python, and Ruby and provides a runtime API, which allows you to use any additional programming languages to author your functions.
  - Publicly accessible website to host any images, audio files, or video files used in your skill.
    - If you host your skill backend with the Alexa-hosted hosting option, an Amazon Simple Storage Service (Amazon S3) will be provisioned for you.
    - If you use another hosting option, such as AWS Lambda, you may use Amazon S3 to host files used in your skill.
    - If you do not have files other than a skill icon, you do not need to host any resources.
  - (Optional) Alexa-enabled device for testing.
    - Skills work with all Alexa-enabled devices, such as the Amazon Echo, Echo Dot, Fire TV Cube, and devices that use the Alexa Voice Service (AVS).
    - If you don't have a device, you can use the Alexa simulator in the developer console. Through the simulator, you can see the display templates for Echo Show and Echo Spot, although the display is not interactive. If your skill includes display and touch interactions, you need an Alexa-enabled device with a screen to test the skill.

---

# Module 2: Design an Engaging Voice User Interface

## How Users Interact With Alexa

To create a voice user interface
- user wakes an Alexa-enabled device with the `wake word` (“Alexa”) and `asks a question` or `makes a request`.
- For Alexa-enabled devices with a screen, a user can also `touch the screen to interact` with Alexa.




## Voice Design Concepts: Utterances, Intents, and Slots

![chapter2-utterance-intent](https://i.imgur.com/GZ6iS4J.png)

- **wake word**: The wake word tells Alexa to start listening to your commands.
- **Launch word**: A launch word is a `transitional action word` that signals Alexa that a `skill invocation` will likely follow.
  - Sample launch words include `tell, ask, open, launch, and use`.
- **Invocation name**: To begin interacting with a skill, a user says the skill's invocation name.
  - For example, to use the Daily Horoscope skill, the user could say, "Alexa, read my daily horoscope."
- **Utterance**: Simply put, an utterance is a user's spoken request. These spoken requests can invoke a skill, provide inputs for a skill, confirm an action for Alexa, and so on. Consider the many ways a user could form their request.
- **Prompt**: A string of text that should be spoken to the customer to ask for information. You include the prompt text in your response to a customer's request.
- **Intent**: An intent represents an `action that fulfills a user's spoken request`.
  - Intents can optionally have arguments called slots.
- **Slot value**: Slots are `input values provided in user's spoken request`. These values help Alexa figure out the user's intent.
  - Slots can be defined with different types.
  - The travel date slot in the above example uses Amazon's built-in `AMAZON.DATE` type to convert words that indicate dates (such as "today" and "next Friday") into a date format, while both from City and to City use the built-in `AMAZON.US_CITY` slot.
  - If you extended this skill to ask the user what activities they plan to do on the trip, you might add a custom `LIST_OF_ACTIVITIES` slot type to reference a list of activities such as hiking, shopping, skiing, and so on.
  - to identify slots for an intent: `create a dialog model for the skill.`

![Screen Shot 2020-07-16 at 08.40.17](https://i.imgur.com/zZkGx6s.png)


## Key Concepts: Interaction Model and Situational Design
### Interaction model
An interaction model: a `combination of utterances, intents, and slots` that identify for skill.

To create an interaction model:
- define the `requests (intents)` and the `words (sample utterances)`.
- Your `Lambda skill code` then determines how your skill handles each intent.
- start defining the intents and utterances on paper and iterate on those to try to cover as many possible ways the user can interact with the skill.
- Then, go to the `Alexa developer console` and start `creating the intents`, utterances, and slots.
  - The console `creates JSON code of your interaction model`.
  - You can also create the interaction model in JSON yourself using any JSON tool and then copy and paste it in the developer console.

### Voice design
- A major part of the experience is designing your skill to mimic human conversation well.
- Before you write one line of code, you should work really hard to think through how your customers will interact with your skill.
- Skipping this step will result in a poorly written skill that will not work well with your users.
- While it may be tempting to use a flow chart to represent how a conversation may branch, don't! Flow charts are not conversational. They are complicated, impossible to read, and tend to lead to an inferior experience not unlike a phone tree. No one likes calling customer support and diving into a phone tree, avoid that.
- Instead of flow charts, you should use situational design.

### Situational Design
Situational Design:na voice-first method to design a voice user interface.

- You start with a simple dialog which helps keep the focus on the conversation.
  - Each interaction between your customer and the skill represents a turn.
  - Each turn has a situation that represents the context.
  - If it's the customer's first time interacting with the skill, there is a set of data that is yet unknown.
  - Once the skill has stored the information, it will be able to use it the next time the user interacts with the skill.

- With situational Design, start with the conversation and work backwards to your solution.
  - Each interaction between the user and Alexa is treated as a turn.
  - In the example below, the situation is that the user's birthday is unknown and the skill will need to ask for it.
  - Practice: The script below shows how the skill “Cake Time” asks the user for their birthday and remembers it. Later, it will be able to tell them the number of days until their next birthday and to wish them Happy Birthday on their birthday.

![chapter2-situational](https://i.imgur.com/GjH4mam.png)

- Each `turn` can be represented as a `card` that contains, `the user utterance, situation and Alexa's response`.
  - Combine these cards together to form a storyboard which shows how the user will progress through the skill over time. Storyboards are conversational, flow charts are not.

![chapter2-situational-turns](https://i.imgur.com/k1SbPIg.png)


## Characteristics of a Well-Designed Voice User Interface

1. Uses **natural forms of communication**
   - user should not be required to learn a new language or remember the rules.
   - A machine should conform to the user's paradigm, not the other way around.

2. **Navigates through information easily**
   - offer easy way to cut through layers of information hierarchy by using voice commands to find important information.

3. **Creates an eyes- and hands-free experience**
   - allow user to perform tasks while their eyes and hands are occupied.

4. **Creates a shared experience**
   - et users collaborate, contribute, or play together through natural conversation.
   - For example, a family could play a game together on an Alexa-enabled device.

## Key Challenges of Voice Design
inherent challenges with voice interfaces, including: `context switching or ambiguity in the conversation`, `discovering intent`, and `being unaware of the user's current state or mood`.
- For a good user experience, you should plan for these challenges when developing your skill.
- the user might provides all the needed information at once, but Alexa is unable to parse information provided all at once.
  - This doesn’t mean that Alexa is unable to comprehend what the user says, but rather that the VUI of the skill is not properly or correctly designed to infer information from the natural way a person may speak.
- it is important to design the VUI to be as similar as possible to a natural conversation that might take place between two human beings.
  - A good VUI dramatically increases the ease of use and user satisfaction for any given skill.

## Five Best Practices for Voice Design
Designing a good VUI voice user interface for a skill involves `writing natural dialog`, `engaging the user throughout the skill`, and `staying true to Alexa's personality`.


1. Stay close to Alexa's persona
   - Alexa's personality is friendly, upbeat, and helpful. She's honest about anything blocking her way but also fun, personable, and able to make small talk without being obtrusive or inappropriate.
   - keep the tone of your skill’s VUI as close to Alexa’s persona as possible.
     - One way to do this is by keeping the VUI natural and conversational.
   - Slightly vary the responses given by Alexa for responses like "thank you" and "sorry". Engaging the user with questions is also a good technique for a well-designed VUI.
   - **Alexa should be helpful by providing the correct answer**.
       - **Do**
         - Alexa: `That's not quite right. One more try`. What year was the Bill of Rights signed?
         - User: 1986
         - Alexa: `Shoot`. That wasn't it. The correct answer was 1791.
       - **Don't**
         - Alexa: `That's not quite right. One more try`. What year was the Bill of Rights signed?
         - User: 1986
         - Alexa: `That's not correct`. Let's move on.
   - **Engage the user with questions and avoid ending questions with "yes or no?"**
      - **Do**
         - Alexa: Do you want to keep shopping?
      - **Don't**
         - Alexa: Do you want to keep shopping? `Yes or no?`


2. Write for the ear, not the eye
   - The way we speak is far less formal than the way we write. Therefore, it's important to `write Alexa’s prompts to the user in a conversational tone`.
   - No matter how good a prompt sounds when you say it, it may sound odd in text-to-speech (TTS).
   - It is important to listen to the prompts on your test device and then iterate on the prompts based on how they sound.
   - **Keep your VUI informal. The following is an example**.
      - Do
         - Alexa: Getting your playlist.
      - Don't
         - Alexa: Acquiring your playlist.
   - **If there are more than two options**, present the user with the options and ask which they would like.
      - Do
         - Alexa: I can `tell you a story, recite a rhyme, or sing a song`. `Which would you like?`
      - Don't
         - Alexa: Do you want me to tell you a story, recite a rhyme, or sing you a song?

3. Be contextually relevant
   - List options in order `from most to least` contextually relevant to make it easier for the user to understand.
   - Avoid giving the user options in an order that changes the subject of the conversation, then returns to it again.
   - This helps the user understand and verbalize their choices better without spending mental time and energy figuring out what's most relevant to them.
      - **Do**
        - Alexa: That show plays again tomorrow at 9 PM. `I can` tell you when a new episode is playing, when another show is playing, or you can do something else. `Which would you like?`
      - **Don't**
        - Alexa: That show plays again tomorrow at 9 PM. `You can find out when` another show is playing, `find out when` a new episode of this show is playing, or do something else. What would you like to do?

4. Be brief
   - Reduce the number of steps to complete a task
   - keep the conversation brief.
   - Simplify messages to their essence wherever possible.
      - **Do**
        - Alexa: Ready to start the game?
      - **Don't**
        - Alexa: All right then, are you ready to get started on a new game?

5. Write for engagement to increase retention
   - Alexa skills should be built to last and grow with the user over time.
   - Your skill should provide a delightful user experience, whether it's the first time a user invokes the skill or the 100th.
   - Design the skill to phase out information that experienced users will learn over time.
   - Give fresh dialog to repeat users so the skill doesn't become tiresome or repetitive.
      - **Do**
        - First use:
          - Alexa: Thanks for subscribing to Imaginary Radio. You can listen to a live game by saying a team name, like Seattle Seahawks, location, like New York, or league, like NFL. You can also ask me for a music station or genre. W`hat would you like to listen to?`
        - Return use:
          - Alexa:`Welcome back` to Imaginary Radio. Want to keep listening to the Kids Jam station?
      - **Don't**
         - First use:
           - Alexa: Thanks for subscribing to ABC Radio. What do you want to listen to?
         - Return use:
           - Alexa: Welcome back. What do you want to listen to?




















.
