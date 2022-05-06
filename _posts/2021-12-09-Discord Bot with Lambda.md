---
title: Discord Bot with Lambda
author:
  name: Pig
  link: https://github.com/kimdh98
date: 2021-12-09 13:21:00 +0800
categories: [Projects]
tags: [Discord, Lambda]
---

## Prerequisites

- Set up an AWS account
- Knowledge on setting up basic Lambda function with Serverless framework
- Experience with Javscript/Typescript
- Set up a bot profile on discord

## Goal

Set up a discord bot interactions endpoint with AWS Lambda and API Gateway to activate/deactivate Minecraft server hosted on EC2 with commands

## Experiment Journal

After reminiscing with a friend about the good old days of playing minecraft together in our server, we decided to start up a fresh new server and get back into it(a~~ll roads lead down to minecraft..~~). 

![Fun times...](/assets/img/post_images/discord1.jpg)

Fun times...

This time however, hosting the server on my laptop and keeping it on standby for my friend to join wasn’t viable since I was often on the move. That leaves either subscribing to minecraft realms, or hosting it on a cloud computing provider like AWS.

After looking through [this](https://dev.to/julbrs/how-to-run-a-minecraft-server-on-aws-for-less-than-3-us-a-month-409p) excellent guide on setting up a minecraft server on EC2 by Julien Bras, it seemed like hosting on EC2 was the cheaper option for 2~3 players. Unfortunately, since t3.tiny EC2 instance available under AWS free tier was not powerful enough for a minecraft server and running other instance types 24/7 would cost more than just subscribing to realms, I needed to make an activate and kill switch for the server.

Julien’s guide used SES detecting incoming emails to start up the server and automatically shut down after 8 hours, but since we were going to set up a discord group anyway, I felt that it would be more convenient to have a discord bot.

**Preparing the bot interactions endpoint**

After setting up a bot profile on [discord](https://discord.com/developers/applications), I needed to make a backend application on Lambda to serve the bot. On the [official discord documentation](https://discord.com/developers/docs/interactions/receiving-and-responding), there are two base prerequisites the server needs to fulfill for the api endpoint to be registered as a bot interaction endpoint. Attempting to register the endpoint before meeting the requirements will resulted in an error in discord.

![Untitled](/assets/img/post_images/discord7.png)

1. Respond to a PING request
    
    ![Official discord documentation on PING request](/assets/img/post_images/discord4.png)
    
    Official discord documentation on PING request
    
    To test if the server is ready to respond to the bot’s requests, it will send out a PING request to the server, signified as a type 1(PING) request. Responding to this request is simple, the server just needs to return a similar response of type 1(PONG). I’ve done this by defining a special type 1 response in my response wrapper function.
    
    ```jsx
    export const response = (data: string, type: number) => {
        if (type == 1) {
            return {
                statusCode: 200,
                headers: {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type, Access-Control-Allow-Headers, Authorization, RefreshToken, X-Requested-With",
                    "Access-Control-Allow-Credentials": false,
                },
                body: JSON.stringify({ type: 1 })
            };
        }
        return {
            statusCode: 200,
            headers: {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type, Access-Control-Allow-Headers, Authorization, RefreshToken, X-Requested-With",
                "Access-Control-Allow-Credentials": false,
            },
            body: JSON.stringify({ 
                type: 4,
                data: {
                    tts: false,
                    content: data
                }
            })
        };
    }
    ```
    
2. Validate request signatures
    
    ![Official discord documentation on validating signatures](/assets/img/post_images/discord5.png)
    
    Official discord documentation on validating signatures
    
    As the documentation explains, this step is required by discord to ensure that our server does not allow access from anywhere except our bot. The bot will send a 'x-signature-ed25519’ in the request header, which is the encoded form of the request body and current timestamp. The request will also include the used timestamp in the headers as 'x-signature-timestamp’.
    
    Validating the signature requires the use of a NaCl library for Javascript, tweetnacl. Since NaCl library is not part of the standard libraries on Lambda, adding a layer is required. To do this, I created an independent node package containing the NaCl library with npm, and zipped it up to be used as a layer in my project directory.
    
    ![Untitled](/assets/img/post_images/discord6.png)
    
    Then, I added the following lines to serverless.yml to add this layer to deployment package
    
    ```jsx
    layers:
      serverbotLayer:
        package:
          artifact: layers/minecraft_server.zip
        compatibleRuntimes: nodejs12.x
        description: "node modules for serverbot"
    ```
    
    One last preparation step to validate incoming signature is finding out my bot’s public key, which can be found under the ‘general information’ section of the bot page. Remember to keep this key stored safely elsewhere and refer to it instead of exposing it in the code.
    
    ![제목 없음.png](/assets/img/post_images/discord2.png)
    
    Now, all that’s left is to follow discord documentation’s sample code and validate the incoming signatures.
    
    ```jsx
    export const discordHandler = async (event: lambda.APIGatewayProxyEvent, context: lambda.Context) => {
        console.log('event receieved: ', event);
        const signature = event.headers['x-signature-ed25519'];
        const timestamp = event.headers['x-signature-timestamp'];
        const body = event.body; // rawBody is expected to be a string, not raw bytes
        
        const isVerified = nacl.sign.detached.verify(
          Buffer.from(timestamp + body),
          Buffer.from(signature, 'hex'),
          Buffer.from(secrets.PUBLIC_KEY, 'hex')
        );
        const data = JSON.parse(event.body);
        if (isVerified) { 
    		//on valid signature
            try {
                switch (data.type) {
                    case 1:
                        return response("", 1); //respond to ping request
                        }
                }
            }
            catch (error) {
                return response("An error has occured: " + error, 2);
            }
        } else {
    		//invalid signature
            return {
                statusCode: 401,
                headers: {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type, Access-Control-Allow-Headers, Authorization, RefreshToken, X-Requested-With",
                    "Access-Control-Allow-Credentials": false,
                },
                body: "invalid request signature"
            };
        }
    }
    ```
    
3. Set up the endpoint with Serverless framework
    
    Register the discordHandler function on serverless.yml and deploy!
    
    ```jsx
    functions:
      discordHandler:
        handler: handler.discordHandler
        layers:
          - { Ref: ServerbotLayerLambdaLayer } //lambda layer reference
        events:
          - http:
              path: /
              method: post
    ```
    
    After the function has been uploaded to lambda, I can now register the endpoint on the bot without any errors.
    
    ![Copy and paste the endpoint url from API gateway here](/assets/img/post_images/discord3.png)
    
    Copy and paste the endpoint url from API gateway here
