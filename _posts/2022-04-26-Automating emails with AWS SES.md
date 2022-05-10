---
title: Automating emails with AWS SES
author: Pig
date: 2022-04-23 16:32:00 +0800
categories: [Projects]
tags: [Catty]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover: assets/img/post_images/catty_cover2.png
---

It’s pretty common for services to automatically send out welcome emails on sign-up. Given the positive influence these emails had on understanding the service I just signed up for, I decided to try adding on one to Catty as well.

&nbsp;

## Set up SES and verify an email address

---

SES only allows emails to be sent from verified sources to prevent impersonating random email accounts. So before working with email templates, a verified identity must be registered first.

Creating a identity is simple. After navigating to SES on the AWS console, under the *verified identities* tab, click on *create identity*. Then select identity type as *email address* and input the desired sender email and save.

This should send a verification email to the registered email account, and just follow the procedures described in the email.

&nbsp;
## Creating email template on SES

---

First, I wrote the welcome email template to be sent to new users with HTML.

<div style="text-align: left" >
  <img src="/assets/img/post_images/email.png" />
</div>

The HTML document can be registered to SES via either AWS SDK or AWS CLI. Using AWS SDK means I have to create a new Lambda function to programmatically call the API, which seemed like too much extra work, I decided to use the CLI instead. Perhaps committing to using AWS SDK can come later when the service expands and new templates need to be added often.

Install AWS CLI
[https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

Save AWS account credentials on CLI

[https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html)

According to the CLI documentation, *create-template* command takes the following as inputs:

```jsx
{
  "TemplateName": "string", /*Name of template*/
  "SubjectPart": "string", /*Email title*/
  "TextPart": "string", /*Email content as text, in the case where HTML is not supported*/
  "HtmlPart": "string" /*Email HTML content as ESCAPED string*/
}
```

For escaping the HTML content, I used the following online tool.
[https://www.freeformatter.com/json-escape.html](https://www.freeformatter.com/json-escape.html)

After saving the inputs as a .json file and executing the command

```jsx
aws --profile myprofile ses create-template --cli-input-json file://path\to\json\file
```

I received the success message and could see the template uploaded on the AWS console.

&nbsp;
## Sending templated email on AWS CLI

---

Before implementing a Lambda function to send emails, it’s always good to check if the template works as intended with the CLI.

Similar to the create-template process, the CLI documentation states that send-templated-email mainly requires the following inputs.

```jsx
"Source": "string", /*Sender email verified on SES*/
"Template": "string", /*Name of template to use*/
"Destination": {  /*Reciever email list*/
	"ToAddresses": ["string", ...],
  "CcAddresses": ["string", ...],
  "BccAddresses": ["string", ...]
},
"TemplateData": "string" /*JSON string mapping parameters referenced in the template.
												   The parameters MUST account for every variable in the template
                           Or the email will not get through*/
```

Again, save the following inputs as .json and execute the following command

```jsx
aws --profile myprofile ses send-templated-email --cli-input-json file://path\to\json\file
```

and I received a email identifier and I should see a new email on the receiver account, but nothing seemed to have come through!

After scouring online for possible problems, apparently there might be some problems with the declaring the styles separately in the HTML...

&nbsp;
## Updating email templates

---

After changing everything in the HTML to inline styling, update the .json file used for creating template with the new HTML string.

This time however, calling the create-template command again with the same file will cause an error saying that the template name already exists.

Instead, the update-template command should be used like the following

```jsx
aws --profile myprofile ses update-template --cli-input-json file://path\to\json\file
```

or delete the existing template with

```jsx
aws --profile myprofile ses delete-template --template-name templatename
```

and use create-template to recreate it again.

Retrying the email sending process on CLI and checking that it sends the desired email to the receiver account, it’s time to create an API with Lambda for the service to call.

&nbsp;
## Sending templated email with AWS SDK

---

Set up the usual serverless framework and create the function to send emails through SES

```jsx
export async function send(_data: string) {
    const data = JSON.parse(_data) as emailParams;
    const sendParams = {
        Content: {
            Template: {
                TemplateData: JSON.stringify(data.templateParams),
                TemplateName: data.template
            }
        },
        Destination: {
            ToAddresses: data.destination
        },
        FromEmailAddress: "sender@email.com"
    };
    const sendResult = await ses.sendEmail(sendParams).promise()
    .then(
        data => {
            return data;
        }
    ).catch(
        err => {
            throw err;
        }
    );
    return sendResult;
};
```

and the API handler

```jsx
export async function sendEmail(event, context) {
    console.log("event received", event);
    try{
        const res = await ses.send(event.body);
        return response(res, 200);
    } catch(e) {
        return response(e, 400);
    }
}
```

```jsx
sendEmail:
    handler: handler.sendEmail
    events:
      - http:
          path: /email
          method: post
          cors: true
```

And it’s complete! Now Catty service can call this API to send welcome emails to new users on sign-up. The API currently receives template name to account for multiple types of emails required to be sent in the future.