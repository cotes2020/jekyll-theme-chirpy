---
title: Develop an Azure App Service Logic App
date: 2018-06-28T22:20:26+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure Logic Apps is a fully managed iPaaS (integration Platform as Service) that helps you simplify and implement scalable integrations and workflows from the cloud. When you create a Logic App, you start out with a trigger, for example, &#8220;When a tweet contains a certain hashtag&#8221;, and then you act on that trigger with many combinations of actions, condition logic, and conversions.

There is a huge amount of built-in connectors like Twitter, Office 365, Azure Blob Storage or Salesforce.

## Create a Logic App connecting SaaS services

The probably biggest strengths of Logic Apps is its ability to connect a large number of SaaS service to create your own custom workflows. In the following demo, I will connect Twitter with an outlook mailbox to email certain tweets as they arrive.

To create a Logic App, follow these steps:

  1. In the Azure portal click on +Create a resource, search for Logic App and click Create.
  2. Provide a name, subscription, resource group and location.
  3. Click Create.

<div id="attachment_1316" style="width: 315px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Create-a-Logic-App.jpg"><img aria-describedby="caption-attachment-1316" loading="lazy" class="size-full wp-image-1316" src="/wp-content/uploads/2018/06/Create-a-Logic-App.jpg" alt="Create a Logic App" width="305" height="513" /></a>
  
  <p id="caption-attachment-1316" class="wp-caption-text">
    Create a Logic App
  </p>
</div>

After the Logic App is created, open it to view the Logic Apps Designer. This is where you design or modify your Logic App. You can select from a series of commonly used triggers, or from several templates you can use as a starting point.

### Create your own template

Following, I will create a template from scratch:

  1. Select Blank Logic App under Templates.
  2. All Logic Apps start with a trigger. Select Twitter from the list.
  3. Click Sign in to connect Twitter with your account and authorize the Logic App to access your account.
  4. After you are logged in, enter your search text to return certain tweets (for example #Azure), and select an interval and frequency for how often to check for items.

<div id="attachment_1309" style="width: 642px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Connect-the-Logic-App-with-Twitter.jpg"><img aria-describedby="caption-attachment-1309" loading="lazy" class="size-full wp-image-1309" src="/wp-content/uploads/2018/06/Connect-the-Logic-App-with-Twitter.jpg" alt="Connect the Logic App with Twitter" width="632" height="304" /></a>
  
  <p id="caption-attachment-1309" class="wp-caption-text">
    Connect the Logic App with Twitter
  </p>
</div>

<ol start="5">
  <li>
    The next step is to add another action by clicking + New step and select Add an action.
  </li>
  <li>
    Search for Gmail and select Gmail &#8211; send email.
  </li>
  <li>
    In the configuration window enter the recipient of the email, a subject, and the body. In the body, you could add for example the name of the person who tweeted and the tweet text.
  </li>
</ol>

<div id="attachment_1319" style="width: 638px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Configure-the-Gmail-connector-to-send-an-email.jpg"><img aria-describedby="caption-attachment-1319" loading="lazy" class="size-full wp-image-1319" src="/wp-content/uploads/2018/06/Configure-the-Gmail-connector-to-send-an-email.jpg" alt="Configure the Gmail connector to send an email" width="628" height="612" /></a>
  
  <p id="caption-attachment-1319" class="wp-caption-text">
    Configure the Gmail connector to send an email
  </p>
</div>

<ol start="8">
  <li>
    Save the template.
  </li>
  <li>
    You can test it immediately by clicking Run.
  </li>
  <li>
    Send a tweet and you should get an email to the configured email.
  </li>
</ol>

<div id="attachment_1320" style="width: 523px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/The-received-email-from-the-Gmail-connector.jpg"><img aria-describedby="caption-attachment-1320" loading="lazy" class="size-full wp-image-1320" src="/wp-content/uploads/2018/06/The-received-email-from-the-Gmail-connector.jpg" alt="The received email from the Gmail connector" width="513" height="165" /></a>
  
  <p id="caption-attachment-1320" class="wp-caption-text">
    The received email from the Gmail connector
  </p>
</div>

<ol start="11">
  <li>
    If you set everything up correctly, you should receive an email.
  </li>
</ol>

First, I wanted to use Outlook 365 to send emails but I couldn&#8217;t log in although I tried three different accounts. Gmail only worked with the second account I tried. I couldn&#8217;t find anything about that on Google nor do I have any idea why it didn&#8217;t work.

## Create a Logic App with B2B capabilities

Logic Apps support business-to-business (B2B) workflows through the Enterprise Integration pack. This allows organizations to exchange messages electronically, even if they use different protocols and formats. Enterprise integration allows you to store all your artifacts in one place, within your integration account, and secure message through encryption and digital signature.

### Create an integration account

To create an integration account, follow these steps:

  1. In the Azure portal click on +Create a resource, search for Integration Account and click Create.
  2. Provide a name, subscription, resource group, pricing tier and location. Note that your integration account and Logic App must be in the same location, otherwise you can&#8217;t link them.
  3. Click Create.

<div id="attachment_1322" style="width: 318px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Create-a-new-Integration-account.jpg"><img aria-describedby="caption-attachment-1322" loading="lazy" class="size-full wp-image-1322" src="/wp-content/uploads/2018/06/Create-a-new-Integration-account.jpg" alt="Create a new Integration account" width="308" height="455" /></a>
  
  <p id="caption-attachment-1322" class="wp-caption-text">
    Create a new integration account
  </p>
</div>

### Add partners to your integration account

Messages between partners are called agreement. You need at least two partners in your integration account to create an agreement. Your organization must be the host partner, and the other partner(s) are guests. Guest partners can be outside organizations or even a different department in your organization.

To add a partner to your integration account, follow these steps:

  1. Open your integration account and click on the Partners blade under the Settings menu.
  2. On the Partners blade click on +Add and provide a name, qualifier, and value to help identify documents that transfer through your apps. As the qualifier, you have to select, AS2Identity, otherwise you can&#8217;t create an AS2 agreement in the next section.
  3. Click OK.

<div id="attachment_1326" style="width: 327px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Add-a-new-partner.jpg"><img aria-describedby="caption-attachment-1326" loading="lazy" class="size-full wp-image-1326" src="/wp-content/uploads/2018/06/Add-a-new-partner.jpg" alt="Add a new partner" width="317" height="249" /></a>
  
  <p id="caption-attachment-1326" class="wp-caption-text">
    Add a new partner
  </p>
</div>

I added another partner and both are added to the list on the Partners blade.

<div id="attachment_1325" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Both-partners-are-listed.jpg"><img aria-describedby="caption-attachment-1325" loading="lazy" class="wp-image-1325" src="/wp-content/uploads/2018/06/Both-partners-are-listed.jpg" alt="Both partners are listed" width="700" height="175" /></a>
  
  <p id="caption-attachment-1325" class="wp-caption-text">
    Both partners are listed
  </p>
</div>

### Add an agreement

After the partners are associated with the integration account, you have to allow them to communicate using industry standard protocols through agreements. These agreements are based on the type of information exchanged, and through which protocol or standard they will communicate: AS2, X12 or EDIFACT.

To create an AS2 agreement, follow these steps:

  1. Open your integration account and click on the Agreements blade under the Settings menu.
  2. On the Agreementsblade click on +Add and provide a name and select AS2 as agreement type.
  3. Select your previously created host and guest partner and their identity.
  4. Click OK

<div id="attachment_1327" style="width: 321px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Create-a-new-AS2-agreement.jpg"><img aria-describedby="caption-attachment-1327" loading="lazy" class="size-full wp-image-1327" src="/wp-content/uploads/2018/06/Create-a-new-AS2-agreement.jpg" alt="Create a new AS2 agreement" width="311" height="561" /></a>
  
  <p id="caption-attachment-1327" class="wp-caption-text">
    Create a new AS2 agreement
  </p>
</div>

### Link your Logic App to your Enterprise Integration account

After the integration account is set up, you can link it with your Logic App to create B2B workflows. As mentioned before, the integration account and Logic App must be in the same region to be linked.

To link them, follow these steps:

  1. Open your Logic App and click on the Workflow settings blade under the Settings menu.
  2. On the Workflow settings blade, select your integration account in the &#8220;Select an Integration account&#8221; drop-down menu.
  3. Click Save.

<div id="attachment_1328" style="width: 578px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Link-your-Logic-App-and-integration-account.jpg"><img aria-describedby="caption-attachment-1328" loading="lazy" class="size-full wp-image-1328" src="/wp-content/uploads/2018/06/Link-your-Logic-App-and-integration-account.jpg" alt="Link your Logic App and integration account" width="568" height="609" /></a>
  
  <p id="caption-attachment-1328" class="wp-caption-text">
    Link your Logic App and integration account
  </p>
</div>

### Use B2B features to receive data in Logic Apps

After the Logic App and integration account are linked, you can create a B2B workflow using the Enterprise Integration Pack., following these steps:

  1. Open your Logic App and click on the Logic App Designer blade under the Development Tools menu.
  2. On the Logic App Designer click on Blank Logic App under templates.
  3. Search for http and select Request &#8211; When a HTTP request is received.
  4. Click on + New step, then on Add an action and search for AS2.
  5. Select AS2 &#8211; Decode AS2 message.
  6. Provide a connection name, select your integration account and click Create.

<div id="attachment_1330" style="width: 642px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Add-the-decode-AS2-message-form.jpg"><img aria-describedby="caption-attachment-1330" loading="lazy" class="size-full wp-image-1330" src="/wp-content/uploads/2018/06/Add-the-decode-AS2-message-form.jpg" alt="Add the decode AS2 message form" width="632" height="456" /></a>
  
  <p id="caption-attachment-1330" class="wp-caption-text">
    Add the decode AS2 message form
  </p>
</div>

<ol start="7">
  <li>
    In the body section select the Body from the HTTP request and as Headers select Headers. If you can&#8217;t select Headers, click on the small button on the right to switch to text mode.
  </li>
</ol>

<div id="attachment_1331" style="width: 658px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Setting-the-Decode-AS2-Message-body-and-headers-information-form-in-the-Logic.jpg"><img aria-describedby="caption-attachment-1331" loading="lazy" class="size-full wp-image-1331" src="/wp-content/uploads/2018/06/Setting-the-Decode-AS2-Message-body-and-headers-information-form-in-the-Logic.jpg" alt="Setting the Decode AS2 Message body and headers information form in the Logic" width="648" height="397" /></a>
  
  <p id="caption-attachment-1331" class="wp-caption-text">
    Setting the Decode AS2 Message body and headers information form in the Logic
  </p>
</div>

<ol start="8">
  <li>
    Click on + New step, then on Add an action and search for X12.
  </li>
  <li>
    Select X12 &#8211; Decode X12 Message.
  </li>
  <li>
    Enter a connection name, select your integration account and click Create.
  </li>
</ol>

<div id="attachment_1332" style="width: 646px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Enter-a-connection-name-and-select-your-integration-account.jpg"><img aria-describedby="caption-attachment-1332" loading="lazy" class="size-full wp-image-1332" src="/wp-content/uploads/2018/06/Enter-a-connection-name-and-select-your-integration-account.jpg" alt="Enter a connection name and select your integration account" width="636" height="688" /></a>
  
  <p id="caption-attachment-1332" class="wp-caption-text">
    Enter a connection name and select your integration account
  </p>
</div>

<ol start="11">
  <li>
    Since the message content is JSON-formated and base64-encoded, you must specify an expression as the input. Enter the following expression in the X12 flat file message to decode textbox: @base64ToString(body(&#8216;Decode_AS2_Message&#8217;)?[&#8216;AS2Message&#8217;]?[&#8216;Content&#8217;])
  </li>
</ol>

<div id="attachment_1335" style="width: 638px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Decode-the-AS2-message.jpg"><img aria-describedby="caption-attachment-1335" loading="lazy" class="size-full wp-image-1335" src="/wp-content/uploads/2018/06/Decode-the-AS2-message.jpg" alt="Decode the AS2 message" width="628" height="205" /></a>
  
  <p id="caption-attachment-1335" class="wp-caption-text">
    Decode the AS2 message
  </p>
</div>

<ol start="12">
  <li>
    Click on + New step, then on Add an action and search for response.
  </li>
  <li>
    Select Request &#8211; Response.
  </li>
  <li>
    Paste @base64ToString(body(&#8216;Decode_AS2_message&#8217;)?[&#8216;OutgoingMdn&#8217;]?[&#8216;Content&#8217;]) into the Body textbox.
  </li>
</ol>

<div id="attachment_1334" style="width: 648px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Configure-the-response.jpg"><img aria-describedby="caption-attachment-1334" loading="lazy" class="size-full wp-image-1334" src="/wp-content/uploads/2018/06/Configure-the-response.jpg" alt="Configure the response" width="638" height="506" /></a>
  
  <p id="caption-attachment-1334" class="wp-caption-text">
    Configure the response
  </p>
</div>

<ol start="15">
  <li>
    Click Save. If you configured everything right, your template will be saved, if not you get an error message, which is usually helpful.
  </li>
</ol>

## Create a Logic App with XML capabilities

Often, businesses send and receive data between organizations in the XML format. Schemas are used to transform data from one format to another. Transforms are also known as maps, which consist of source and target XML schemas. Linking your Logic App with an integration account enables your Logic App to use Enterprise Integration Pack XML capabilities.

The XML features in the Enterprise Integration Pack are:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        XML feature
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        XML validation
      </td>
      
      <td>
        XML validation is used to validate incoming and outgoing XML messages against a specific schema.
      </td>
    </tr>
    
    <tr>
      <td>
        XML transform
      </td>
      
      <td>
        XML transform is used to convert data from one format to another.
      </td>
    </tr>
    
    <tr>
      <td>
        Flat file encoding/decoding
      </td>
      
      <td>
        Flat file encoding/decoding is used to encode XML content prior sending or to convert XML content to flat files.
      </td>
    </tr>
    
    <tr>
      <td>
        XPath
      </td>
      
      <td>
        XPath is used to extract specific properties form a message, using an XPath expression.
      </td>
    </tr>
  </table>
</div>

### Add schemas to your integration account

Since schemas are used to validate and transform XML messages, you must add one or more to your integration account before working with the Enterprise Integration Pack XML feature within your linked logic app.

To add a new schema, follow these steps:

  1. Open your integration account and select the Schemas blade under the under the Settings menu.
  2. On the Schemas blade, click +Add.
  3. Provide a name, select whether it is a small or large file and upload an XSD file. (You can find a sample XSD <a href="https://msdn.microsoft.com/en-us/library/dd489283.aspx" target="_blank" rel="noopener">here</a>)
  4. Click OK.

<div id="attachment_1336" style="width: 321px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Add-a-schema-to-your-integration-account.jpg"><img aria-describedby="caption-attachment-1336" loading="lazy" class="size-full wp-image-1336" src="/wp-content/uploads/2018/06/Add-a-schema-to-your-integration-account.jpg" alt="Add a schema to your integration account" width="311" height="264" /></a>
  
  <p id="caption-attachment-1336" class="wp-caption-text">
    Add a schema to your integration account
  </p>
</div>

### Add maps to your Integration Account

If your Logic App should transform data from one format to another one, you have to add a map (schema) first.

To add a new schema, follow these steps:

  1. Open your integration account and select the Maps blade under the under the Settings menu.
  2. On the Maps blade, click +Add.
  3. Enter a name and upload an XSLT file (You can find a sample XSLT <a href="https://docs.microsoft.com/en-us/previous-versions/windows/desktop/ms765388(v=vs.85)" target="_blank" rel="noopener">here</a>).
  4. Click OK.

<div id="attachment_1337" style="width: 326px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Add-a-map-to-your-integration-account.jpg"><img aria-describedby="caption-attachment-1337" loading="lazy" class="size-full wp-image-1337" src="/wp-content/uploads/2018/06/Add-a-map-to-your-integration-account.jpg" alt="Add a map to your integration account" width="316" height="254" /></a>
  
  <p id="caption-attachment-1337" class="wp-caption-text">
    Add a map to your integration account
  </p>
</div>

### Add XML capabilities to the linked Logic App

After adding an XML schema and map to the integration account, the application is ready to use the Enterprise Integration Pack&#8217;s XML validation, XPath Extract, and Transform XML operations in Logic App.

Follow these steps to use XML capabilities in your Logic App:

  1. Open the Logic App Designer in your Logic App.
  2. Select Blank Logic App under Templates.
  3. Search for http and select Request &#8211; When a HTTP request is received.
  4. Click on + New Step and select Add an action.
  5. Search for xml and select XML &#8211; XML Validation.
  6. Select Body as Content and the previously uploaded schema

<div id="attachment_1339" style="width: 654px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Add-XML-validation-to-your-Logic-App.jpg"><img aria-describedby="caption-attachment-1339" loading="lazy" class="size-full wp-image-1339" src="/wp-content/uploads/2018/06/Add-XML-validation-to-your-Logic-App.jpg" alt="Add XML validation to your Logic App" width="644" height="294" /></a>
  
  <p id="caption-attachment-1339" class="wp-caption-text">
    Add XML validation to your Logic App
  </p>
</div>

<ol start="7">
  <li>
    Click on + New step and select Add an action.
  </li>
  <li>
    Search for xml and select Transform XML &#8211; Transform XML.
  </li>
  <li>
    Select the HTTP Body as Content and your previously added map.
  </li>
</ol>

<div id="attachment_1340" style="width: 650px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Add-XML-transformation-to-your-Logic-App.jpg"><img aria-describedby="caption-attachment-1340" loading="lazy" class="size-full wp-image-1340" src="/wp-content/uploads/2018/06/Add-XML-transformation-to-your-Logic-App.jpg" alt="Add XML transformation to your Logic App" width="640" height="463" /></a>
  
  <p id="caption-attachment-1340" class="wp-caption-text">
    Add XML transformation to your Logic App
  </p>
</div>

Unfortunately the Azure Portal displayed various errors at this point and Microsoft&#8217;s documentation was outdated.

## Trigger a Logic App from another app

The most common type of triggers are those that create HTTP endpoints. Triggers based on HTTP endpoints tend to be more widely used due to the simplicity of making REST-based calls from practically any web-enabled development platform.

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Trigger
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Request
      </td>
      
      <td>
        The HTTP endpoint responds to incoming HTTP request to start the Logic App&#8217;s workflow in real time. Very versatile, in that it can be called from any web-based application external webhook events, even from another Logic App with a request and response action.
      </td>
    </tr>
    
    <tr>
      <td>
        HTTP Webhook
      </td>
      
      <td>
        A webhook is an event-based trigger that does not rely on polling for new items. Register subscribe and unsubscribe methods with a callback URL are used to trigger the Logic App. Whenever an external or app or service makes an HTTP POST to the callback URL, the Logic App fires, and includes any data passed into the request.
      </td>
    </tr>
    
    <tr>
      <td>
        API Connection Webhook
      </td>
      
      <td>
        The API connection trigger is similar to the HTTP trigger in its basic functionality. However, the parameters for identifying the action are slightly different
      </td>
    </tr>
  </table>
</div>

### Create an HTTP endpoint for your Logic App

To create an HTTP endpoint to receive an incoming request for a Request Trigger, follow these steps:

  1. Open the Logic App Designer in your Logic App.
  2. Select Blank Logic App under Templates.
  3. Search for http and select Request &#8211; When a HTTP request is received.
  4. Optionally enter a JSON schema for the payload that you expect to be sent to the trigger. This schema can be added to the Request Body JSON Schema field. To generate the schema, select the Use sample payload to generate schema link at the bottom of the form. This displays a dialog where you can type in or paste a sample JSON payload. The advantage of having a schema defined is that the designer will use the schema to generate tokens that your logic app can use to consume, parse, and pass data from the trigger through your workflow.

<div id="attachment_1341" style="width: 663px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Enter-a-JSON-schema.jpg"><img aria-describedby="caption-attachment-1341" loading="lazy" class="size-full wp-image-1341" src="/wp-content/uploads/2018/06/Enter-a-JSON-schema.jpg" alt="Enter a JSON schema" width="653" height="447" /></a>
  
  <p id="caption-attachment-1341" class="wp-caption-text">
    Enter a JSON schema
  </p>
</div>

<ol start="5">
  <li>
    After saving, the HTTP Post URL is generated on the Receiver trigger. This is the URL your app or service uses to trigger your logic app. The URL contains a Shared Access Signature (SAS) token used to authenticate the incoming requests.
  </li>
</ol>

<div id="attachment_1342" style="width: 648px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/The-generated-HTTP-POST-URL-for-the-trigger.jpg"><img aria-describedby="caption-attachment-1342" loading="lazy" class="size-full wp-image-1342" src="/wp-content/uploads/2018/06/The-generated-HTTP-POST-URL-for-the-trigger.jpg" alt="The generated HTTP POST URL for the trigger" width="638" height="102" /></a>
  
  <p id="caption-attachment-1342" class="wp-caption-text">
    The generated HTTP POST URL for the trigger
  </p>
</div>

## Create custom and long-running actions

You can create your own APIs that provide custom actions and triggers. Because these are web-based APIs that use REST API endpoints, you can build them in any language you like.

API apps are preferred because to host your APIs since they will make it easier to build, host, and consume your APIs used by Logic Apps. Another recommendation is to provide an OpenAPI (Swagger) specification to describe your REST API endpoints, their operations, and parameters. This makes it much easier to reference your custom API from a Logic App workflow because all of the endpoints are <span class="fontstyle0">selectable</span> within the designer. you can use libraries like Swashbuckle to automatically generate the OpenAPI file for you. You can read more about Azure&#8217;s API Service in [Design Azure App Services API Apps](https://www.programmingwithwolfgang.com/design-azure-app-service-api-apps/).

If your custom API has long-running tasks to perform, it is more than likely that your Logic App will timeout waiting for the operation to complete. This is because the Logic App will only wait around two minutes before timing out. If your task takes several minutes or even hours to complete, you need to implement a REST-based async pattern on your API. These types of patterns are already fully supported <span class="fontstyle0">natively</span> by the Logic Apps workflow engine, so you don&#8217;t need to worry about implementing it there.

### Long-running action patterns

The asynchronous polling pattern and the asynchronous webhook pattern allow your Logic App to wait for long-running tasks to finish.

#### Asynchronous polling

The asynchronous polling pattern works the following way:

  1. When your API receives the initial request to start work, it starts a new thread with the long-running task, and immediately returns an HTTP Response 202 Accepted code with a location header. This immediate response prevents the request from timing out and causes the workflow engine to start polling for changes.
  2. The location header points to the URL for the Logic Apps to check the status of the long-running job. By default, the engine checks every 20 seconds, but you can also add a &#8220;Retry-after&#8221; header to specify the number of seconds until the next poll.
  3. After the allotted time of 20 seconds, the engine poll the URL on the location header. If the long-running job is still going, you should return another 202 Accepted with a location header. If the job was completed, return a 200 OK code with any relevant data. The Logic App will continue its workflow with this data.

#### Asynchronous Webhooks

The asynchronous webhook pattern works by creating two endpoints on your API controller the following way:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Endpoint
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Subscribe
      </td>
      
      <td>
        The Logic App engine calls the subscribe endpoint defined in the workflow action for your API. Included in this call is a callback URL created by the Logic App that your API stores for when work is complete. When your long-running task is complete, your API calls back with an HTTP POST method to the URL, along with any returned content and headers, as input to the Logic App.
      </td>
    </tr>
    
    <tr>
      <td>
        Unsubscribe
      </td>
      
      <td>
        The unsubscribe endpoint is called any time the logic app run is canceled. When your API receives a request to this endpoint, it should unregister the callback URL and stop any running processes.
      </td>
    </tr>
  </table>
</div>

## Monitor Logic Apps

To monitor your Logic Apps, you can use out-of-the-box tools within your Log App to detect any issues it may have. For real-time event monitoring and richer debugging, you can enable diagnostics and send events to OMS with Log Analytics, or to other services, such as Azure Storage or Event Hubs.

Select Metrics under the Monitoring menu and select the metrics you want to look at such as runs started or runs succeeded. You can configure the type of event and the time span you want to look at.

<div id="attachment_1343" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/06/Monitoring-your-Logic-App.jpg"><img aria-describedby="caption-attachment-1343" loading="lazy" class="wp-image-1343" src="/wp-content/uploads/2018/06/Monitoring-your-Logic-App.jpg" alt="Monitoring your Logic App" width="700" height="493" /></a>
  
  <p id="caption-attachment-1343" class="wp-caption-text">
    Monitoring your Logic App
  </p>
</div>

On the Overview blade, you can see the history of the runs, the status, start time and duration.

## Conclusion

In this post, I showed different operations which can be done with Logic Apps. Working with XML messages and transformation didn&#8217;t work well. Additionally the documentation from Microsoft is outdated and not helpful. My favorite part was using the Logic App to interact with Twitter and send an email when a certain hashtag was used. With Azure&#8217;s built-in connectors it would be easy to add image recognition or text recognition using AI. For example, I could analyze the images and if they are inappropriate, notify someone via email.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="https://www.programmingwithwolfgang.com/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.