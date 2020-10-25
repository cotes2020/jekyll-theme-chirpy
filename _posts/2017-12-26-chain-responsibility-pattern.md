---
title: Chain of Responsibility Pattern
date: 2017-12-26T16:58:08+01:00
author: Wolfgang Ofner
categories: [Design Pattern]
tags: ['C#', software architecture]
---
I think that the chain of responsibility pattern is pretty easy to learn. It is not used too often but it is very useful when sending messages to a receiver where the sender doesn&#8217;t care too much about which receiver handles the message.

## Definition

Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle the request. Chain the receiving objects and pass the request along the chain until an object handles it. (<a href="http://www.dofactory.com/net/chain-of-responsibility-design-pattern" target="_blank" rel="noopener">Gang of Four</a>)

## Real world example

A real world example for the chain of responsibility is the chain of command in a company. For example if an employee needs an approval for a task, he gives the report to his manager. If the manager can&#8217;t approve the report, for example because the costs surpass his authority, he gives the report to his manager until a manger with enough authority is found or until the report is rejected.

The normal employee does neither care nor know who gets the report. He only knows his manager who knows his manager and so on.

## Benefits

The goals of the chain of responsibility pattern are:

  * reduction of the coupling
  * dynamically manage message handlers
  * end of chain behavior can be defined as needed

## UML Diagram

[<img loading="lazy" class="aligncenter wp-image-496" src="/assets/img/posts/2017/12/Chain-of-Responsibility-pattern-UML-diagram.jpg" alt="Chain of Responsibility pattern UML diagram" width="700" height="491" />](/assets/img/posts/2017/12/Chain-of-Responsibility-pattern-UML-diagram.jpg)

The client and concrete handler link together to form the chain of responsibility. The client could be the employee who needs his report approved and the handler are different managers like team leader, area manager and CEO.

## Implementation without the Chain of Responsibility Pattern

I will now implement the real world example which I mentioned before. You can find the source code on <a href="https://github.com/WolfgangOfner/WithoutChainOfResponsibility" target="_blank" rel="noopener">GitHub</a>.

The user can enter an amount and then different managers are asked for their approval. If the amount is even too high for the CEO, the approval is denied. The problem with this approach is that all the business logic happens in the main method. Lets say the employee goes to his manager and he can&#8217;t approve the amount. He then sends the report back and tells the employee to go to his manager instead of passing the report directly.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2017/12/Finding-a-manager-to-approve-the-report.jpg"><img aria-describedby="caption-attachment-436" loading="lazy" class="size-full wp-image-436" src="/assets/img/posts/2017/12/Finding-a-manager-to-approve-the-report.jpg" alt="Finding a manager to approve the report" /></a>
  
  <p>
    Finding a manager to approve the report
  </p>
</div>

All the managers are stored in the employees list and the normal employee has to ask the first manager for the approval. If the approval can&#8217;t be given, the next manager has to be ask, and so on. Next I want to implement the chain of responsibility which cleans up the code and moves to business logic closer to the manager.

## Implementation of the Chain of Responsibility Pattern

To implement the chain of responsibility pattern I reuse the code from the previous example but I will clean it up a bit.

First I implement the ExpenseHandler which handles the approving process. If the current manager is not able to approve the amount, the costs are given to the next manager in line to approve it. To be able to do that, the managers of the chain have to be registered. The register process replaces the adding of the managers to the list of the previous example.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2017/12/Create-chain-of-responsibility.jpg"><img aria-describedby="caption-attachment-437" loading="lazy" class="size-full wp-image-437" src="/assets/img/posts/2017/12/Create-chain-of-responsibility.jpg" alt="Create chain of responsibility" /></a>
  
  <p>
    Create chain of responsibility
  </p>
</div>

To get an approval of the report, the only thing Tom has to do is call the approve method with the expenses as parameter. The ExpenseHandler will take care of sending the message to the right manager. As a result of this, the code looks cleaner and is easier to read.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2017/12/Get-approval-of-the-report.jpg"><img aria-describedby="caption-attachment-438" loading="lazy" class="size-full wp-image-438" src="/assets/img/posts/2017/12/Get-approval-of-the-report.jpg" alt="Get approval of the report" /></a>
  
  <p>
    Get approval of the report
  </p>
</div>

### Improving the code

With this changes, the code already works. But the code produces a null reference exception if the costs can&#8217;t be approved by anyone. The reason for this exception is that the handler calls the next manager until the costs are approved. If the handler doesn&#8217;t get an approval by the last manager, it calls the approve method on the next manager which doesn&#8217;t exist (and therefore is null). There are two solutions to this problem. Either I could implement a null check or I could implement another handler which handles the end of chain operation.

I implemented another handler called EndOfChainExpenseHandler which handles the end of the chain for me. If this handler is called to approve the report, it denies the report, because no manager could approve the report. Furthermore it is not impossible to add a manager as next because the EndOfChainExpenseHandler is always last in the chain. To add the EndOfChainExpenseHandler to the chain I only have to add it as next when I register another manager. So a new manager has the EndOfChainExpenseHandler is next which prevents the program from running into a null reference exception and also enables me to do whatever I want when the end of the chain is reached.

## When to use the Chain of Responsibility Pattern

You should use this pattern if:

  * You have more than one handler for a message
  * The appropriate handler is not known to the sender
  * The set of handlers can be dynamically defined

## Conclusion

In this example I showed how to use the chain of responsibility pattern to decouple the business logic from the main method and how to send the message to different receiver, which the sender doesn&#8217;t even know. Additionally I implemented another handler, which takes care of the case that no receiver could handle the message.

The source code to the example can be found on <a href="https://github.com/WolfgangOfner/ChainOfResponsibility" target="_blank" rel="noopener">GitHub</a>.

&nbsp;